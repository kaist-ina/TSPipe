import torch
import yaml
from torchvision import datasets
import dataset.datasets as small_datasets
from dataset.loader import TwoCropsTransform, GaussianBlur, Solarize
import torchvision.transforms as transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer, DummyBYOLTrainer, TSPipeBYOLTrainer
import torch
import tspipe

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    # online network
    online_network = ResNet18(**config['network'])

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head'])

    # target encoder
    target_network = ResNet18(**config['network'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    # setup 
    if config['dataset']['name'] == 'stl10':
        train_dataset = datasets.STL10(config['dataset']['path'], split='train+unlabeled', download=True,
                                    transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))
    elif config['dataset']['name'] == 'imagenet':
        train_dataset = small_datasets.ImageNet100(config['dataset']['path'], split='train', 
                                    transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))
    elif config['dataset']['name'] == 'imagenet1000':
        train_dataset = datasets.ImageNet(config['dataset']['path'], split='train', 
                                    transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))
                                    
    online_network = online_network.to_sequential()
    target_network = target_network.to_sequential()
    predictor = predictor.to_sequential()
    
    # do not send to cuda if tspipe enabled
    if config['trainer']['acceleration'] != 'tspipe':
        online_network, target_network, predictor = online_network.cuda(), target_network.cuda(), predictor.cudaa()
    
    # setup optimizer
    if config['trainer']['optimizer_type'] == 'sgd':
        optimizerClass = torch.optim.SGD
    elif config['trainer']['optimizer_type'] == 'adamw':
        optimizerClass = torch.optim.AdamW
    elif config['trainer']['optimizer_type'] == 'lars':
        optimizerClass = LARS
    else:
        raise NotImplementedError()
    optimizer = optimizerClass(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
                                
    # setup learning rate scheduler
    if config['trainer']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['trainer']['max_epochs'], eta_min=0)
    elif config['trainer']['lr_scheduler'] == None or config['trainer']['lr_scheduler'] == 'None':
        scheduler = None
    else:
        raise NotImplementedError()

    if config['trainer']['acceleration'] == 'tspipe':
        trainerClass = TSPipeBYOLTrainer
    elif config['trainer']['acceleration'] == 'dummy':
        trainerClass = DummyBYOLTrainer
    else:
        trainerClass = BYOLTrainer

    trainer = trainerClass(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          scheduler=scheduler,
                          **config['trainer'], **config['data_transforms'])
    with tspipe.profiler.TSPipeProfiler(filename='profile_out.csv'):
        trainer.train(train_dataset)

if __name__ == '__main__':
    main()
