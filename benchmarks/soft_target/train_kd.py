from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import time
from functools import partial
from itertools import chain
from typing import Iterable, Optional

import numpy as np
import torch
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset.datasets as small_datasets
from kd_losses import Logits, SoftTarget
from models.factory import create_model
from tspipe import TSPipe
from tspipe.tspipe import TSPipeMode
from utils import (AverageMeter, accuracy, count_parameters,
                   count_parameters_in_MB, create_exp_dir,
                   load_pretrained_model, save_checkpoint)

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

parser.add_argument('--tspipe-enable', action='store_true', default=False, dest='tspipe')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')  # cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')     # resnet20/resnet110
parser.add_argument('--s_name', type=str, required=True, help='name of student')     # resnet20/resnet110

# hyperparameter
parser.add_argument('--kd_mode', type=str, required=True, help='mode of kd, which can be:'
                                                               'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                               'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
parser.add_argument('--m', type=float, default=2.0, help='margin for AB')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma in Gaussian RBF for CC')
parser.add_argument('--P_order', type=int, default=2, help='P-order Taylor series of Gaussian RBF for CC')
parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')
parser.add_argument('--sf', type=float, default=1.0, help='scale factor for VID, i.e. mid_channels = sf * out_channels')
parser.add_argument('--init_var', type=float, default=5.0, help='initial variance for VID')
parser.add_argument('--att_f', type=float, default=1.0, help='attention factor of mid_channels for AFD')


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def dummy_target_update(m, online_new_param: Optional[Iterable[torch.Tensor]], 
                        target_param: Optional[Iterable[torch.nn.Parameter]] = None):
    return target_param


def main():
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    logging.info('----------- Network Initialization --------------')
    image_size = 224 if 'cifar' not in args.data_name else 32
    snet = create_model(args.s_name, num_class=args.num_class, image_size=image_size)
    checkpoint = torch.load(args.s_init, 'cpu')
    load_pretrained_model(snet, checkpoint['net'])
    # logging.info('Student: %s', snet)
    logging.info('Student param size = %fMB, %d params', count_parameters_in_MB(snet), count_parameters(snet))

    tnet = create_model(args.t_name, num_class=args.num_class, image_size=image_size)
    checkpoint = torch.load(args.t_model, 'cpu')
    load_pretrained_model(tnet, checkpoint['net'])
    tnet.eval()
    for param in tnet.parameters():
        param.requires_grad = False
    # logging.info('Teacher: %s', tnet)
    logging.info('Teacher param size = %fMB, %d params', count_parameters_in_MB(tnet), count_parameters(tnet))
    logging.info('-----------------------------------------------')

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    else:
        raise Exception('Invalid kd mode...')
    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.SGD(snet.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # define transforms
    if args.data_name == 'cifar10':
        dataset = dst.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_dataset = partial(dataset, train=True, download=True)
        test_dataset = partial(dataset, train=False, download=True)
    elif args.data_name == 'cifar100':
        dataset = dst.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        train_dataset = partial(dataset, train=True, download=True)
        test_dataset = partial(dataset, train=False, download=True)
    elif args.data_name == 'imagenet100':
        dataset = small_datasets.ImageNet100
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_dataset, test_dataset = partial(dataset, split='train'), partial(dataset, split='val')
    else:
        raise Exception('Invalid dataset name...')

    if 'cifar' in args.data_name:
        train_transform = transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        test_transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    else:
        train_transform = transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        test_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    # define data loader
    train_loader = torch.utils.data.DataLoader(
            train_dataset(root=args.img_root, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=24, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset(root=args.img_root, transform=test_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=24, pin_memory=True)

    # initialize tspipe (if needed)
    if args.tspipe:
        if not isinstance(snet, torch.nn.Sequential):
            snet = snet.to_sequential()
        if not isinstance(tnet, torch.nn.Sequential):
            tnet = tnet.to_sequential()
        
        optimizer = torch.optim.SGD(snet.parameters(),
                                    lr = args.lr,
                                    momentum = args.momentum, 
                                    weight_decay = args.weight_decay,
                                    nesterov = True)

        tspipe_trainer = TSPipe(
            snet,
            tnet,
            None,
            optimizer,
            tspipe_loss,
            dummy_target_update,
            1,
            artifact_dir = args.save_root,
            tspipe_mode=TSPipeMode.SUPERVISED_MOMENTUM,
            target_train_mode=False,
            extra_args=args
        )
        assert args.kd_mode == 'logits' or args.kd_mode == 'st'
    else:
        snet = snet.cuda()
        tnet = tnet.cuda()  

    writer = SummaryWriter()

    # warp nets and criterions for train and test
    nets = {'snet': snet, 'tnet': tnet}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    # first initilizing the student nets
    if args.kd_mode in ['fsp', 'ab']:
        logging.info('The first stage, student initialization......')
        train_init(train_loader, nets, optimizer, criterions, 50)
        args.lambda_kd = 0.0
        logging.info('The second stage, softmax training......')

    best_top1 = 0
    best_top5 = 0
    for epoch in range(1, args.epochs+1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        if args.tspipe:
            for param_group in optimizer.param_groups:
                tspipe_trainer.update_lr(param_group['lr'])
                break
            train_tspipe(tspipe_trainer, train_loader, nets, optimizer, criterions, epoch, writer)
        else:
            train(train_loader, nets, optimizer, criterions, epoch, writer)

        continue

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top5 = test(test_loader, nets, criterions, epoch)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # save model
        is_best = False
        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top5 = test_top5
            is_best = True
        logging.info('Saving models......')
        save_checkpoint({
            'epoch': epoch,
            'snet': snet.state_dict(),
            'tnet': tnet.state_dict(),
            'prec@1': test_top1,
            'prec@5': test_top5,
        }, is_best, args.save_root)
    if args.tspipe:        
        tspipe_trainer.stop()


def train_init(train_loader, nets, optimizer, criterions, total_epoch):
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.train()

    for epoch in range(1, total_epoch+1):
        adjust_lr_init(optimizer, epoch)

        batch_time = AverageMeter()
        data_time  = AverageMeter()
        cls_losses = AverageMeter()
        kd_losses  = AverageMeter()
        top1       = AverageMeter()
        top5       = AverageMeter()

        epoch_start_time = time.time()
        end = time.time()
        for i, (img, target) in enumerate(train_loader, start=1):
            data_time.update(time.time() - end)

            if args.cuda:
                img = img.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
            stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)

            cls_loss = criterionCls(out_s, target) * 0.0
            if args.kd_mode in ['fsp']:
                kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
                           criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
                           criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
            elif args.kd_mode in ['ab']:
                kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
                           criterionKD(rb2_s[0], rb2_t[0].detach()) +
                           criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
            else:
                raise Exception('Invalid kd mode...')
            loss = cls_loss + kd_loss

            prec1, prec5 = accuracy(out_s, target, topk=(1,5))
            cls_losses.update(cls_loss.item(), img.size(0))
            kd_losses.update(kd_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                           'Time:{batch_time.val:.4f} '
                           'Data:{data_time.val:.4f}  '
                           'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                           'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                           'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                           'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                           epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                           cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
                logging.info(log_str)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))


def auto_convert(output):
    if isinstance(output, torch.Tensor):
        return None, None, None, None, None, output
    return output

def train(train_loader, nets, optimizer, criterions, epoch, writer):
    global niter

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.train()
    if args.kd_mode in ['vid', 'ofd']:
        for i in range(1,4):
            criterionKD[i].train()

    end = time.time()
    pbar = tqdm(train_loader)
    for i, (img, target) in enumerate(pbar, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.kd_mode in ['sobolev', 'lwm']:
            img.requires_grad = True

        stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = auto_convert(snet(img))
        stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = auto_convert(tnet(img))

        cls_loss = criterionCls(out_s, target)
        if args.kd_mode in ['logits', 'st']:
            kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        else:
            raise Exception('Invalid kd mode...')
        loss = cls_loss + kd_loss
        # print(cls_loss.item(), kd_loss.item(), loss.item())

        writer.add_scalar('loss', loss, global_step=niter)
        pbar.set_postfix({'loss': loss, 'batch_id': niter})
        niter += 1

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                       'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                       cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
            logging.info(log_str)

def tspipe_loss(model_out: torch.Tensor, ema_model_out: torch.Tensor, label: torch.Tensor, tspipe_args: argparse.Namespace, args: argparse.Namespace, epoch: int):

    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    else:
        assert False
    
    criterionCls = torch.nn.CrossEntropyLoss()
    
    cls_loss = criterionCls(model_out, label)
    kd_loss = criterionKD(model_out, ema_model_out.detach()) * args.lambda_kd

    return cls_loss + kd_loss

niter = 0
def train_tspipe(tspipe_trainer:TSPipe, train_loader, nets, optimizer, criterions, epoch, writer):
    global niter

    pbar = tqdm(train_loader)
    for i, (img, target) in enumerate(pbar, start=1):
        loss = tspipe_trainer.feed(img, img, target)

        if loss is None:
            continue

        writer.add_scalar('loss', loss, global_step=niter)
        pbar.set_postfix({'loss': loss, 'batch_id': niter})

        niter += 1


def test(test_loader, nets, criterions, epoch):
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.eval()
    if args.kd_mode in ['vid', 'ofd']:
        for i in range(1,4):
            criterionKD[i].eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.kd_mode in ['sobolev', 'lwm']:
            img.requires_grad = True
            stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
            stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)
        else:
            with torch.no_grad():
                stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = (None, None, None, None, None, snet(img))
                stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = (None, None, None, None, None, tnet(img))

        cls_loss = criterionCls(out_s, target)
        if args.kd_mode in ['logits', 'st']:
            kd_loss  = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        elif args.kd_mode in ['fitnet', 'nst']:
            kd_loss = criterionKD(rb3_s[1], rb3_t[1].detach()) * args.lambda_kd
        elif args.kd_mode in ['at', 'sp']:
            kd_loss = (criterionKD(rb1_s[1], rb1_t[1].detach()) +
                       criterionKD(rb2_s[1], rb2_t[1].detach()) +
                       criterionKD(rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['pkt', 'rkd', 'cc']:
            kd_loss = criterionKD(feat_s, feat_t.detach()) * args.lambda_kd
        elif args.kd_mode in ['fsp']:
            kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
                       criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
                       criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['ab']:
            kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
                       criterionKD(rb2_s[0], rb2_t[0].detach()) +
                       criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['sobolev']:
            kd_loss = criterionKD(out_s, out_t, img, target) * args.lambda_kd
        elif args.kd_mode in ['lwm']:
            kd_loss = criterionKD(out_s, rb2_s[1], out_t, rb2_t[1], target) * args.lambda_kd
        elif args.kd_mode in ['irg']:
            kd_loss = criterionKD([rb2_s[1], rb3_s[1], feat_s, out_s],
                                  [rb2_t[1].detach(),
                                   rb3_t[1].detach(),
                                   feat_t.detach(), 
                                   out_t.detach()]) * args.lambda_kd
        elif args.kd_mode in ['vid', 'afd']:
            kd_loss = (criterionKD[1](rb1_s[1], rb1_t[1].detach()) +
                       criterionKD[2](rb2_s[1], rb2_t[1].detach()) +
                       criterionKD[3](rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['ofd']:
            kd_loss = (criterionKD[1](rb1_s[0], rb1_t[0].detach()) +
                       criterionKD[2](rb2_s[0], rb2_t[0].detach()) +
                       criterionKD[3](rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
        else:
            raise Exception('Invalid kd mode...')

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
    logging.info('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
    scale   = 0.1
    lr_list = [args.lr*scale] * 30
    lr_list += [args.lr*scale*scale] * 10
    lr_list += [args.lr*scale*scale*scale] * 10

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
    scale   = 0.1
    lr_list =  [args.lr] * 100
    lr_list += [args.lr*scale] * 50
    lr_list += [args.lr*scale*scale] * 50

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
