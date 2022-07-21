from torch.nn.modules.container import Sequential
import torchvision.models as models
import torch
from models.mlp_head import MLPHead


class ResNetReshape(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18()
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50()
        elif kwargs['name'] == 'resnet101':
            resnet = models.resnet101()
        elif kwargs['name'] == 'resnet152':
            resnet = models.resnet152()
        
        # print(resnet)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

        self.reshaper = ResNetReshape()
        
        # print(self.encoder)
        
        # print(self.projection)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

    def __iter__(self):
        for m in self.encoder:
            yield m
        yield self.reshaper
        for m in self.projection:
            yield m

    def __len__(self):
        return len(self.encoder) + 1 + len(self.projection)


    @staticmethod
    def remove_inplace_ops(module: torch.nn.Module) -> None:
        # inplace ops is not compatible with gpipe
        if hasattr(module, 'inplace') and module.inplace:
            # print("Removing inplace ops from ", module)
            module.inplace = False
        for child in module.children():
                ResNet18.remove_inplace_ops(child)        

    def to_sequential(self, without_projection: bool = False) -> Sequential:
        encoder_layers = []
        for child in self.encoder.children():
            if isinstance(child, Sequential):
                encoder_layers.extend(child.children())
            else:
                encoder_layers.append(child)

        projection = self.projection.to_sequential().children() if not without_projection else []

        p =  Sequential(
            *encoder_layers,
            self.reshaper,
            *projection,
        )
        ResNet18.remove_inplace_ops(p)

        return p

if __name__ == "__main__":
    resnet = ResNet18(name="resnet18", projection_head={'mlp_hidden_size':512, 'projection_size': 128})
    for m in resnet:
        print(m)