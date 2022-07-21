
from tspipe.model_base import SequentialableModel
from . import resnet
from . import vits
from . import efficient_net


def create_vit(network_name: str, num_classes: int, image_size: int = 224):
    return vits.__dict__[network_name](num_classes=num_classes, img_size=image_size)


def create_resnet(network_name: str, num_classes: int, image_size: int = 224):
    return resnet.ResNetBase(network_name, num_classes, image_size)


def create_efficientnet(network_name: str, num_classes: int, image_size: int = 224):
    return efficient_net.EfficientNet.from_name(network_name, num_classes=num_classes, image_size=image_size)


def create_model(name: str, num_class: int, image_size: int = 224) -> SequentialableModel:
    params = {'network_name': name, 'num_classes': num_class, 'image_size': image_size}

    if 'vit' in name:
        return create_vit(**params)
    if 'resnet' in name:
        return create_resnet(**params)
    if 'efficientnet' in name:
        return create_efficientnet(**params)

    raise Exception('model name does not exist.')
