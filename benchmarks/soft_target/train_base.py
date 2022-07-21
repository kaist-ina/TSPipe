from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dst
import torchvision.transforms as transforms

import dataset.datasets as small_datasets
from models.factory import create_model
from utils import (AverageMeter, accuracy, count_parameters_in_MB,
                   create_exp_dir, save_checkpoint)

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')  # cifar10/cifar100
parser.add_argument('--net_name', type=str, required=True, help='name of basenet')   # resnet20/resnet110


args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    image_size = 32 if 'cifar' in args.data_name else 224

    logging.info('----------- Network Initialization --------------')
    net = create_model(args.net_name, num_class=args.num_class, image_size=image_size)
    net = net.cuda()
    logging.info('%s', net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info('-----------------------------------------------')

    # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'initial_r{}.pth.tar'.format(args.net_name[6:]))
    torch.save({
        'epoch': 0,
        'net': net.state_dict(),
        'prec@1': 0.0,
        'prec@5': 0.0,
    }, save_path)

    # initialize optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

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
            batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset(root=args.img_root, transform=test_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    best_top1 = 0
    for epoch in range(1, args.epochs+1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        train(train_loader, net, optimizer, criterion, epoch)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top5 = test(test_loader, net, criterion)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # save model
        is_best = False
        if test_top1 > best_top1:
            best_top1 = test_top1
            is_best = True
        logging.info('Saving models......')
        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'prec@1': test_top1,
            'prec@5': test_top5,
        }, is_best, args.save_root)


def train(train_loader, net, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        out = net(img)
        if isinstance(out, tuple):
            _, _, _, _, _, out = out
        loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
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
                       'loss:{losses.val:.4f}({losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                            losses=losses, top1=top1, top5=top5))
            logging.info(log_str)


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()

    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            out = net(img)
            if isinstance(out, tuple):
                _, _, _, _, _, out = out
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr(optimizer, epoch):
    scale = 0.1
    lr_list = [args.lr] * 100
    lr_list += [args.lr*scale] * 50
    lr_list += [args.lr*scale*scale] * 50

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
