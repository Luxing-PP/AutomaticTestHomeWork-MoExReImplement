import argparse
import time
import ssl

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pyramidnet_moex as PYRM_MOEX
import numpy as np

import warnings

warnings.filterwarnings("ignore")

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10, CIFAR-100 Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=200, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=240, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')

# todo check
parser.add_argument('--moex_prob', default=0.7, type=float,
                    help='moex_probability')
parser.add_argument('--lam', default=0.9, type=float,
                    help='moex probability')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

'''
Dataset: CiFar100
model:pyramidnet
'''


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    train_loader, val_loader, numOfClass = loadData()

    model = PYRM_MOEX.PyramidNet(args.dataset,
                                 200,
                                 240,
                                 numOfClass,
                                 args.bottleneck)

    model = torch.nn.DataParallel(model).cuda()

    # print(model)
    print('lam = ' + str(args.lam) + 'p =' + str(args.moex_prob))
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # 动量优化
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # CUDA 自优化
    cudnn.benchmark = True

    for epoch in range(0, args.epochs):
        # lr修正
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prediction
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        f = open('MoExRes.txt', 'a+')
        f.write('lam = ' + str(args.lam) +
                'epoch = ' + str(epoch) +
                'Current best accuracy (top-1 and 5 error):' + str(best_err1) + ', ' + str(best_err5) + "\n")
        f.close()

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)

    # Save Result
    f = open('MoExRes.txt', 'a+')
    f.write('lam = ' + str(args.lam) + ': Best accuracy (top-1 and 5 error):' + str(best_err1) + ', ' + str(
        best_err5) + "\n")
    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    f.close()


# 按照parser参数选择数据库读取
def loadData():
    # 预计算的均值与方差
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 图像增扩， 随机切割后强制缩放到32像素
        transforms.RandomHorizontalFlip(),  # 图像增扩， 一半概率的水平翻转
        transforms.ToTensor(),
        normalize,  # 正则化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 100
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        numberofclass = 10
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    return train_loader, val_loader, numberofclass


# TSN代码风格
def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    current_LR = get_learning_rate(optimizer)[0]
    lam = args.lam
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)  # 以moex_prob的概率进行混合
        if r < args.moex_prob:
            # generate mixed sample
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            # compute output
            input_a_var = torch.autograd.Variable(input, requires_grad=True)
            input_b_var = torch.autograd.Variable(input[rand_index], requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)
            output = model(input_a_var, input_b_var)
            loss = criterion(output, target_a_var) * lam + criterion(output, target_b_var) * (1. - lam)
        else:
            # compute output
            input_var = torch.autograd.Variable(input, requires_grad=True)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


# 用于存储均值的对象
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    """
    output.data , target ,topk=1,5
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置
    toCmp = target.view(1, -1).expand_as(pred)  # 骚啊0 0 把一个target弄成了K个
    correct = pred.eq(toCmp)

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


# 程序入口
if __name__ == '__main__':
    main()
