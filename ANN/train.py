import numpy as np
import torch
import torch.nn as nn
import os
import shutil
import time
import torch.nn.parallel
from utils.logger import *
from utils.SlipDataset import SlipDataset
from utils.models import *
from utils.misc import AverageMeter, AveragePerEle
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score, precision_score, recall_score



def adjust_learning_rate(optimizer, epoch, opt):
    if epoch % opt.schedule ==0 and epoch !=0 :
        opt.lr *= opt.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def train(data_loader, model, optimizer):
    model.train()

    loss_aver = AverageMeter()
    acc_aver = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()


    for batch_idx, (motion, gt) in enumerate(data_loader):
        # switch to train mode
        model.train()

        data_time.update(time.time() - end)

        # input and ground truth Variable and use cuda
        motion = torch.autograd.Variable(motion)
        gt = torch.autograd.Variable(gt)
        if opt.use_cuda:
            motion = motion.cuda()
            gt = gt.cuda()

        # model forward
        outputs = model(motion)

        # compute loss and accuracy
        loss = F.cross_entropy(outputs, gt, reduction='mean')
        pred_index = torch.max(outputs, dim=1)[1]  # max index
        gt_index = torch.max(gt, dim=1)[1]  # max index

        acc = accuracy_score(y_pred=pred_index.cpu().data.numpy(), y_true=gt_index.cpu().data.numpy())

        # loss and  update
        loss_aver.update(loss.item(), gt.size(0))
        acc_aver.update(acc, gt.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time
        batch_time.update(time.time() - end)
        end = time.time()

    print('train: sum data time %f  sum batch time %f' % (data_time.sum, batch_time.sum))
    return loss_aver.avg, acc_aver.avg


def test(data_loader, model):
    model.eval()

    loss_aver = AverageMeter()
    acc_aver = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    for batch_idx, (motion, gt) in enumerate(data_loader):
        # switch to train mode
        model.eval()

        data_time.update(time.time() - end)

        # input and ground truth Variable and use cuda
        motion = torch.autograd.Variable(motion)
        gt = torch.autograd.Variable(gt)
        if opt.use_cuda:
            motion = motion.cuda()
            gt = gt.cuda()

        # model forward
        outputs = model(motion)

        # compute loss and accuracy
        loss = F.cross_entropy(outputs, gt, reduction='mean')
        pred_index = torch.max(outputs, dim=1)[1]  # max index
        gt_index = torch.max(gt, dim=1)[1]  # max index

        acc = accuracy_score(y_pred=pred_index.cpu().data.numpy(), y_true=gt_index.cpu().data.numpy())

        # loss and  update
        loss_aver.update(loss.item(), gt.size(0))
        acc_aver.update(acc, gt.size(0))

        # time
        batch_time.update(time.time() - end)
        end = time.time()

    print('test: sum data time %f  sum batch time %f' % (data_time.sum, batch_time.sum))
    return loss_aver.avg, acc_aver.avg



if __name__ == '__main__':
    ## Options -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                          help='manual epoch number (useful on restarts)')
    parser.add_argument('--batchSize', default=32, type=int, metavar='N',
                        help='input batch size')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--test', action='store_true', help='if test after each epoch')
    # Training weight decay
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--schedule', type=int, default=20,
                        help='Decrease learning rate at these epochs.')  # nargs='+',
    parser.add_argument('--gamma', type=float, default=0.5, help='LR is mult-\
                                         iplied by gamma on schedule.')
    # GPU
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help='use cuda or not')

    parser.add_argument('--manualSeed', default=3963, type=int, help='manual seed')  # , default=4216

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Checkpoints
    parser.add_argument('--checkpoint', default='results/checkpoint_1', type=str, metavar= \
        'PATH', help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='file name of the latest checkpoint (default: none), e.g. checkpoint.pth.tar')  # model_best.pth.tar
    parser.add_argument('--model_arch', type=str, default='CNN2FC1_MLP1_pos', help='The model arch you selected')

    # plot
    parser.add_argument('--plot', default='Train Loss/Valid Loss', type=str,
                        help='choose which figure to plot, e.g.')

    opt = parser.parse_args()

    # Random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)  #

    if opt.use_cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True
        cudnn.enabled = True

    args = vars(opt)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)
    opt_name = os.path.join(opt.checkpoint, 'opt.txt')
    with open(opt_name, 'at') as opt_file:  # wt 以文本写入 覆盖原文件  at
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    ## Dataset ---------------
    trainset = SlipDataset(seq_length=3, phase='train_0507')
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    if opt.test:
        testset = SlipDataset(seq_length=3, phase='val_0507')
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.workers))

    ## Model -----------------
    if opt.model_arch == 'ConvLSTM':
        model = ConvLSTM(conv_1=16, conv_2=32, fc_1=64, lstm_layers=1, lstm_hidden_size=64, cls_num=2)
    elif opt.model_arch == 'Conv':
        model = Conv(in_channel=6, conv_1=16, conv_2=32, cls_num=2)
    else:
        print('model selection error')
        exit()

    ## Use cuda model ---------------
    if opt.use_cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
    else:
        model.to(torch.device('cpu'))

    print('Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('Trainable params: %.4fM' % (
                sum(p.numel() for p in model.parameters() if p.requires_grad == True) / 1000000.0))

    ## Optimizer --------------
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    ## Resume and logger------------------
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(os.path.join(opt.checkpoint, opt.resume)), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(opt.checkpoint, opt.resume))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), resume=True)
        opt.lr = optimizer.param_groups[0]['lr']
        # print(opt.lr)
    else:
        start_epoch = 0
        best_acc = 0  # +∞
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'))
        if opt.test:
            logger.set_names(['Learning Rate', 'Train Loss', 'Train acc', 'Val Loss', 'Val acc'])
        else:
            logger.set_names(['Learning Rate', 'Train Loss', 'Train acc'])

    ## epochs
    for epoch in range(start_epoch, opt.epochs):

        train_loss, train_acc = train(train_loader, model, optimizer)

        if opt.test:
            test_loss, test_acc = test(test_loader, model)
            print(
                'Epoch: [%d | %d] LR: %f | train loss: %4f | train_acc: %4f || test loss: %4f | test acc: %4f\n' % (
                    epoch + 1, opt.epochs, opt.lr, train_loss, train_acc, test_loss, test_acc))
            logger.append([opt.lr, train_loss, train_acc, test_loss, test_acc])
        else:
            print('Epoch: [%d | %d] LR: %f | train loss: %4f | train_acc: %4f\n' % (
                epoch + 1, opt.epochs, opt.lr, train_loss, train_acc))
            logger.append([opt.lr, train_loss, train_acc])
        # print('\n')

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'model': model,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=opt.checkpoint)

    # logger.plot(names=['Train Loss', 'Test Loss'], figname='loss.png', ylabel='loss')
    # logger.plot(names=['Train acc', 'Test acc'], figname='acc.png', ylabel='accuracy')

    logger.plot(names=['Train Loss', 'Val Loss'], figname='loss.png', ylabel='loss')
    logger.plot(names=['Train acc', 'Val acc'], figname='acc.png', ylabel='accuracy')

    logger.close()


