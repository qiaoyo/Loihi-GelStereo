import numpy as np
import torch
import torch.nn as nn
import os
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


if __name__ == '__main__':
    ## Options -------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', default=32, type=int, metavar='N',
                        help='input batch size')
    # GPU
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help='use cuda or not')

    parser.add_argument('--manualSeed', default=3963, type=int, help='manual seed')  # , default=4216

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Dataset
    parser.add_argument('--dataset', default='testset', type=str, help='which dataset to test, testset or trainset')

    # Test model
    parser.add_argument('--checkpoint', default='results/checkpoint_1', type=str, metavar= \
        'PATH', help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--test_model', default='model_best.pth.tar', type=str, help='***.pth.tar')
    parser.add_argument('--model_arch', type=str, default='ConvLSTM', help='The model arch you selected')

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

    # if not os.path.exists(opt.checkpoint):
    #     os.makedirs(opt.checkpoint)
    # opt_name = os.path.join(opt.checkpoint, 'opt.txt')
    # with open(opt_name, 'at') as opt_file:  # wt 以文本写入 覆盖原文件  at
    #     opt_file.write('------------ Options -------------\n')
    #     for k, v in sorted(args.items()):
    #         opt_file.write('%s: %s\n' % (str(k), str(v)))
    #     opt_file.write('-------------- End ----------------\n')


    ## Dataset ---------------
    if opt.dataset == 'trainset':
        trainset = SlipDataset(seq_length=3, phase='train')
        data_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.workers))
    elif opt.dataset == 'testset':
        testset = SlipDataset(seq_length=3, phase='test')
        data_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.workers))
    else:
        print('dataset selection error')
        exit()

    ###---------set model arch and load par-----------------------
    # # Model
    if opt.model_arch == 'ConvLSTM':
        model = ConvLSTM(conv_1=16, conv_2=32, fc_1=64, lstm_layers=1, lstm_hidden_size=64, cls_num=2)
    elif opt.model_arch == 'Conv':
        model = Conv(in_channel=6, conv_1=16, conv_2=32, cls_num=2)
    else:
        print('model selection error')
        exit()
    #
    # cuda
    if opt.use_cuda:
        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(torch.device('cpu'))

    # load model
    model_path = os.path.join(opt.checkpoint, opt.test_model)
    assert os.path.isfile(model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    ###----------------------------------------------------------

    # ###-------------------load model----- arch+params---------
    # # load model
    # model_path = os.path.join(opt.checkpoint, opt.test_model)
    # assert os.path.isfile(model_path), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load(model_path)
    # model = checkpoint['model']

    # ## Use cuda model ---------------
    # if opt.use_cuda:
    #     # model = torch.nn.DataParallel(model).cuda()
    #     model = model.cuda()
    # else:
    #     model.to(torch.device('cpu'))
    ###------------------------------------------------------------

    print('Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.eval()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    infer_time = AverageMeter()
    end = time.time()

    pred_cls = []
    gt_cls = []

    for batch_idx, (motion, gt) in enumerate(data_loader):
        # switch to train mode
        model.eval()

        data_time.update(time.time() - end)

        # input and ground truth Variable and use cuda
        t1 = time.time()
        motion = torch.autograd.Variable(motion)
        gt = torch.autograd.Variable(gt)
        if opt.use_cuda:
            motion = motion.cuda()
            gt = gt.cuda()

        # model inference
        outputs = model(motion)

        infer_time.update(time.time()-t1, gt.size(0))

        # compute accuracy, precision and recall
        pred_index = torch.max(outputs, dim=1)[1]  # max index
        gt_index = torch.max(gt, dim=1)[1]  # max index

        pred_cls += pred_index.cpu().data.numpy().tolist()
        gt_cls += gt_index.cpu().data.numpy().tolist()

        # time
        batch_time.update(time.time() - end)
        end = time.time()


    acc = accuracy_score(y_pred=pred_cls, y_true=gt_cls)
    precision = precision_score(y_pred=pred_cls, y_true=gt_cls)
    recall = recall_score(y_pred=pred_cls, y_true=gt_cls)

    print('inference time', infer_time.avg)
    print('accuracy', acc)
    print('precision', precision)
    print('recall', recall)

