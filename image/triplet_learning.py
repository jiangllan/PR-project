import os
import time
import random
import warnings

warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from utils import *
from model import TripletShopeeImageEmbeddingNet
from dataset import TripletShopeeImageDataset

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

best_loss = 10000
train_loss = []
valid_loss = []
lr_log = []
f1_log = []
mAP_log = []
mrr_log = []


def cosine_similarity(x, y):
    return 1 - F.cosine_similarity(x, y)


def main(args):
    args.multiprocessing_distributed = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    global train_loss
    global valid_loss
    global lr_log
    global f1_log
    global mAP_log
    global mrr_log
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = TripletShopeeImageEmbeddingNet(model=args.model_name)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    train_csv = pd.read_csv(os.path.join(args.data_dir, 'split_data', 'train.csv'))
    train_csv['image'] = args.data_dir + 'train_images/' + train_csv['image']
    tmp = train_csv.groupby('label_group').posting_id.agg('unique').to_dict()
    train_csv['target'] = train_csv.label_group.map(tmp)

    val_csv = pd.read_csv(os.path.join(args.data_dir, 'split_data', 'val.csv'))
    val_csv['image'] = args.data_dir + 'train_images/' + val_csv['image']
    tmp = val_csv.groupby('label_group').posting_id.agg('unique').to_dict()
    val_csv['target'] = val_csv.label_group.map(tmp)

    test_csv = pd.read_csv(os.path.join(args.data_dir, 'split_data', 'test.csv'))
    test_csv['image'] = args.data_dir + 'train_images/' + test_csv['image']
    tmp = test_csv.groupby('label_group').posting_id.agg('unique').to_dict()
    test_csv['target'] = test_csv.label_group.map(tmp)

    train_dataset = TripletShopeeImageDataset(
        train_csv,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        train=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = TripletShopeeImageDataset(
        val_csv,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        train=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    test_dataset = TripletShopeeImageDataset(
        test_csv,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        train=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # state = {'model': args.model_name,
    #           'state_dict': model.state_dict()}
    # torch.save(state, '/home/jhj/PR-project/shopee/data/best_triplet_d201.pth')
    #
    # assert 1==0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    if args.test:
        topns = [2]
        for topn in topns:
            args.topn = topn
            f1, mAP, mrr = test(test_loader, test_csv, model, args)

        return

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_similarity,
                                                 margin=args.tripletloss_margin).cuda(args.gpu)

    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    epoch_time = AverageMeter('Epoch Tiem', ':6.3f')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        ### Train for one epoch
        tr_loss, lr = train(train_loader, model, criterion, optimizer, epoch, args)

        ### Evaluate on validation set
        val_loss = validate(val_loader, model, criterion, args)

        ### Remember best Acc@1 and save checkpoint
        is_best = val_loss > best_loss
        best_loss = min(val_loss, best_loss)

        f1, mAP, mrr = test(test_loader, test_csv, model, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            train_loss.append(tr_loss)
            valid_loss.append(val_loss)
            lr_log.append(lr)
            f1_log.append(f1)
            mAP_log.append(mAP)
            mrr_log.append(mrr)

            df = pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss, 'lr_log': lr_log})
            log_file = os.path.join(args.log_dir, args.log_name)

            with open(log_file, "w") as f:
                df.to_csv(f)

            save_checkpoint({
                'epoch': epoch,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'lr_log': lr_log,
            }, args, is_best, filename='checkpoint_epoch{}.pth.tar'.format(epoch))

            epoch_time.update(time.time() - start_time, 1)
            print('Duration: %4f H, Left Time: %4f H' % (
                epoch_time.sum / 3600, epoch_time.avg * (args.epochs - epoch - 1) / 3600))
            start_time = time.time()

            df = pd.DataFrame(
                {'f1_log': f1_log, 'mAP': mAP_log, 'mrr': mrr_log})

            log_file = os.path.join(args.log_dir, 'test_result.txt')
            with open(log_file, "w") as f:
                df.to_csv(f)

    return


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    ### Switch to train mode
    model.train()
    model.module.base_model.eval()
    end = time.time()

    for i, (anchor, positive, negative) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        ### Measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            anchor = anchor.cuda(args.gpu, non_blocking=True)
            positive = positive.cuda(args.gpu, non_blocking=True)
            negative = negative.cuda(args.gpu, non_blocking=True)

        ### Compute output
        anchor_feature, positive_feature, negative_feature = model(anchor, positive, negative)
        loss = criterion(anchor_feature, positive_feature, negative_feature)

        ### Measure accuracy and record loss
        losses.update(loss.item(), anchor.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            print('LR: %6.4f' % (lr))

    return losses.avg, lr


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Val: ')

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(val_loader):
            if args.gpu is not None:
                anchor = anchor.cuda(args.gpu, non_blocking=True)
                positive = positive.cuda(args.gpu, non_blocking=True)
                negative = negative.cuda(args.gpu, non_blocking=True)

            ### Compute output
            anchor_feature, positive_feature, negative_feature = model(anchor, positive, negative)
            loss = criterion(anchor_feature, positive_feature, negative_feature)

            ### Measure accuracy and record loss
            losses.update(loss.item(), anchor.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg


def test(test_loader, test_csv, model, args):
    ### Switch to evaluate mode
    model.eval()

    cnn_feature = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            ### Compute output
            input_feature = model(input)
            input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1])
            input_feature = input_feature.data.cpu().numpy()
            cnn_feature.append(input_feature)

        cnn_feature = np.vstack(cnn_feature)
        cnn_feature = normalize(cnn_feature)
        cnn_feature = torch.from_numpy(cnn_feature)
        cnn_feature = cnn_feature.cuda()
        f1, mAP, mrr = cos_similarity(cnn_feature, test_csv, drop_itself=args.drop_itself, f1_threshold=args.f1_threshold,
                                      mAP_threshold=args.mAP_threshold, topn=args.topn)
        print('{} topn/drop_itself = {}/{}, f1 = {:.4f}, mAP = {:.4f}, mrr = {:.4f}'.format(
            args.model_name, args.topn, args.drop_itself, f1, mAP, mrr))

    return f1, mAP, mrr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='DIR', default='../data/', help='path to dataset')
    parser.add_argument("--log_dir", type=str, default="../log/image-only/triplet/")
    parser.add_argument("--log_name", type=str, default="train_log.txt")
    parser.add_argument("--tripletloss_margin", type=float, default=1.0)
    parser.add_argument("--drop_itself", action="store_true", default=False)
    parser.add_argument("--f1_threshold", type=float, default=0.)
    parser.add_argument("--mAP_threshold", type=float, default=0.)
    parser.add_argument("--raw_image", action="store_true")
    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument('--model_name', default='densenet201', type=str, metavar='M',
                        help='model to train the dataset')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)')
    parser.add_argument('--lr_type', default='cosine', type=str, metavar='T',
                        help='learning rate strategy (default: cosine)',
                        choices=['cosine', 'multistep'])
    parser.add_argument('--warmup_epoch', default=None, type=int, metavar='N',
                        help='number of epochs to warm up')
    parser.add_argument('--warmup_lr', default=0.1, type=float,
                        metavar='LR', help='initial warm up learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--resume', type=str, help='test the model')

    # multiprocess
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:29501', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--visible_gpus', type=str, default='0',
                        help='visible gpus')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir, 'new_data_split_{}_with_embed_layer_epoch{}_lr{}_{}_bs{}_margin{}'.format(
        args.model_name, args.epochs, args.lr, args.lr_type, args.batch_size, args.tripletloss_margin
    ))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    main(args)
