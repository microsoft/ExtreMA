#!/usr/bin/env python

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import json
import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import lib.builder
from lib.logger import setup_logger
from lib.augment import create_augmentation
from lib.misc import NativeScalerWithGradNormCount as NativeScaler
from lib.dataload_optim import PersistentDataLoader, SoftwarePipeline
import lib.misc as misc

from timm.optim import optim_factory, create_optimizer
import vits

model_names = ['vit_small', 'vit_base', 'vit_large'] 

parser = argparse.ArgumentParser(description='ExtreMA Arguments')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vit_base)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers per gpu (default: 6)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2048), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--weight-decay-end', default=None, type=float,
                    metavar='W', help='weight decay end (default: 1e-6)',
                    dest='weight_decay_end')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=5, type=int,
                    metavar='N', help='save frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--log_dir', default="tf_logs", type=str,
                    help='dir of logs')
parser.add_argument('--output_dir', default="results", type=str,
                    help='dir of checkpoints')

# siamese specific configs:
parser.add_argument('--proj-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--ema-momentum', default=0.996, type=float,
                    help='momentum of updating momentum encoder (default: 0.996)')
parser.add_argument('--contrast-temp', default=1.0, type=float,
                    help='contrastive softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--drop_path_rate', type=float, default=0.0, help="stochastic depth rate")
parser.add_argument('--attn_drop_rate', type=float, default=0.0, help="attention dropout rate")
parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                    help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
parser.add_argument('--class_attention_layers', default=2, type=int)

# other hyper-params
parser.add_argument('--opt', default='adamw', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: adamw)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                     help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--adjust-weight-decay', action='store_true',
                    help='cosine weight decay')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.2, type=float,
                    help='minimum scale for random cropping student (default: 0.2)')

# augmentation options
parser.add_argument('--aug-spatial', action='store_true',
                    help='use spatial data augmentation')
parser.add_argument('--aug-centercrop', action='store_true',
                    help='use centercrop data augmentation')
parser.add_argument('--aug-spatialconsistent-color', action='store_true',
                    help='use spatial consistent with colorjitter data augmentation')
parser.add_argument('--loss', default='byol', type=str,
                    choices=['infonce', 'byol'],
                    help='loss function to use')

# add mask options
parser.add_argument('--mask-ratio', default=0.8, type=float,
                    help='mask ratio for student augmentation')
parser.add_argument('--num-masks', default=1, type=int)
parser.add_argument('--disjoint', action='store_true',
                    help='use disjoint sampling of patches')
parser.set_defaults(disjoint=True)


def main_worker(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print("args.output_dir", args.output_dir)
    print("args.log_dir", args.log_dir)

    if global_rank == 0 and args.log_dir is not None:
        with open(args.log_dir + '/config.json', "w") as config_file:
            json.dump(vars(args), config_file)
        os.makedirs(args.log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        summary_writer = None
    logger = setup_logger(output=args.log_dir, distributed_rank=global_rank, name="byol")
    
    device = torch.device(args.device)

    if args.seed is not None:
        seed = args.seed + misc.get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    cudnn.benchmark = True
    
    local_batch_size = int(args.batch_size / misc.get_world_size())
    augmentation  = create_augmentation(args)
    logger.info(augmentation)

    # Data loading
    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transform=augmentation,
    )
        
    if True: #args.distributed:
        num_tasks = misc.get_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    train_loader = SoftwarePipeline(PersistentDataLoader(
        train_dataset, batch_size=local_batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True))

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    base_encoder = partial(vits.__dict__[args.arch], drop_path_rate=args.drop_path_rate, attn_drop_rate=args.attn_drop_rate, init_values=args.layer_scale_init_value, class_attention_layers=args.class_attention_layers)
    ema_encoder = partial(vits.__dict__[args.arch], init_values=args.layer_scale_init_value, class_attention_layers=args.class_attention_layers)
    model = lib.builder.ExtreMA(
        base_encoder, ema_encoder,
        args.proj_dim, args.mlp_dim, args.contrast_temp, args.mask_ratio, args.num_masks, args.disjoint)
    model.to(device)

    # infer learning rate before changing batch sizex
    args.lr = args.lr * args.batch_size / 256

    if True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    logger.info(model) 

    param_groups = optim_factory.add_weight_decay(model.module, args.weight_decay, model.module.no_weight_decay())
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.opt_betas)

    logger.info(optimizer)    
    scaler = NativeScaler()

    # auto resume from a checkpoint
    args.resume = os.path.join(args.output_dir, 'current.pth.tar')
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and misc.get_rank() == 0 and (epoch+1) % args.save_freq == 0): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.output_dir, epoch))
            shutil.copyfile('{}/checkpoint_{:04d}.pth.tar'.format(args.output_dir, epoch), '{}/current.pth.tar'.format(args.output_dir))

    if misc.get_rank() == 0:
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    loss_scales = AverageMeter('LossScale', ':.4e')
    weight_decays = AverageMeter('WeightDecay', ':.4e')
    grad_norms = AverageMeter('GradNorm', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, grad_norms, loss_scales, weight_decays, learning_rates],
        prefix="Epoch: [{}]".format(epoch))
    logger = logging.getLogger('byol')

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    ema_momentum = args.ema_momentum
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        ema_momentum = adjust_ema_momentum(epoch + i / iters_per_epoch, args)
        if args.adjust_weight_decay:
            wd = adjust_decay_rate(optimizer, epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            if isinstance(images, list):
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
                bsz = images[0].size(0)
            else:
                images = images.cuda(args.gpu, non_blocking=True)
                bsz = images.size(0)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images, ema_momentum, args.loss)
        losses.update(loss.item(), bsz)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        norm = scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            grad_norms.update(norm)
            loss_scales.update(scaler.state_dict()["scale"])
            learning_rates.update(optimizer.param_groups[1]['lr'])
            weight_decays.update(optimizer.param_groups[1]['weight_decay'])
            progress.display(logger,i)

    if misc.get_rank() == 0:
        summary_writer.add_scalar("losses", losses.avg, epoch )
        summary_writer.add_scalar("opt/grad_norm", grad_norms.avg, epoch )
        summary_writer.add_scalar("opt/loss_scale", loss_scales.avg, epoch )
        summary_writer.add_scalar("opt/lr", learning_rates.avg, epoch )
        summary_writer.add_scalar("opt/wd", weight_decays.avg, epoch )
       

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, logger, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def adjust_decay_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine"""
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd = args.weight_decay_end + 0.5 * (args.weight_decay - args.weight_decay_end) * (1. + math.cos(math.pi * (epoch / args.epochs)))
    for param_group in optimizer.param_groups:
        if param_group['weight_decay'] > 0:
            param_group['weight_decay'] = wd
    return wd

def adjust_ema_momentum(epoch, args):
    """Decays the momentum paramter with half-cycle cosine"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.ema_momentum)
    return m

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    args = parser.parse_args()
    main_worker(args)
