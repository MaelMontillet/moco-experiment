#!/usr/bin/env python

import os
import torch
import argparse
import math
import shutil
import time
from functools import partial

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.loader
import moco.optimizer
from moco import stain_augmentation
import vits
from torchvision.transforms import v2
import webdataset as wds
import io
import glob
from PIL import Image
import functools
import pandas


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo Pathology specific data augmentation Pre-Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=48, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')


# add with the fork
parser.add_argument('--tar-dir', help="tar of the dataset, it will be mounted recursively")
parser.add_argument('--output', help="path to the directory where the checkpoint will be stored", default="./")
parser.add_argument('--cache', help="Cache directory for WebDataset")
parser.add_argument('--num-tiles', help="Cache directory for WebDataset", type=int, default=1000000)
parser.add_argument('--version', help="Version of the stain augmentation: free or realistic.", default="free")


def png_decoder(sample):
    # find the first key ending with ".png"
    for k, v in sample.items():
        if k.endswith(".png"):
            # decode to PIL
            sample["png"] = Image.open(io.BytesIO(v)).convert("RGB")
            break
    return sample

def is_png(sample):
    # keep only samples with at least one key ending in ".png"
    return any(k.endswith(".png") for k in sample.keys())

def make_dataloader(args, buffer_size=10000):
    to_tensor =  v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    tar_files = glob.glob(os.path.join(args.tar_dir, "*.tar"))
    #Debugging setup:
    files = []
    table = pandas.read_csv("/p/scratch/mfmpm/mael/data/slide_table_tcga_brca.csv")
    ref = [file.split(".")[0] for file in table["FILENAME"]]
    for file in tar_files:
        if os.path.basename(file).split(".")[0] in ref:
            files.append(file)
    print0(f"Number of slides: {len(files)}")
    tar_files = files

    train_dataset = wds.WebDataset(
        tar_files, 
        resampled=True,
        shardshuffle=False,
        cache_dir=args.cache,
        nodesplitter=wds.split_by_node
    )
    train_dataset = (
        train_dataset.shuffle(buffer_size)
        .select(is_png)      # skip JSON or other files
        .map(png_decoder)    # decode and add 'png' key
        .to_tuple("png")     # now safe to extract
        .map_tuple(to_tensor)
    )

    # For IterableDataset objects, the batching needs to happen in the dataset.
    train_dataset = train_dataset.batched(args.batch_size)
    trainloader = wds.WebLoader(
        train_dataset, batch_size=None, 
        pin_memory=True, 
        num_workers=args.workers//get_world_size() - 1, 
        persistent_workers=True
        )

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(buffer_size).batched(args.batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    trainloader = trainloader.with_epoch(args.num_tiles // (args.batch_size * get_world_size()))

    return trainloader

@functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0

@functools.lru_cache(maxsize=None)
def get_local_rank():
    """Return the local rank of this process."""
    return int(os.getenv('LOCAL_RANK'))

@functools.lru_cache(maxsize=None)
def get_world_size():
    """Return the world size."""
    return torch.distributed.get_world_size()


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)


def main():
    args = parser.parse_args()

    torch.distributed.init_process_group(backend='cpu:gloo,cuda:nccl')
    # Get and set device.
    if not torch.cuda.is_available():
        raise ValueError("Cuda anavailable.")
    local_rank = get_local_rank()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)

    os.environ['TORCH_KERNEL_CACHE_PATH'] = '/tmp/torch_kernel_cache'
    os.makedirs('/tmp/torch_kernel_cache', exist_ok=True)

    if args.version != "free" and args.version != "realistic" and args.version != "none":
        raise ValueError()                    
        
    # create model
    print0("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256 
    
    # apply SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / get_world_size())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.amp.GradScaler("cuda")
    summary_writer = SummaryWriter() if is_root_process() else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print0("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print0("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        v2.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        v2.RandomApply([
            v2.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        v2.RandomHorizontalFlip(),
        normalize
    ]

    augmentation2 = [
        v2.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        v2.RandomApply([
            v2.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        v2.RandomApply([moco.loader.Solarize()], p=0.2),
        v2.RandomHorizontalFlip(),
        normalize
    ]

    if args.version == "free":
        stain_augmentor = stain_augmentation.all_free_version()
    elif args.version == "realistic":
        stain_augmentor = stain_augmentation.realistic_version()
    else:
        stain_augmentor = lambda x : x

    my_transform = moco.loader.CustomTransform(v2.Compose(augmentation1), 
                                               v2.Compose(augmentation2), 
                                               stain_augmentor)
    
    print0("Creating Dataset / DataLoader")
    train_loader = make_dataloader(args)
    print0("Starting training")


    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, transform=my_transform)

        if is_root_process(): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=os.path.join(args.output, 'checkpoint_%04d.pth.tar' % epoch))
    
    if is_root_process():
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, transform):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        args.num_tiles // (args.batch_size * get_world_size()),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = args.num_tiles // args.batch_size
    moco_m = args.moco_m
    for i, batch in enumerate(train_loader):
        batch = batch[0]
        batch = batch.cuda(get_local_rank(), non_blocking=True)
        images = transform(batch)

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        # compute output
        with torch.amp.autocast("cuda", enabled=True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if is_root_process():
            summary_writer.add_scalar(f"loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


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

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print0('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
