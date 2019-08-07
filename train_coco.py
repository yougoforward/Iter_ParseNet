import argparse
import os
import random
import sys
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

# from dataset.data_coco import DatasetGenerator
from dataset.data_submit import TrainGenerator
# from network.coconet import get_model
from network.gamanet_fea import get_model
from progress.bar import Bar
# from utils.loss import SegmentationMultiLoss
from utils.lovasz_loss import TrainValLovaszLoss
from utils.metric import *
from utils.parallel import DataParallelModel, DataParallelCriterion


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument('--method', type=str, default='coco_finetune')
    # Datasets
    # parser.add_argument('--root', default='/home/ubuntu/Data/VOC2012/', type=str)
    # parser.add_argument('--lst', default='./dataset/COCO/coco.txt', type=str)
    parser.add_argument('--root', default='/home/ubuntu/Data/LIP/', type=str)
    parser.add_argument('--lst', default='./dataset/LIP/train_val.txt', type=str)
    parser.add_argument('--crop-size', type=int, default=513)
    # parser.add_argument('--num-classes', type=int, default=21)
    parser.add_argument('--num-classes', type=int, default=20)
    parser.add_argument('--hbody-cls', type=int, default=3)
    parser.add_argument('--fbody-cls', type=int, default=2)
    # Optimization options
    parser.add_argument('--epochs', default=151, type=int)
    parser.add_argument('--batch-size', default=40, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--ignore-label', type=int, default=255)
    # Checkpoints
    parser.add_argument('--restore-from', default='./checkpoints/exp_coco/backbone_coco_final.pth', type=str)
    parser.add_argument('--snapshot_dir', type=str, default='./checkpoints/exp_coco/')
    parser.add_argument('--log-dir', type=str, default='./runs/coco/')
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--save-num', type=int, default=4)
    # Misc
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method='poly'):
    if method == 'poly':
        current_step = epoch * iters_per_epoch + i_iter
        max_step = args.epochs * iters_per_epoch
        lr = args.learning_rate * ((1 - current_step / max_step) ** 0.9)
    else:
        lr = args.learning_rate
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main(args):
    # initialization
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.method))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # conduct seg network
    seg_model = get_model(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    new_params = seg_model.state_dict().copy()

    if args.init:
        for i in saved_state_dict:
            i_parts = i.split('.')
            # if not i_parts[0] == 'fc':
            #     new_params['encoder.' + '.'.join(i_parts[:])] = saved_state_dict[i]
            if not i_parts[2] == 'conv4':
                if not i_parts[1] == 'layer_dsn':
                    new_params['.'.join(i_parts[:])] = saved_state_dict[i]
        seg_model.load_state_dict(new_params)
        print('loading params w/o fc')
    else:
        seg_model.load_state_dict(saved_state_dict)
        print('loading params all')

    model = DataParallelModel(seg_model)
    model.float()
    model.cuda()

    # define dataloader
    train_loader = data.DataLoader(TrainGenerator(root=args.root, list_path=args.lst, crop_size=args.crop_size),
                                   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # define criterion & optimizer
    criterion = TrainValLovaszLoss(ignore_index=args.ignore_label, only_present=True)
    criterion = DataParallelCriterion(criterion).cuda()

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    start = time.time()

    for epoch in range(0, args.epochs):
        print('\n{} | {}'.format(epoch, args.epochs - 1))
        # training
        _ = train(model, train_loader, epoch, criterion, optimizer, writer)

        if epoch == args.epochs - 1:
            model_dir = os.path.join(args.snapshot_dir, args.method + '_final.pth')
            torch.save(seg_model.state_dict(), model_dir)
            print('Model saved to %s' % model_dir)

    print('Complete using', time.time() - start, 'seconds')


def train(model, train_loader, epoch, criterion, optimizer, writer):
    # set training mode
    model.train()
    train_loss = 0.0
    iter_num = 0

    # Iterate over data.
    bar = Bar('Processing | {}'.format('train'), max=len(train_loader))
    bar.check_tty = False
    for i_iter, batch in enumerate(train_loader):
        sys.stdout.flush()
        start_time = time.time()
        iter_num += 1
        # adjust learning rate
        iters_per_epoch = len(train_loader)
        lr = adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method=args.lr_mode)

        # image, label, _ = batch
        # images, labels = image.cuda(), label.long().cuda()
        image, label, hlabel, flabel, _ = batch
        images, labels, hlabel, flabel = image.cuda(), label.long().cuda(), hlabel.cuda(), flabel.cuda()
        torch.set_grad_enabled(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output loss
        preds = model(images)
        # loss = criterion(preds, labels)  # batch mean
        loss = criterion(preds, [labels, hlabel, flabel])
        train_loss += loss.item()

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i_iter % 10 == 0:
            writer.add_scalar('learning_rate', lr, iter_num + epoch * len(train_loader))
            writer.add_scalar('train_loss', train_loss / iter_num, iter_num + epoch * len(train_loader))

        batch_time = time.time() - start_time
        # plot progress
        bar.suffix = '{} / {} | Time: {batch_time:.4f} | Loss: {loss:.4f}'.format(iter_num, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  loss=train_loss / iter_num)
        bar.next()

    epoch_loss = train_loss / iter_num
    writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
    bar.finish()

    return epoch_loss


if __name__ == '__main__':
    args = parse_args()
    main(args)
