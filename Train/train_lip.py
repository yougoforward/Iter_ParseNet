import argparse
import os
import random
import sys
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data

from dataset.datasets import DatasetGenerator
from network.ocnet import get_model
from progress.bar import Bar
from utils.loss import SegmentationMultiLoss
from utils.metric import *
from utils.parallel import DataParallelModel, DataParallelCriterion
from utils.visualize import inv_preprocess, decode_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument('--method', type=str, default='se_ocnet_stem_dsn_fres2_weight')
    # Datasets
    parser.add_argument('--root', default='/home/ubuntu/Data/LIP/train_set/', type=str)
    parser.add_argument('--val-root', default='/home/ubuntu/Data/LIP/val_set/', type=str)
    parser.add_argument('--lst', default='./dataset/LIP/train_id.txt', type=str)
    parser.add_argument('--val-lst', default='./dataset/LIP/val_id.txt', type=str)
    parser.add_argument('--crop-size', type=int, default=473)
    parser.add_argument('--num-classes', type=int, default=20)
    # Optimization options
    parser.add_argument('--epochs', default=151, type=int)
    parser.add_argument('--batch-size', default=40, type=int)
    parser.add_argument('--learning-rate', default=7e-3, type=float)
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--ignore-label', type=int, default=255)
    # Checkpoints
    parser.add_argument('--restore-from', default='./checkpoints/init/resnet101_stem.pth', type=str)
    parser.add_argument('--snapshot_dir', type=str, default='./checkpoints/exp/')
    parser.add_argument('--log-dir', type=str, default='./runs/')
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
            if not i_parts[0] == 'fc':
                new_params['encoder.' + '.'.join(i_parts[:])] = saved_state_dict[i]
        seg_model.load_state_dict(new_params)
        print('loading params w/o fc')
    else:
        seg_model.load_state_dict(saved_state_dict)
        print('loading params all')

    model = DataParallelModel(seg_model)
    model.float()
    model.cuda()

    # define dataloader
    train_loader = data.DataLoader(DatasetGenerator(root=args.root, list_path=args.lst,
                                                    crop_size=args.crop_size, training=True),
                                   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(DatasetGenerator(root=args.val_root, list_path=args.val_lst,
                                                  crop_size=args.crop_size, training=False),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # define criterion & optimizer
    criterion = SegmentationMultiLoss(ignore_index=args.ignore_label)

    criterion = DataParallelCriterion(criterion).cuda()

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # key points
    best_val_mIoU = 0
    best_val_pixAcc = 0
    start = time.time()

    for epoch in range(0, args.epochs):
        print('\n{} | {}'.format(epoch, args.epochs - 1))
        # training
        _ = train(model, train_loader, epoch, criterion, optimizer, writer)

        # validation
        if epoch > 99:
            val_pixacc, val_miou = validation(model, val_loader, epoch, writer)
            # save model
            if val_pixacc > best_val_pixAcc:
                best_val_pixAcc = val_pixacc
            if val_miou > best_val_mIoU:
                best_val_mIoU = val_miou
                model_dir = os.path.join(args.snapshot_dir, args.method + '_miou.pth')
                torch.save(seg_model.state_dict(), model_dir)
                print('Model saved to %s' % model_dir)

    print('Complete using', time.time() - start, 'seconds')
    print('Best pixAcc: {} | Best mIoU: {}'.format(best_val_pixAcc, best_val_mIoU))


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

        image, label, _ = batch
        images, labels = image.cuda(), label.long().cuda()
        torch.set_grad_enabled(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output loss
        preds = model(images)
        loss = criterion(preds, labels)  # batch mean
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


def validation(model, val_loader, epoch, writer):
    # set evaluate mode
    model.eval()

    total_correct, total_label = 0, 0
    hist = np.zeros((args.num_classes, args.num_classes))

    # Iterate over data.
    bar = Bar('Processing {}'.format('val'), max=len(val_loader))
    bar.check_tty = False
    for idx, batch in enumerate(val_loader):
        image, target, _ = batch
        image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            h, w = target.size(1), target.size(2)
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            preds = F.interpolate(input=outputs[0], size=(h, w), mode='bilinear', align_corners=True)
            if idx % 50 == 0:
                img_vis = inv_preprocess(image, num_images=args.save_num)
                label_vis = decode_predictions(target.int(), num_images=args.save_num, num_classes=args.num_classes)
                pred_vis = decode_predictions(torch.argmax(preds, dim=1), num_images=args.save_num,
                                              num_classes=args.num_classes)

                # visual grids
                img_grid = torchvision.utils.make_grid(torch.from_numpy(img_vis.transpose(0, 3, 1, 2)))
                label_grid = torchvision.utils.make_grid(torch.from_numpy(label_vis.transpose(0, 3, 1, 2)))
                pred_grid = torchvision.utils.make_grid(torch.from_numpy(pred_vis.transpose(0, 3, 1, 2)))
                writer.add_image('val_images', img_grid, epoch * len(val_loader) + idx + 1)
                writer.add_image('val_labels', label_grid, epoch * len(val_loader) + idx + 1)
                writer.add_image('val_preds', pred_grid, epoch * len(val_loader) + idx + 1)

            # pixelAcc
            correct, labeled = batch_pix_accuracy(preds.data, target)
            # mIoU
            hist += fast_hist(preds, target, args.num_classes)

            total_correct += correct
            total_label += labeled
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = round(np.nanmean(per_class_iu(hist)) * 100, 2)
            # plot progress
            bar.suffix = '{} / {} | pixAcc: {pixAcc:.4f}, mIoU: {IoU:.4f}'.format(idx + 1, len(val_loader),
                                                                                  pixAcc=pixAcc, IoU=IoU)
            bar.next()

    mIoU = round(np.nanmean(per_class_iu(hist)) * 100, 2)

    writer.add_scalar('val_pixAcc', pixAcc, epoch)
    writer.add_scalar('val_mIoU', mIoU, epoch)
    bar.finish()

    return pixAcc, mIoU


if __name__ == '__main__':
    args = parse_args()
    main(args)
