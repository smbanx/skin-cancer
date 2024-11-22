import argparse
import os
import shutil
import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader, test_loader
from timm.models import VisionTransformer, Bottleneck, ResNetV2, resnetv2
from timm.models.resnet import resnet50
from timm.utils import AverageMeter
from tqdm import tqdm

best_acc = 0
best_loss = float('inf')
best_epoch = -1

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1.0e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)


def main():
    global best_acc
    global best_loss
    global best_epoch

    # model = VisionTransformer(num_classes=4)
    model = timm.create_model('vit_base_patch16_224', num_classes=2, pretrained=False)
    initialize_weights(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-6)

    start_epoch = 0

    val_accs = []
    val_losses = []
    test_accs = []
    test_losses = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = train(train_loader, model, optimizer, criterion)
        # _, train_acc = validate(train_loader, model, criterion)
        val_loss, val_acc = validate(val_loader, model, criterion)
        test_loss, test_acc = validate(test_loader, model, criterion)
        print('Validation Loss: {:.2f}'.format(val_loss))
        print('Validation Top-1 Accuracy: {:.2f}%'.format(val_acc))
        print('Test Top-1 Accuracy: {:.2f}%'.format(test_acc))

        # save model
        # is_best = val_acc > best_acc
        # best_acc = max(val_acc, best_acc)
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print('Best Loss: {:.2f}'.format(best_loss))
        print('Best Accuracy: {:.2f}%'.format(best_acc))
        print('Best Epoch: {:d}'.format(best_epoch))

def train(train_loader, model, optimizer, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    for inputs_x, targets_x in tqdm(train_loader, desc="Training", leave=True):
        if torch.isnan(inputs_x).any() or torch.isinf(inputs_x).any():
            print("Inputs contain NaN or Inf after normalization!")

        # measure data loading time
        data_time.update(time.time() - end)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        logits = model(inputs_x)

        loss = criterion(logits, targets_x)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating", leave=True):
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Inputs contain NaN or Inf after normalization!")
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return (losses.avg, top1.avg)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()
