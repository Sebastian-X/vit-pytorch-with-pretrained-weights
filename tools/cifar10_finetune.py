import os
import sys
import _init_paths

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from vit_pytorch_loc.vit_pytorch import ViT
from utils.utils import set_gpu, seed_all, _pil_interp, load_partial_weight

from tqdm import tqdm
import argparse
import math
import ipdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    # data parameters
    parser.add_argument('--data_path', default='./datasets/cifar10/', type=str, help='path to ImageNet data')
    parser.add_argument('--ckpt_path', default='datasets/pretrained_models/base_p16_224_backbone.pth', type=str, help='path to checkpoint')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader') 
    parser.add_argument('--crop_pct', default=0.9, type=float, help='crop ratio')
    parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation method')

    # model parameters
    parser.add_argument('--input_size', default=224, type=int, help='size of input')
    parser.add_argument('--patch_size', default=16, type=int, help='size of patch') 
    parser.add_argument('--num_classes', default=10, type=int, help='num_classes') 
    parser.add_argument('--dim', default=768, type=int, help='dim') 
    parser.add_argument('--depth', default=12, type=int, help='depth') 
    parser.add_argument('--heads', default=12, type=int, help='heads') 
    parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp_dim') 
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout') 
    parser.add_argument('--emb_dropout', default=0.1, type=float, help='emb_dropout') 
    parser.add_argument('--qkv_bias', default=True, type=bool, help='use qkv_bias')

    # training parameters
    parser.add_argument('--max_epoch', default=200, type=int, help='max epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--val_per', default=1, type=int, help='validate per epochs')
    parser.add_argument('--val_begin',  action='store_true', help='validate before training')


    # other parameters
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--gpu', default='0', type=str, help='gpu')

    args = parser.parse_args()

    print('Called With Args:')
    for k,v in sorted(vars(args).items()):
        print('    ', k,'=',v)
    print()

    seed_all(args.seed)
    set_gpu(args.gpu)


    # build validation dataset
    data_path = args.data_path
    batch_size = args.batch_size
    workers = args.workers
    img_size = args.input_size # set img_size = input_size
    crop_pct = args.crop_pct
    interpolation = args.interpolation
    
    scale_size = int(math.floor(img_size / crop_pct))
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            ])
    val_transform = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # normalize,
            ])

    train_dataset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    # ipdb.set_trace()
    

    # build ViT model
    input_size = args.input_size
    patch_size = args.patch_size
    num_classes = args.num_classes
    dim = args.dim 
    depth = args.depth
    heads = args.heads
    mlp_dim = args.mlp_dim
    dropout = args.dropout
    emb_dropout = args.emb_dropout
    qkv_bias = args.qkv_bias

    v = ViT(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout,
        qkv_bias= qkv_bias
    )
    print('Building ViT Model:\n{}'.format(v))
    print()

    # load weight
    ckpt_path = args.ckpt_path
    print('Loading Weights from \'{}\''.format(ckpt_path))
    print()
    weight = torch.load(ckpt_path)
    load_partial_weight(v, weight)
    v.cuda()
    
    # build optimizer
    max_epoch = args.max_epoch
    val_per_epoch = args.val_per
    lr = args.lr

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(v.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_epoch/4,eta_min=0.0003)

    # validate before training
    if args.val_begin:
        print('Validating before Training')
        v.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc='Validating'):
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()
                output = v(imgs)
                _,predict_labels = torch.max(output.data,1)
                predict_labels = predict_labels.view(-1)
                correct+= torch.sum(torch.eq(predict_labels,labels)).item()
                total+=len(labels)
            print('Validated on {} Images, Accuracy: {}%'.format(total, correct/total*100.0))
            print()

    # run train on cifar10
    for epoch in range(1, max_epoch+1):
        v.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_data_num = 0
        total_train_correct = 0
        for data in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            output = v(imgs)

            loss = criterion(output, labels)
            total_train_loss += loss * imgs.shape[0]
            total_data_num += imgs.shape[0]
            _,predict_labels = torch.max(output.data,1)
            predict_labels = predict_labels.view(-1)
            total_train_correct += torch.sum(torch.eq(predict_labels,labels)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_loss /= total_data_num
        total_train_acc = total_train_correct / total_data_num * 100
        print('Training Loss: {}, Training Acc: {}%'.format(total_train_loss, total_train_acc))
        # run validation
        if (epoch%val_per_epoch==0):
            v.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in tqdm(val_loader, desc='Validating'):
                    imgs, labels = data
                    imgs, labels = imgs.cuda(), labels.cuda()
                    output = v(imgs)
                    _,predict_labels = torch.max(output.data,1)
                    predict_labels = predict_labels.view(-1)
                    correct+= torch.sum(torch.eq(predict_labels,labels)).item()
                    total+=len(labels)
                print('Validated Epoch {} on {} Images, Accuracy: {}%'.format(epoch, total, correct/total*100.0))
                # print('Final Accuracy: %f%%'%(correct/total*100.0))
                
    # run test on cifar10
    print()
    print('Training finished')
    print('Testing ViT on Cifar10 testset')
    v.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            output = v(imgs)
            _,predict_labels = torch.max(output.data,1)
            predict_labels = predict_labels.view(-1)
            correct+= torch.sum(torch.eq(predict_labels,labels)).item()
            total+=len(labels)
        print('Tested on {} Images'.format(total))
        print('Final Accuracy: %f%%'%(correct/total*100.0))