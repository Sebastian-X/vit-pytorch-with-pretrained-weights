import os
import sys
import _init_paths

import torch
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
    parser.add_argument('--data_path', default='./datasets/imagenet/', type=str, help='path to ImageNet data')
    parser.add_argument('--ckpt_path', default='datasets/pretrained_models/base_p16_224.pth', type=str, help='path to checkpoint')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader') 
    parser.add_argument('--crop_pct', default=0.9, type=float, help='crop ratio')
    parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation method')

    # model parameters
    parser.add_argument('--input_size', default=224, type=int, help='size of input')
    parser.add_argument('--patch_size', default=16, type=int, help='size of patch') 
    parser.add_argument('--num_classes', default=1000, type=int, help='num_classes') 
    parser.add_argument('--dim', default=768, type=int, help='dim') 
    parser.add_argument('--depth', default=12, type=int, help='depth') 
    parser.add_argument('--heads', default=12, type=int, help='heads') 
    parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp_dim') 
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout') 
    parser.add_argument('--emb_dropout', default=0.1, type=float, help='emb_dropout') 
    parser.add_argument('--qkv_bias', default=True, type=bool, help='use qkv_bias')

    # other parameters
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--gpu', default='0', type=str, help='set gpu')

    args = parser.parse_args()

    print('Called With Args:')
    for k,v in sorted(vars(args).items()):
        print('    ', k,'=',v)
    print()

    seed_all(args.seed)
    set_gpu(args.gpu)


    # build validation dataset
    data_path = args.data_path
    ckpt_path = args.ckpt_path
    val_dir = os.path.join(data_path, 'val')
    batch_size = args.batch_size
    workers = args.workers
    img_size = args.input_size # set img_size = input_size
    crop_pct = args.crop_pct
    interpolation = args.interpolation
    
    scale_size = int(math.floor(img_size / crop_pct))
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    transform = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
            ])

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    

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
    print('Loading Weights from \'{}\''.format(ckpt_path))
    print()
    weight = torch.load(ckpt_path)
    # v.load_state_dict(weight)
    load_partial_weight(v, weight)
    v.cuda()
    

    # run test on imagenet
    print('Testing ViT on ImageNet')
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