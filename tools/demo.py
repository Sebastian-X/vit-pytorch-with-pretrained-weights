import os
import sys
import _init_paths

import torch
from vit_pytorch_loc.vit_pytorch import ViT
import ipdb
import os

def save_weight_dict(root_path, file_name, key, shape):
    file_path = os.path.join(root_path, file_name)
    fo = open(file_path, "a")
    string = key + '\t' + str(shape) + '\n'
    fo.write(string)
    fo.close()

v = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

ckpt_path = 'datasets/pretrained_models/jx_vit_base_p16_224-80ecf9dd.pth'
weight = torch.load(ckpt_path)
img = torch.randn(1, 3, 256, 256)
mask = torch.ones(1, 8, 8).bool() # optional mask, designating which patch to attend to

print(v)
# preds = v(img, mask = mask) # (1, 1000)
for key in weight:
    save_weight_dict('datasets/weight_txt', 'ckpt_weight_keys.txt', key, weight[key].shape)
for key in v.state_dict():
    save_weight_dict('datasets/weight_txt', 'model_keys.txt', key, v.state_dict()[key].shape)
