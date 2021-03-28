import os.path as osp
import sys
import ipdb

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
path = '/home/xueruixin/zxc/vit-pytorch-main/'
add_path(path)
