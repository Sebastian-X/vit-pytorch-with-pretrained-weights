import os
import sys
import _init_paths

import torch

import ipdb

def save_weight_dict(root_path, file_name, key, shape):
    file_path = os.path.join(root_path, file_name)
    fo = open(file_path, "a")
    string = key + '\t' + str(shape) + '\n'
    fo.write(string)
    fo.close()


if __name__ == '__main__':
    # !!!  whether to remain fc layers
    backbone = True

    # related input and output files
    weight_root_path = 'datasets/pretrained_models'
    ori_weight_file = 'jx_vit_base_p16_224-80ecf9dd.pth'
    if backbone:
        trans_weight_file = 'base_p16_224_backbone.pth'
    else:
        trans_weight_file = 'base_p16_224.pth'

    weight_txt_root_path = 'datasets/weight_txt'
    if backbone:
        weight_txt_file = 'trans_weight_backbone.txt'
    else:
        weight_txt_file = 'trans_weight.txt'



    ori_weight_path = os.path.join(weight_root_path, ori_weight_file)
    trans_weight_path = os.path.join(weight_root_path, trans_weight_file)

    ori_weight = torch.load(ori_weight_path)
    trans_weight = {}
    for key in ori_weight:
        div_key = key.split('.')
        if div_key[0] == 'norm':
            trans_key = 'mlp_head.0.' + div_key[-1]
        elif div_key[0] == 'head':
            trans_key = 'mlp_head.1.' + div_key[-1]
        elif div_key[0] == 'pos_embed':
            trans_key = 'pos_embedding'
        elif div_key[0] == 'patch_embed':
            trans_key = 'to_patch_embedding.0.' + div_key[-1]
        elif div_key[0] == 'blocks':
            if div_key[2] in ['norm1', 'attn']:
                sub_b = '0.'
            else:
                sub_b = '1.'

            prefix = 'transformer.layers.' + div_key[1] + '.' + sub_b
            mod_n = 'fn.'

            if div_key[2] in ['norm1', 'norm2']:
                mod_n += 'norm.'
            else:
                mod_n += 'fn.'
                if div_key[3] == 'qkv':
                    mod_n += 'to_qkv.'
                elif div_key[3] == 'proj':
                    mod_n += 'to_out.0.'
                elif div_key[3] == 'fc1':
                    mod_n += 'net.0.'
                elif div_key[3] == 'fc2':
                    mod_n += 'net.3.'
                else:
                    assert ValueError('This should not happen: \'{}\''.format(key))
            
            trans_key = prefix + mod_n + div_key[-1]
        else:
            trans_key = key

        print('{}  ->  {}'.format(key, trans_key))
        '''
        if trans_key == 'to_patch_embedding.1.weight':
            # ipdb.set_trace()
            tmp = ori_weight[key].data.contiguous().view(ori_weight[key].shape[0], -1)
            trans_weight[trans_key] = tmp
        else:
            trans_weight[trans_key] = ori_weight[key]
        '''
        if backbone:
            if 'mlp_head.1' in trans_key:
                continue
        trans_weight[trans_key] = ori_weight[key]

    torch.save(trans_weight, trans_weight_path)
    print('\nTrans_weight saved to \'{}\''.format(trans_weight_path))

    for key in trans_weight:
        save_weight_dict(weight_txt_root_path, weight_txt_file, key, trans_weight[key].shape)
    print('Weight log saved to \'{}\''.format(os.path.join(weight_txt_root_path, weight_txt_file)))
