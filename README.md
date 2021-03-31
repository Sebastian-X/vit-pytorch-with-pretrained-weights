### 带有预训练权重的ViT pytorch实现



#### 简介

本代码在<a href="https://github.com/lucidrains/vit-pytorch">lucidrains</a>实现的ViT基础上修改，适配了<a href="ttps://github.com/rwightman/pytorch-image-models">Ross Wightman</a>从官方JAX库中提取出的权重文件，并添加了转换权重文件以及简单的test、fine-tune程序。



#### 环境要求

代码在pytorch 1.6.0 + torchvision 0.7.0 + cudatoolkit 10.1环境下可以正常运行。



#### 数据集准备

代码支持在ImageNet上的测试和在Cifar10上的fine-tune。

准备数据集时，可以直接利用``ln -s``命令将数据集软链接至`./datasets/`路径下，链接名分别为`imagenet`和`cifar10`。当然也可以在文件中修改路径参数为数据集所在的路径（麻烦，不推荐）。



#### 预训练模型准备

Ross Wightman从官方JAX库中转换为支持pytorch的`.pth`权重文件可以在<a href="https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-vitjx">这里</a>下载得到。因为下载需要翻墙，且速度很不稳定，我把自己已经下好的ViT-base-p16-224权重文件上传至<a href="https://pan.baidu.com/s/1vQN-J2XJ8wJ9GGtteV4PhQ">百度云</a>，提取码为8zi4，大家需要自取。

下载得到权重文件后，需要利用`./tools/trans_weight.py`对其进行转换，才能顺利加载到模型中。具体用法请见下文。



#### 用法

主要的可执行文件均在路径`./tools/`下：

- `./tools/demo.py`：测试模型的demo

  测试搭建模型的过程。执行后会在`./datasets/weight_txt/`路径下保存模型和权重的state_dict（即参数名及其尺寸），可以用于确定权重能否被模型正常读取。

- `./tools/trans_weight.py`：权重转换

  转换下载好的权重，使其可以被正常加载进模型。需要注意，代码可能需要根据权重文件的不同进行额外调整。`backbone`变量决定了转换模型时是否抛弃用于分类的fc层的参数。

- `./tools/imagenet_test.py`：在imagenet上测试

- `./tools/cifar10_finetune.py`：在cifar10上训练并验证



在使用时，请务必注意args参数的设置，并注意训练时可能覆盖上一次训练保存的checkpoint，如有需要请另外设置保存checkpoint的路径（用`--save_path`）！



##### Tips

- 利用`tee`命令将终端显示信息另存至一个文本文件。例：

  ```
  python ./tools/imagenet_test --gpu 3 --batch_size 512 | tee logs/test.log
  ```

- 利用bash脚本文件省去每次运行都需要设置参数的烦恼。如编写`finetune.sh`文件如下：

  ```
  nohup python ./tools/cifar10_finetune.py --gpu 0 --save_path ./save/save1 --val_per 5 \
  --max_epoch 100 --lr 1e-2 >output.txt 2>&1 &
  ```

  那么在训练时只需执行：

  ```
  bash finetune.sh
  ```



##### To Do List

- 训练调优

  训练未经调优，如有兴趣或需要，请自行调优。

  - 超参数
  - optimizer
  - scheduler（学习率衰减）

- log文件输出

- 训练信息保存

- 多GPU分布式训练





