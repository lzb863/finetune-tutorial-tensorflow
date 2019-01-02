# Finetune-tutorial-tensorflow

该教程用于记录如何利用tensorflow进行finetune.教程基于Semantic Segmentation任务,采用[deeplab v3](https://arxiv.org/pdf/1706.05587v1.pdf)模型完成.

## 前言
该项目的idea来自如何用原生的tensorflow来实现finetune.<br>
其实finetune本身并不是一项很复杂的工作,无论是对于[pytorch](https://pytorch.org/), [Keras](https://keras-cn.readthedocs.io/en/latest/)还是基于tensorflow的高级库[slim](https://github.com/tensorflow/models/tree/master/research/slim)来说,finetune都可以通过简单的几行代码实现.但是,一个前提是所定义的模型必须和预训练模型文件相匹配.所以,当我想要用tensorflow实现finetune的时候,第一选择就是slim,这样的话我就必须接受slim的代码,比如[ResNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py),看了代码就能感受到,风格真是一言难尽.这里就要强推一波Pytorch了,直接上Pytorch版的[ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)代码,简洁,模块化,逻辑分明,就是我想要的风格!<br>
所以,利用tensorflow的预训练模型,模型文件可以在[这里](https://github.com/tensorflow/models/tree/master/research/slim)下载,自己定义相对应的模型,最终使模型与预训练模型文件(.ckpt或.npy)是促成该教程的原因.

## 准备
实现finetune的两个必备元素:预训练模型与模型定义.
其中tensorflow的预训练模型可以在[这里](https://github.com/tensorflow/models/tree/master/research/slim),同时模型的定义也有相应的定义,但是由于我们想自己定义模型,所以关于模型的定义可以依照自己的习惯编写.<br>
但是无论如何,需要保证的是,预训练模型中的变量名称空间需要和自己定义模型的变量的模型空间保持一致,从而可以正常匹配变量并完成赋值.<br>
对于给定的.ckpt文件,获取其中变量名的方式如下:
```
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader('../pretrained_models/resnet_v2_50/resnet_v2_50.ckpt')
weights_dict = reader.get_variable_to_shape_map()
for op_name in weights_dict:
  print(op_name)
  print(reader.get_tensor(op_name))
```
使得预训练模型中的变量名称空间和自己定义模型保持匹配是一件必要且繁琐的事,请务必保持耐心.

## 实现
当模型定义完成之后,接下来就应该是如何在训练的时候直接使用预训练模型中的变量.该教程中的做法是实现一个完成权重初始化的函数,具体可见[这里](https://github.com/Tramac/finetune-tutorial-tensorflow/blob/master/models/resnet.py#121)<br>
需要说明的是,代码中我写了两个版本的初始化函数,并且第二个版本另包含两个版本.思路分别是:<br>
1.以预训练模型中的变量名为引导,通过遍历预训练模型中的变量,完成所有匹配赋值.<br>
此为version 1.但是由于slim中提供的预训练模型中变量实在是太多,虽然目前还未搞清楚是否冗余,但是比Pytorch的预训练模型的参数多了不少,所以教程中并没有使用该版本,但是该版本是最合理的解决方案,如果允许应该成为首选.详细代码可见[这里](https://github.com/Tramac/finetune-tutorial-tensorflow/blob/master/models/resnet.py#122)<br>
2.以自定义模型中的变量为引导,通过变量自定义模型中的变量,完成所有匹配赋值.<br>
此为version 2.首先获取到自定义模型的所有参数,通过遍历参数,然后从预训练模型中取出变量值完成匹配,此为version2.1,但是自己定义的模型的变量名并没有和预训练中的变量名完全匹配(这种情况是不允许的,一定要保持完全一致),所以version 2.1并不能完成运行.所以,version 2.2中对此做出修改,是目前可以执行的方案,但是存在漏洞,并没有完成所有变量的匹配,后续会做出改进.

## TODO
目前版本并没有最合理的实现,只是提供了一个通用的思路,后续需要更新实现并完成验证.

## Reference
* [Finetuning AlexNet with TensorFlow](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html)
