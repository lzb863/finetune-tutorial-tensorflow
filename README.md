# Finetune-tutorial-tensorflow

该教程用于记录如何利用tensorflow进行finetune.教程基于Semantic Segmentation任务,采用[deeplab v3](https://arxiv.org/pdf/1706.05587v1.pdf)模型完成.

## 前言
该项目的idea来自如何用原生的tensorflow来实现finetune.<br>
其实finetune本身并不是一项很复杂的工作,无论是对于[pytorch](https://pytorch.org/), [Keras](https://keras-cn.readthedocs.io/en/latest/)还是基于tensorflow的高级库[slim](https://github.com/tensorflow/models/tree/master/research/slim)来说,finetune都可以通过简单的几行代码实现.但是,一个前提是所定义的模型必须和预训练模型文件相匹配.所以,当我想要用tensorflow实现finetune的时候,第一选择就是slim,这样的话我就必须接受slim的代码,比如[ResNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py),看了代码就能感受到,风格真是一言难尽.这里就要强推一波Pytorch了,直接上Pytorch版的[ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)代码,简洁,模块化,逻辑分明,就是我想要的风格!<br>
所以,利用tensorflow的预训练模型,模型文件可以在[这里](https://github.com/tensorflow/models/tree/master/research/slim)下载,自己定义相对应的模型,最终使模型与预训练模型文件(.ckpt或.npy)是促成该教程的原因.

## TODO

## Reference
* [Finetuning AlexNet with TensorFlow](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html)
