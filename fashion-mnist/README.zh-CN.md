# Fashion-MNIST

[![GitHub stars](https://img.shields.io/github/stars/zalandoresearch/fashion-mnist.svg?style=flat&label=Star)](https://github.com/zalandoresearch/fashion-mnist/)
[![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)
[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](README.md)
[![Readme-JA](https://img.shields.io/badge/README-日本語-green.svg)](README.ja.md)


FashionMNIST是一个替代[MNIST手写数字集](http://yann.lecun.com/exdb/mnist/)的图像数据集。 它是由Zalando（一家德国的时尚科技公司）旗下的[研究部门](https://research.zalando.com/)提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。FashionMNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致。60000/10000的训练测试数据划分，28x28的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，且**不需要**改动任何的代码。

这个数据集的样子大致如下（每个类别占三行）：

![](doc/img/fashion-mnist-sprite.png)

<img src="doc/img/embedding.gif" width="100%">

## 为什么要做这个数据集？

[经典的MNIST数据集](http://yann.lecun.com/exdb/mnist/)包含了大量的手写数字。十几年来，来自机器学习、机器视觉、人工智能、深度学习领域的研究员们把这个数据集作为衡量算法的基准之一。你会在很多的会议，期刊的论文中发现这个数据集的身影。实际上，MNIST数据集已经成为算法作者的必测的数据集之一。有人曾调侃道：*"如果一个算法在MNIST不work, 那么它就根本没法用；而如果它在MNIST上work, 它在其他数据上也可能不work！"*
 

`Fashion-MNIST`的目的是要成为MNIST数据集的一个直接替代品。作为算法作者，你不需要修改任何的代码，就可以直接使用这个数据集。`Fashion-MNIST`的图片大小，训练、测试样本数及类别数与经典MNIST**完全相同**。

### 写给专业的机器学习研究者

我们是认真的。取代MNIST数据集的原因由如下几个：

- **MNIST太简单了。** 很多深度学习算法在测试集上的准确率已经达到99.6%！不妨看看[我们基于scikit-learn上对经典机器学习算法的评测](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) 和这段代码： ["Most pairs of MNIST digits can be distinguished pretty well by just one pixel"（翻译：大多数MNIST只需要一个像素就可以区分开！）](https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478)
- **MNIST被用烂了。** 参考：["Ian Goodfellow wants people to move away from mnist"（翻译：Ian Goodfellow希望人们不要再用MNIST了。）](https://twitter.com/goodfellow_ian/status/852591106655043584)
- **MNIST数字识别的任务不代表现代机器学习。** 参考：["François Cholle: Ideas on MNIST do not transfer to real CV" （翻译：在MNIST上看似有效的想法没法迁移到真正的机器视觉问题上。）](https://twitter.com/fchollet/status/852592598128615424)

## 获取数据

你可以使用以下链接下载这个数据集。`Fashion-MNIST`的数据集的存储方式和命名与[经典MNIST数据集](http://yann.lecun.com/exdb/mnist/)完全一致。

| 名称  | 描述 | 样本数量 | 文件大小 | 链接
| --- | --- |--- | --- |--- |
| `train-images-idx3-ubyte.gz`  | 训练集的图像  | 60,000|26 MBytes | [下载](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|
| `train-labels-idx1-ubyte.gz`  | 训练集的类别标签  |60,000|29 KBytes | [下载](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|
| `t10k-images-idx3-ubyte.gz`  | 测试集的图像  | 10,000|4.3 MBytes | [下载](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|
| `t10k-labels-idx1-ubyte.gz`  | 测试集的类别标签  | 10,000| 5.1 KBytes | [下载](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|

或者，你可以直接克隆这个代码库。数据集就放在`data/fashion`下。这个代码库还包含了一些用于评测和可视化的脚本。
   
```bash
git clone git@github.com:zalandoresearch/fashion-mnist.git
```

### 类别标注
每个训练和测试样本都按照以下类别进行了标注：

| 标注编号 | 描述 |
| --- | --- |
| 0 | T-shirt/top（T恤）|
| 1 | Trouser（裤子）|
| 2 | Pullover（套衫）|
| 3 | Dress（裙子）|
| 4 | Coat（外套）|
| 5 | Sandal（凉鞋）|
| 6 | Shirt（汗衫）|
| 7 | Sneaker（运动鞋）|
| 8 | Bag（包）|
| 9 | Ankle boot（踝靴）|

## 如何载入数据？

### 使用Python (需要安装`numpy`)
- 你可以直接使用`utils/mnist_reader`：
```python
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
```

### 使用Tensorflow
请确保你已经[下载了我们的数据集](#获取数据)并把它放到了`data/fashion`下。不然， *Tensorflow会自动下载并使用原始的MNIST。*

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')

data.train.next_batch(100)
```

### 使用其它机器学习库
- [Apache MXNet Gluon (master ver.)](https://mxnet.incubator.apache.org/versions/master/api/python/gluon.html#vision)
- [deeplearn.js](https://pair-code.github.io/deeplearnjs/demos/model-builder/model-builder-demo.html)
- [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)

欢迎你同我们一起，为各个机器学习库增加对`Fashion-MNIST`的支持。


### 使用其它的语言

作为机器学习领域里最常使用的数据集，人们用各种语言为MNIST开发了很多载入工具。有一些方法需要先解压数据文件。注意，我们并没有测试过所有的载入方法。

- [C](https://stackoverflow.com/a/10409376)
- [C++](https://github.com/wichtounet/mnist)
- [Java](https://stackoverflow.com/a/8301949)
- [Python](https://pypi.python.org/pypi/python-mnist)和[这里](https://pypi.python.org/pypi/mnist)
- [Scala](http://mxnet.io/tutorials/scala/mnist.html)
- [Go](https://github.com/schuyler/neural-go/blob/master/mnist/mnist.go)
- [C#](https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/)
- [NodeJS](https://github.com/ApelSYN/mnist_dl)和[这里](https://github.com/cazala/mnist)
- [Swift](https://github.com/simonlee2/MNISTKit)
- [R](https://gist.github.com/brendano/39760)和[这里](https://github.com/maddin79/darch)
- [Matlab](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset)和[这里](https://de.mathworks.com/matlabcentral/fileexchange/27675-read-digits-and-labels-from-mnist-database?focused=5154133&tab=function)
- [Ruby](https://github.com/gbuesing/mnist-ruby-test/blob/master/train/mnist_loader.rb)


## 基准测试
我们使用`scikit-learn`做了一套自动评测系统。它涵盖了除深度学习之外的129种经典机器学习模型（包含不同的参数）。[你可以在这里以互动的方式查看结果。](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)

<img src="doc/img/benchmark.gif" width="100%">

你可以运行`benchmark/runner.py`对结果进行重现。而我们更推荐的方法是使用Dockerfile打包部署后以Container的方式运行。

我们欢迎你提交自己的模型评测。请使用Github新建一个Issue。不妨先看看[如何贡献](https://github.com/zalandoresearch/fashion-mnist#参与贡献)。如果你提交自己的模型，请先确保这个模型没有[在这个列表中](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)被测试过。

下面这个表格总结了提交的一些测试结果。注意，**我们并没有对这些结果的准确性进行验证**。你可以通过提交者附带的代码尝试对结果进行重现。当然，测试准确率最终取决于Epoch的多少，Batch的大小等因素。如果你发现了下表中的不妥，欢迎提交新的Issue。

| 算法  | 预处理 | Fashion测试集准确率| 经典MNIST测试集准确率| 提交者| 代码|
| --- | --- | --- | --- | --- |--- |
|2 Conv Layers with max pooling (Keras) | None | 0.876 | - | [Kashif Rasul](https://twitter.com/krasul) | [:link:](https://gist.github.com/kashif/76792939dd6f473b7404474989cb62a8) |
|2 Conv Layers with max pooling (Tensorflow) >300 epochs | None | 0.916| - |[Tensorflow's doc](https://www.tensorflow.org/tutorials/layers) | [:link:](/benchmark/convnet.py)|
|2 Conv Layers net | Normalization, random horizontal flip, random vertical flip, random translation, random rotation. | 0.919 |0.971 | [Kyriakos Efthymiadis](https://github.com/kefth)| [:link:](https://github.com/kefth/fashion-mnist)|
|2 Conv Layers net <100K parameter | None | 0.925 | 0.992 |[@hardmaru](https://twitter.com/hardmaru) | [:link:](https://github.com/hardmaru/pytorch_notebooks/blob/master/pytorch_tiny_custom_mnist_adam.ipynb)| 
|3 Conv layers and 2 FC | None | 0.907 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|GRU+SVM | None| 0.888 | 0.965 | [@AFAgarap](https://github.com/AFAgarap) | [:link:](https://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/828fbda0e466dacb1fad66549e0e3022e1c7263a/gru_svm_zalando.py)|
|GRU+SVM with dropout | None| 0.897 | 0.976 | [@AFAgarap](https://github.com/AFAgarap) | [:link:](https://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/58dbe7cd8b0d83e4386cd6896766113b1a9af096/gru_svm_zalando_dropout.py)|
|WRN40-4 8.9M params | standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips)| 0.967 | - |[@ajbrock](https://github.com/ajbrock) | [:link:](https://github.com/xternalz/WideResNet-pythttps://gist.githubusercontent.com/AFAgarap/92c1c4a5dd771999b0201ec0e7edfee0/raw/58dbe7cd8b0d83e4386cd6896766113b1a9af096/gru_svm_zalando_dropout.pyorch)  [:link:](https://github.com/ajbrock/FreezeOut) |
|DenseNet-BC 768K params| standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) | 0.954 | - |[@ajbrock](https://github.com/ajbrock)  | [:link:](https://github.com/bamos/densenet.pytorch)  [:link:](https://github.com/ajbrock/FreezeOut) |
|MobileNet | augmentation (horizontal flips)| 0.950|- | [@苏剑林](https://github.com/bojone)| [:link:](http://kexue.fm/archives/4556/)|
|ResNet18 | Normalization, random horizontal flip, random vertical flip, random translation, random rotation. | 0.949 | 0.979 |[Kyriakos Efthymiadis](https://github.com/kefth)| [:link:](https://github.com/kefth/fashion-mnist)|
|GoogleNet with cross-entropy loss | None | 0.937 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|AlexNet with Triplet loss| None | 0.899 | - | [@Cenk Bircanoğlu](https://github.com/cenkbircanoglu) | [:link:](https://github.com/cenkbircanoglu/openface/tree/master/fashion_mnist)|
|SqueezeNet with cyclical learning rate 200 epochs| None| 0.900| - | [@snakers4](https://github.com/snakers4) | [:link:](https://github.com/zalandoresearch/fashion-mnist/files/1263340/squeeze_net_mnist.zip)|
|Dual path network with wide resnet 28-10|standard preprocessing (mean/std subtraction/division) and augmentation (random crops/horizontal flips) |0.957|-|[@Queequeg](https://github.com/Queequeg92)|[:link:](https://github.com/Queequeg92/DualPathNet)|
|MLP 256-128-64| None | 0.900| - | [@lianghong](https://github.com/lianghong)| [:link:](https://github.com/lianghong/fashion_mnist-on-mxnet)| 
|VGG16 26M parameters | None | 0.935| - | [@QuantumLiu](https://github.com/QuantumLiu)|[:link:](https://github.com/QuantumLiu/fashion-mnist-demo-by-Keras) [:link:](https://zhuanlan.zhihu.com/p/28968219)|

### 更多在Fashion-MNIST上的探索和尝试

#### 生成对抗网络 (GANs) 
- [Tensorflow implementation of various GANs and VAEs.](https://github.com/hwalsuklee/tensorflow-generative-model-collections) (**推荐阅读！** 注意不同GANs的算法在Fashion-MNIST上生成的样本明显不同，而这点在经典的MNIST数据集上是观察不到的。)
- [Make a ghost wardrobe using DCGAN](https://twitter.com/spaceLenny/status/901488938023403520)
- [fashion-mnist的gan玩具](http://kexue.fm/archives/4540/)
- [CGAN output after 5000 steps](https://github.com/a7b23/Conditional-GAN-using-tensorflow-slim)


## 数据可视化

### t-SNE在Fashion-MNIST(左侧)和经典MNIST上的可视化(右侧) 
<img src="doc/img/34d72c08.png" width="50%"><img src="doc/img/01e0c4be.png" width="50%">

### PCA在Fashion-MNIST(左侧)和经典MNIST上的可视化(右侧) 
<img src="doc/img/f04ba662.png" width="50%"><img src="doc/img/4433f0e1.png" width="50%">


## 参与贡献
我们热烈欢迎您参与贡献这个项目。[请先阅读这里！](/CONTRIBUTING.md) 并查看有什么[open issues](https://github.com/zalandoresearch/fashion-mnist/issues)可以帮助解决。

## 联系
若要讨论这个数据集上的应用和评测，请使用这个聊天室[![Gitter](https://badges.gitter.im/zalandoresearch/fashion-mnist.svg)](https://gitter.im/fashion-mnist/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)


## 在论文中引用Fashion-MNIST
如果你在你的研究工作中使用了这个数据集，欢迎你引用这篇论文：

**Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. [arXiv:1708.07747](http://arxiv.org/abs/1708.07747)**

亦可使用Biblatex:
```latex
@online{xiao2017/online,
  author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  date         = {2017-08-28},
  year         = {2017},
  eprintclass  = {cs.LG},
  eprinttype   = {arXiv},
  eprint       = {cs.LG/1708.07747},
}
```

## License

The MIT License (MIT) Copyright © [2017] Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
