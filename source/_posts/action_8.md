---
title: Convolutional Two-Stream Network Fusion for Video Action Recognition (CVPR 2016)
date: 2019-10-26 21:07:29
categories: 动作识别
tags:
- 双流网络
- 时空融合
---
&emsp;&emsp;最近的动作识别方法多是基于CNN结构来提出不同的解决方案，本文探索了几种融合卷积网络的方法，得到以下结论：在卷积层上进行时空融合，而非softmax层，既不会降低性能，还能减少参数；在最后卷积层进行（spatially）融合网络比在之前的网络中融合的效果好，在类别预测层上的融合可以提高精度；沿周边的时空特征池化抽象的卷积特征能进一步提升性能。基于以上结论，本文提出了一个新的时空特征融合的CNN网络，[代码](https://github.com/feichtenhofer/twostreamfusion)是基于MATLAB来写的。
# 相关工作
&emsp;&emsp;本文指出，实际上在UCF-101和HMDB51上，目前最好的方法是卷积网络与基于轨迹手动提取特征的Fisher Vector encoding方法。可能因为目前用于训练的数据库规模还比较小，或包含的噪声较多；另一个原因是当前的卷积网络不能完全使用temporal information的优势，它们的性能常常被spatial（外观）识别所主导，在识别动作时大多是依靠spatial特征来进行识别。
&emsp;&emsp;像之前Early Fuson、Late Fusion的方法，对时间并不是很敏感，只是通过纯粹的spatial网络达到了类似的效果，这表明并没有从时间信息中获得更多；C3D模型比前一个方法深得多，结构类似于之前一个非常深的网络；将3D卷积分解成2D空间卷积+1D时间卷积的方法，它的时间卷积是一个随时间和特征通道而来的2D卷积，而且只在网络更高的层上进行。
&emsp;&emsp;而与本文最接近的，也是本文基于进行扩展的方法，是双流网络，该方法是将深度学习应用于动作识别，特别是在有限的数据集的情况下，最有效的方法；另一个相关的方法是双线性方法，Bilinear CNN models for fine-grained visual recognition (ICCV 2015)，通过在图像每个位置的外籍来关联两个卷积层的输出，在所有位置池化产生的双线性特征，形成一个无序的描述子。
&emsp;&emsp;在数据集方面，Sports-1M数据库是自动收集的视频，可能包含标签噪声；另一个大型数据库是THUMOS，有超过4500万帧，但只有一小部分包含对监督学习有用的labelled action。由于以上标签噪声，学习时空特征的卷积网络依旧很大程度上依赖于更小的，但是时序一致的数据库，如UCF-101或者HMDB-51，这有助于学习，也有可能产生严重的过拟合。
# 网络结构
\--------未完待续
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
