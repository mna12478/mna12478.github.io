---
title: NetVLAD CNN architecture for weakly supervised place recognition
date: 2019-11-07 21:20:14
tags:
- CNN
- Pooling
---
&emsp;&emsp;本文是为解决大规模visual place recognition问题，快速准确识别一个查询照片的位置，也就是说，给定一张图片A，要从其他图片中找到一张图片，这张中带有图片A中的地点，如下图所示。本文主要有三个贡献点：提出一个端到端可训练的卷积网络，核心部分NetVLAD是一个新的VLAD层，VLAD (Vector of Locally Aggregated Descriptors)是一个常用在图片检索中的图片表示方法；其次，基于弱监督排行loss提出一个训练策略；最后，提出的方法性能很好。
![](/images/NetVLAD/fig_exam.png "尽管query中有很多噪声，如人、汽车等，本文的方法依旧能正确识别")
# place recognition的深度网络
&emsp;&emsp;本文将place recognition看成图像检索问题来处理，带有未知地点的query图片用来可视化搜索大型地理标记图像数据库，top ranked的图片用来作为query的location的建议。通常要设计一个函数f作为图片表示提取器，给定图片I<sub>i</sub>，生成一个固定尺寸的向量f(I<sub>i</sub>)，函数f可以离线计算整个数据库{I<sub>i</sub>}的表示，在线提取query图片的表示f(q)，在测试时，通过查找与query最近的数据库图片来进行视觉搜索，或者近似最近邻查找，通过基于f(q)和f(I<sub>i</sub>)的欧式距离d(q, I<sub>i</sub>)。
&emsp;&emsp;大部分图片检索方法是基于提取局部描述子，然后以无序的方式pool描述子，这种方法对translation和partial occlusion有很强的鲁棒性。在本文提出的方法中，我们首先在最后的卷积层对CNN进行crop，并将其看成一个密集描述子提取器，那么若最后一个卷积层的输出为H\*W\*D，一系列在H\*W区域提取的D维描述子，然后基于VLAD设计一个新的pooling层，将提取的描述子pool为一个固定长度的图片表示，而且这层的参数可以通过反向传播来学习，我们将这个新的层成为NetVLAD层。
## NetVLAD：一个通用的VLAD层（f<sub>VLAD</sub>）
&emsp;&emsp;VLAD是一个常用的描述子pooling方法，常用于instance level检索和图片分类，能获取聚集在图像上的局部描述子的统计信息。bag-of-visual-words集成方法保留可视化单词的计数，VLAD存储每个可视化单词的残差（描述子与对应聚类中心的差异向量）的和，
&emsp;&emsp;给定N个D维的局部图像描述子{x<sub>i</sub>}作为输入，K个聚类中心（可视化单词）{c<sub>k</sub>}作为VLAD参数，输出的VLAD图像表示V是K\*D维的，为了方便，将V写成K\*D维的矩阵，但是这个矩阵是转换为向量，经过归一化之后，作为图像的表示，V中的元素(j,k)的计算方法如下所示，x<sub>i</sub>(j)和c<sub>k</sub>(j)分别是第j维的第i个描述子和第k个聚类中心，alpha<sub>k</sub>(x<sub>i</sub>)表示x<sub>i</sub>与第k个可视化单词之间的关系，如果聚类c<sub>k</sub>是最接近描述子x<sub>i</sub>的类，那么alpha=1，否则为0.V中每个D维的列k记录了分配给c<sub>k</sub>的、残差(x<sub>i</sub>-c<sub>k</sub>)的和，之后对矩阵V进行列方向的内部L2归一化，转化为向量，最后整体L2归一化。
![](/images/NetVLAD/func_VLAD.png "")
&emsp;&emsp;受益于图像检索其他方法，本文提出在CNN网络中模拟VLAD，设计一个可训练的通用VLAD层，即NetVLAD，结果是一个强大的可端到端训练的图像表示方法。但是网络中的层需要可微，而VLAD原本是不连续的，因为描述子x<sub>i</sub>到聚类中心c<sub>k</sub>的hard assignment值alpha<sub>k</sub>(x<sub>i</sub>)。为了使这个操作可微，我们将其改为描述子到多个聚类的soft assignment，表示方法如下所示，让权重与描述子x<sub>i</sub>与聚类中心c<sub>k</sub>的接近程度成正比，但是与其他聚类中心的接近程度相关，新的alpha的值分布在0和1之间。
![](/images/NetVLAD/func_soft.png "")
&emsp;&emsp;通过展开公式中的平方项，我们看到分子分母中的e<sup>-alpha||x<sub>i</sub>||<sup>2</sup></sup>可以抵消，得到以下形式的soft assgnment。
![](/images/NetVLAD/func_soft1.png "")
&emsp;&emsp;其中w<sub>k</sub>=2\*alpha\*c<sub>k</sub>, b<sub>k</sub>=-alpha\*||c<sub>k</sub>||<sup>2</sup>，那么NetVLAD层最终的形式如下所示。
![](/images/NetVLAD/func_soft2.png "")
&emsp;&emsp;。其中{w<sub>k</sub>}, {b<sub>k</sub>}, {c<sub>k</sub>}是聚类k中的可训练参数，与原始VLAD描述子类似，NetVLAD层集成了残差x<sub>i</sub>-c<sub>k</sub>在描述子空间不同part的第一顺序的统计特征，权重为描述子x<sub>i</sub>到聚类k的soft-assignment，即新的alpha<sub>k</sub>(x<sub>i</sub>)。但是与原始的VLAD参数{c<sub>k</sub>}相比，NetVLAD的参数有{w<sub>k</sub>}, {b<sub>k</sub>}, {c<sub>k</sub>}，这能促进更好的平滑性，如下图所示。红色和绿色的圈是来自两个不同图片的局部描述子，分配给了相同的聚类，在VLAD的编码下，它们对两幅图像之间相似性得分的贡献是对应残差之间的标量积（因为最终的VLAD向量是L2归一化的），残差向量是描述子与聚类锚点之间的差异，锚点c<sub>k</sub>可以看成一个对于特定聚类k的新的坐标系统的原点，在标准的VLAD中，锚点是作为聚类中心（x）来使残差均匀地分布在数据库中，但是在一个监督学习中，我们知道两个描述子属于不匹配的图片，所以可能会学习一个更好的锚点（\*），来使得新的残差的标量积更小。
![](/images/NetVLAD/fig_benefit.png "")
&emsp;&emsp;如下图所示，NetVLAD层可以看做一个元层，这个元层进一步分解为连接在有向非循环图中的CNN层。
![](/images/NetVLAD/fig_archi.png "")
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
--------未完待续