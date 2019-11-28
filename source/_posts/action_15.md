---
title: An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition (CVPR 2019)
date: 2019-11-28 20:50:35
categories: 
- 动作识别
- 骨架
tags:
---
&emsp;&emsp;本文提出了一种基于骨架的动作识别方法，Attention Enhanced Graph Convolutional LSTM Network (AGC-LSTM)，不仅能提取空间和时间的判别性特征，而且能探索其中的相互关系；还提出了一种时序分级结构来增加模型顶层的时序感受野，提高学习高级语义表示的能力，而且减少计算量；另外，为了选择有判别性的空间信息，使用attention机制来增加每个层关键关节点的信息。
&emsp;&emsp;动作识别的方法有很多，有些是基于RGB视频，有些是基于3D骨架数据。基于RGB视频的方法主要是从RGB图像和光流数据中建模时空表示，但是这种方法在处理背景复杂、亮度变化和外观变化场景时会有局限性；3D骨架用一系列关键关节点的3D坐标位置来表示身体结构，尽管不包含颜色信息，但是没有RGB视频所受的那些限制，允许建模更有判别性的时序特点，而且有研究和实验（Visual perception of biological motion and a model for its analysis）表明关键关节点能提供人类动作的高度有效的信息，而且现在有很多算法能轻易获取到人的关节点数据。
&emsp;&emsp;人的骨架序列一般有三个特点，每个节点和与之相连的关节点之间强相关，骨架frame包含丰富的身体结构信息；时间上的连续性不仅存在于相同的关节（例如手，腕和肘部）中，而且还存在于身体结构中；时间和空间领域存在相互关系。
&emsp;&emsp;基于图的模型主要分为两种结构，一种为GNN (Graph Neural Network)，是图和RNN的组合，经过节点的多次信息传递和状态更新，每个节点获取其相邻节点的语义关系和结构信息，可以用来检测和识别人机交互、环境识别；另一种结构是GCN (Graph Convolutional Network)，将卷积神经网络扩展到图，一般有两种类型的GCN：spectral GCN和spatial GCN，第一种将图信号转换到图谱领域，并在频谱域应用频谱滤波器，比如依靠图的拉普拉斯在频谱领域使用CNN；另一种是基于卷积操作，使用每个节点及其相邻节点，计算一个新的特征向量
&emsp;&emsp;本文提出的AGC-LSTM模型如下图所示，首先将每个关节点的坐标用线性层转换成时空特征，然后将时空特征和每两个连续帧之间的特征差异组合，得到一个扩增的特征，然后用LSTM处理每个关节点序列，消除特征差异和位置特征之间的比例差异，接着使用三个AGC-LSTM层建模时空特征。
![](/images/AGC/fig_archi.png "Feature augmentation (FA)计算具有位置特征的特征差异，并将位置特征和特征差异组合，LSTM用于消除特征差异和位置特征之间的比例差异，三个AGC-LSTM层可以建模时空特征。")
&emsp;&emsp;AGC-LSTM层的具体结构如下图所示，其内部的图卷积operator不仅能高效提取空间configuration和时序动态中的判别性特征，而且能探索时间空间领域的相互关系。另外，使用attention机制来增强每个时间步中关键关节点的特征，保证AGC-LSTM能学到更有判别性的特征，比如，肘、手腕和手的特征对于“挥手”的动作很重要，在识别动作时应该被突出。
![](/images/AGC/fig_AGC.png "AGC-LSTM层的结构，与传统LSTM不同，此结构中的图卷积operator使得输入、隐层状态和cell memory都是图结构的数据。")
# 模型结构
## 图卷积神经网络
&emsp;&emsp;GCN是一种通用的高效的学习图结构数据的表示方法，对于基于骨架的动作识别，用*G<sub>t</sub>={V<sub>t</sub>, E<sub>t</sub>}*表示时间t单帧图像上人的骨架的图，*V<sub>t</sub>*是N个关节点的集合，*E<sub>t</sub>*是节点边的集合，节点v<sub>ti</sub>的邻接点的组合表示为N(v<sub>ti</sub>)={v<sub>tj</sub>|d(v<sub>ti</sub>, v<sub>tj</sub>)<=D}，其中d是表示从v<sub>tj</sub>到v<sub>ti</sub>的最短路径长度，给图打标签的函数为*l*: *V<sub>t</sub>*->{1,2,...K}，加你个标签{1,2,...K}赋给每个图的节点v<sub>ti</sub>（属于V<sub>t</sub>），将v<sub>ti</sub>的邻接点的集合N(v<sub>ti</sub>)分成固定数量的K个子集，图卷积的计算方法一般如下所示，其中X(v<sub>tj</sub>)是节点v<sub>tj</sub>的特征，W是权重函数，从K个权重中分配一个由标签*l*(v<sub>tj</sub>)索引的权重，Z<sub>ti</sub>(v<sub>tj</sub>)是对应的子集的数量，能归一化特征表示，Y表示图卷积在节点的输出。
![](/images/AGC/for_gc.png "")
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
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
------未完待续