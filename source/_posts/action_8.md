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
&emsp;&emsp;之前的双流网络有两个不足：首先，由于只在分类层进行融合，所以无法学习时空特征之间pixel-wise的对应关系，其次，时间规模有限，空间卷积只在单帧上操作，时序卷积只在堆叠的L (L=10)个时序相邻的光流帧上，双流网络通过在规则的空间采样上使用pooling，在一定程度上解决了后一个问题，但并没有对行为的时间演变进行建模。
## 空间融合
&emsp;&emsp;这部分的目的是（在一个特定的卷积层）融合两个网络，使得相同pixel位置的通道响应能对应，以区分刷牙和梳头为例，如果人的手在周期性地在某个空间位置移动，那么时间网络就能识别到这个动作，空间网络能识别到位置（牙齿或者头发），两个网络的融合就能区分动作。
&emsp;&emsp;当两个网络在要融合的层有相同的分辨率时，通过一个网络与另一个网络的重叠（叠加）层，很容易获得空间响应，但同样存在一个问题，这个网络的通道对应另一个网络的哪个通道。假定空间网络中不同的通道负责不同的脸部区域（如嘴、头发等），时间网络中一个通道负责这个类型的周期性的运动场，那么经过通道的堆叠后，后面层的滤波器必须学习这些合适的通道间的对应关系，，以便更好地区分这些动作。
&emsp;&emsp;具体来说，假定融合函数f能融合两个在时刻t的feature map，x<sup>a</sup><sub>t</sub>, x<sup>b</sup><sub>t</sub>->y<sub>t</sub>，x<sup>a</sup><sub>t</sub>的shape=H\*W\*D，x<sup>b</sup><sub>t</sub>的shape=H'\*W'\*D'，y<sub>t</sub>的shape=H''\*W''\*D''，在包含卷积、全连接、池化和非线性层的卷积网络中，f可以用于网络中不同的层来实现Early-Fusion, Late-Fusion和多层融合，可以使用多种融合函数f，为了简便，假定H=H'=H'', W=W'=W'',D=D'，并去掉下标t。
### Sum-Fusion
&emsp;&emsp;y<sup>sum</sup>=f<sup>sum</sup>(x<sup>a</sup>, x<sup>b</sup>)计算了两个相同空间位置i, j, d的feature map的和:
![](/images/Fusion/sum-fusion.png "")
### Max-Fusion
&emsp;&emsp;y<sup>max</sup>=f<sup>max</sup>(x<sup>a</sup>, x<sup>b</sup>)简单地取两个feature map的最大值，定义方法与Sum-Fusion类似。
### Concatenation-Fusion
&emsp;&emsp;y<sup>cat</sup>=f<sup>cat</sup>(x<sup>a</sup>, x<sup>b</sup>)沿通道d，在相同的空间位置i,j堆叠两个feature map。
![](/images/Fusion/concatenation.png "")
### Conv-Fusion
&emsp;&emsp;y<sup>conv</sup>=f<sup>conv</sup>(x<sup>a</sup>, x<sup>b</sup>)首先按Concatenation-Fusion的方式堆叠两个feature map，然后对堆叠后的数据进行卷积，卷积核为f (shape=1\*1\*2D\*D)。
![](/images/Fusion/conv-fusion.png "")
&emsp;&emsp;输出通道数量是D，这里的卷积核f是用来减少维度，并且能以权重的方式建模相同空间位置的两个feature map，当作为可训练的卷积核时，f能学到两个feature map之间的对应关系，以此来减少loss。
### Bilinear-Fusion
&emsp;&emsp;y<sup>bil</sup>=f<sup>bil</sup>(x<sup>a</sup>, x<sup>b</sup>)计算每个特征图在每个像素位置的外积，
![](/images/Fusion/bilinear.png "")
&emsp;&emsp;由此得到的特征获取了对应空间位置乘法的交互，这个特征主要不足是维度较高（y<sup>bil</sup>的shape=D<sup>2</sup>），为了使其在实际中可用，通常使用RELU5，移除全连接层，使用L2正则化，基于线性SVM进行分类。Bilinear-Fusion的优势是网络的每个通道都与其他网络的每个通道相结合（作为积），缺点是空间信息在这个点被边缘化了。
&emsp;&emsp;融合层的注入会对双流网络的参数和层产生重要影响，尤其是只保留了被融合的层，而其他层被截断，如下图左侧所示，
![](/images/Fusion/example.png "")
&emsp;&emsp;所以，可以在任意两个有相同空间维度的feature map上进行融合，即H=H', W=W'。此外，也可以在两个层进行融合，如上图右侧所示，
## 空间融合
&emsp;&emsp;空间融合，即沿时间t组合特征图x<sub>t</sub>，得到结果y<sub>t</sub>，一种处理时序输入的方法是沿时间平均化网络的预测，这种情况只对2D(x, y)进行pool，如下图a中所示。考虑一个时序pooling层的输入为x， shape=H\*W\*T\*D，通过沿时间t=1...T来堆叠空间的特征图生成。
![](/images/Fusion/temporal.png "")
### 3D-Pooling
&emsp;&emsp;使用尺寸为W'\*H'\*T'的3D pooling块对堆叠的数据进行max-pooling，这是将2D pooling延伸到时间域的最直接的扩展方法，如上图b所示，例如要pool三个时序样本，那么可以沿三个堆叠的对应通道进行3\*3\*3的max-pooling，没有不同通道间的pooling。
### 3D Conv+Pooling
&emsp;&emsp;首先对四通道的输入x进行卷积，滤波器尺寸为W''\*H''\*D\*D'，卷积之后是上述的3D poolinig层，此方法如上图c所示，滤波器能以权重的形式，使用W''\*H''\*D的卷积核建模一个局部时空区域的特征的组合，这个区域通常是3\*3\*3（空间\*时间)。
## 提出的网络结构
\----------------未完待续
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

