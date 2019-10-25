---
title: P-CNN Pose-based CNN Features for Action Recognition (ICCV 2015)
date: 2019-10-25 20:09:20
categories: 动作识别
tags:
- 位姿
---
&emsp;&emsp;本文通过对人身体部位的跟踪集成了运动和外观信息，提出了P-CNN (Pose-based Convolutional Neural Network)描述子。在以往的动作识别方法中，基于局部运动描述子的方法在识别移动相机中的粗糙的动作中很成功，如站起来、挥手和跳舞等，基于全局特征的方法，由于缺少结构，不太适合识别微小的变化。
&emsp;&emsp;为了构建P-CNN特征，我们首先计算相邻帧的光流，计算方法借鉴ECCV-2004的论文：High accuracy optical flow estimation based on a theory for warping，速度较快，精度较高，在其他基于光流的CNN方法中都有应用，如双流方法，运动场v<sub>x</sub>和v<sub>y</sub>的值转换到[0, 255]，转换的方法为a\*v<sub>x|y</sub>+b，其中a=16，b=128。低于0的值和高于255的值都被截断，并将光流值转换为光流图，三个通道分别是转换后的v<sub>x</sub>和v<sub>y</sub>和光流幅值。
&emsp;&emsp;给定视频帧和对应的身体关节位置，我们将RGB图片patch和光流patchcrop成右手、左手、上身、整个身体和整张图片，每个patch均resize成224\*224作为CNN的输入，为了表示外观和运动patch，我们使用两个独立的CNN网络，包括5个卷积层和3个全连接层，第二个全连接层的神经元个数是4096，作为视频帧的描述子f<sup>p</sup><sub>t</sub>，对于RGB的patch，我们使用VGG-f网络，在ImageNet ILSVRC-2012上预训练；对于光流patch，我们使用CVPR-2015中Finding action tubes的网络，在UCF-101上进行预训练。
![](/images/PCNN/architecture.png "RCNN结构")
&emsp;&emsp;将所有帧的描述子f<sup>p</sup><sub>t</sub>进行集成得到一个固定长度的视频描述子，集成的方法为求最小和最大。那么静态视频描述子可以表示为
![](/images/PCNN/static.png "")
![](/images/PCNN/minmax.png "")
&emsp;&emsp;为了获取每帧描述子随时间的变化，考虑使用描述子之间的差值，同样计算差值的最大值和最小值作为动态视频描述子。
![](/images/PCNN/delta.png "")
![](/images/PCNN/dynamic.png "")
&emsp;&emsp;最终，将所有部位运动和外观特征归一化并拼接，归一化方法是除以训练集中f<sup>p</sup><sub>t</sub>的平均L2范数。当然在后面的实验中也评估了不同集成方法的效果。
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
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;
&emsp;&emsp;