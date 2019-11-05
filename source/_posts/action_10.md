---
title: Actionvlad Learning spatio-temporal aggregation for action classification (CVPR 2017)
date: 2019-11-05 22:09:10
tags:
- 双流网络
- 时空特征
---
&emsp;&emsp;本文介绍了一种新的用于动作识别的视频表示方法，通过将双流网络与科学系的时空特征组合，在视频的整个时空范围内聚合局部卷积特征，得到的结构是端到端可训练的，为整个视频分类。我们探索了沿空间和时间pooling的不同策略，以及几种组合不同stream信号的策略，我们发现联合pool空间和时间很重要，但是外观和运动stream最好汇总到各自单独的表示中。
&emsp;&emsp;3D时空卷积方法能学习到复杂的时空依赖，但是在识别性能方面很难扩展；而双流网络将视频分成运动流和外观流，由于能轻松使用新的深度网络，所以性能逐渐超过时空卷积，但是双流卷积忽视了视频的长时时序结构，在测试阶段分别对采样的一帧或堆叠的几帧进行识别，然后取平均来得到最终的结果，但这种时序的平均不一定能建模复杂的时空结构，以下图篮球投篮为例，给定视频的少数几帧，很有可能与其他的动作，如跑、运球、挑、扔等混淆，使用late fusion或者平均需要视频帧都属于相同的子动作来分配给不同的类别，所以不是一个最优方案，我们需要的是一个全局特征描述子，能对整个视频进行集成，包括场景的外观和人的动作，不需要每帧都分配一个单独的动作类别。
![](/images/VLAD/fig_basket.png "")
&emsp;&emsp;所以，本文提出了ActionVLAD模型，核心是NetVLAD集成层的时空扩展，我们将新的层成为ActionVLAD。在扩展NetVLAD时带来了两个挑战：将不同时间frame-level特征整合为video-level表示的最好的方法是什么；如何组合来自不同stream的信号。
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
