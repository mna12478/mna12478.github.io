---
title: Hidden Two-Stream Convolutional Networks for Action Recognition (ACCV 2018)
date: 2019-11-10 21:59:40
categories: 
- 动作识别
- Fusion
- 双流
tags:
- 双流
- 无监督
---
&emsp;&emsp;现有的方法依赖光流特征，而传统的光流计算需要为CNN预先计算运动信息，这种两阶段方法计算量大，存储空间需求大，不是可端到端训练的。本文提出一种新的CNN结构用于提取运动信息，我们称之为hidden双流CNN，因为只需要原始像素作为输入，在不需要计算光流的情况下直接预测动作类别，速度10倍快与原始的双流网络，在UCF-101、HMDB51、THUMOS14和ActivityNet v1.2上都是最好的实时动作识别方法。
&emsp;&emsp;经过几年的发展，动作识别方法已经从原来的手动提取特征到现在的学习CNN特征；从encoding外观信息到encoding运动信息；从学习局部特征到学习全局特征。最初的CNN用于动作识别的效果并不是太好，甚至还不如iDT，可能是因为这时候的CNN比较难获取视频帧之间的运动信息，后来的双流网络通过使用传统光流预先计算光流特征解决了这个问题，时序stream极大地提升了CNN的精度，在几个数据库上都超过了iDT。但是现有的CNN网络依旧很难直接从视频中提取运动信息，而先计算光流，再把光流映射为动作标签是一个次优的方法：与CNN步骤相比，光流的预计算耗时耗空间；传统的光流评估完全独立于最终的任务，有研究（Video Enhancement with
Task-Oriented Flow, arxiv 2017）表明，固定的光流计算方法不如任务导向的光流计算方法的效果好。所以是次优的。为了解决这个问题，出现了运动向量（20倍快于传统双流网络，但编码之后的运动向量缺少好的结构，并且包含噪声和不正确的运动模式，所以导致精度却下降很多，请见论文：Real-time Action Recognition with Enhanced Motion Vector CNNs, CVPR 2016）、RGB图像差异或者RNN、3DCNN等结构，但是大多数不如光流特征，在动作识别的任务中有效。
# Hidden双流网络
## 无监督光流学习
&emsp;&emsp;我们将光流计算方法看成图像重建问题。给定一对视频帧，我们希望生成光流，允许我们从一帧重建另一帧。例如，给定输入*I<sub>1</sub>*he *I<sub>2</sub>*，CNN网络生成了光流场V，然后使用光球场V和*I<sub>2</sub>*，通过backward warping，我们能得到重建帧*I<sub>1</sub><sup>'</sup>*，且*I<sub>1</sub><sup>'</sup>=T[I<sub>2</sub>,V]*
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
&emsp;&emsp;
--未完待续