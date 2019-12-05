---
title: Action recognition with spatial-temporal discriminative filter banks (ICCV 2019)
date: 2019-12-05 20:18:26
categories: 
- 动作识别
tags:
---
&emsp;&emsp;之前的动作识别算法都是将识别当成分类问题，更多地关注怎么由网络获取运动，却忽略了是什么使得动作变得独特。本文更多地关注动作识别问题本身，展示了动作识别是如何需要对更精细的细节有更多的敏感度，不同于以前的动作识别算法，本文的方法专为细粒度的动作分类而设计。以前的方法多事在顶层使用一个全局平均池化和线性分类层，而本文能提取和使用细粒度信息来分类，这是全局平均池化无法做到的，本文提出了三种分类的分支：第一个分支是最常用的全局平均池化；第二个和第三个共享一组卷积、空间上采样和max-pooling层来帮助提取细粒度信息，但分类器不同。这三个分支是以端到端的方式联合训练，这个新的设计能兼容大多数动作识别算法，能用于基于2D和3D CNN的网络。
![](/images/bank/fig_samp.png "")
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