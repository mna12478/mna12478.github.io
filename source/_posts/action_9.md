---
title: Temporal Segment Networks Towards Good Practices for Deep Action Recognition (ECCV 2016, PAMI 2018)
date: 2019-11-01 09:02:23卷积网络对于静态图片的识别
categories: 动作识别
tags:
- ActivityNet
---
&emsp;&emsp;卷积网络在静态图片的识别中取得了很大的成效，但是在动作识别中，相比于传统方法，优势并没有那么明显，本文探索了基于卷积神经网络的动作识别方法，尤其是在有限的训练样本的情况下，提出了时序分割网络（Temporal Segment Network, TSN)，该方法是基于长时时序结构建模。本文的[代码](https://github.com/yjxiong/temporal-segment-networks)已经公开。
&emsp;&emsp;现有的方法多是在剪辑过的视频上进行试验，但是在实际中我们需要在没有剪辑过的 视频（如THUMOS，ActivityNet）进行试验，每个动作可能只占整个视频的一小部分，主要的背景部分可能会干扰动作识别模型的预测。为了解决这个问题，我们需要考虑定位动作的位置，同时避免背景的影响。其次，训练动作识别模型时会遇到一些问题：训练深度卷积网络通常需要大量的训练样本才能达到最优的表现，但是现有公开的动作识别数据库，如UCF-101和HMDB51在规模和复杂度上都还很局限，模型容易过拟合；提取光流特征来获取短时运动信息成为了一个计算的瓶颈。所以，本文从以下几个方面着手：如何高效学习视频表示，从而提取长时时序结构；如何在实际的未剪辑视频中应用模型进行识别；在给定有限的训练样本的情况下如何有效地学习卷积网络，并将模型应用到大规模的数据中。
&emsp;&emsp;我们提出的TSN网络提供了一个简单并且通用的学习动作视频中动作模型的网络，这个模型是基于，我们观测到连续的视频帧是高度冗余的，稀疏并且全局的时序采样策略更适合并且更高效。TSN框架首先使用稀疏采样方案在较长的视频序列上提取短片段snippet，首先将视频划分为固定数量的分段segment，然后从每个片段中随机采样一个片段。之后，使用分段一致性函数来集成采样的snippet中的信息，这样可以建模整个视频的长时时序结构，计算复杂度独立于视频长度，实际上，我们全面研究了不同段数的影响，并提出了五个汇总函数来总结这些采样片段的预测分数，包括三种基本形式：average pooling, max pooling和加权平均，以及两个高级方案：top-K pooling和自适应attention加权。后面两个的设计是为了在训练时自动化高亮有判别性的snippet。
&emsp;&emsp;为了将学习之后的模型应用到未剪辑视频，我们设计了一个分级集成策略，叫做M-TWI (Multi-scale Temporal Window Integration)。我们首先将未剪辑的视频分成固定时间的短时间窗的序列，然后为每个窗口独立地进行动作识别，窗口内的动作识别是通过max-pooling窗口中所有snippet-level的识别分数得到的，最终，通过时序segment网络的集成函数，我们使用top-K pooling法或attention权重来集成这些窗口的预测，从而得到video-level的识别结果。由于隐含选择有间隔的判别性动作的能力，同时能抑制噪声背景影响，本文提出的集成模型能有效识别未剪辑的视频。
&emsp;&emsp;为了解决有限数据库的问题，
-----------未完待续
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
