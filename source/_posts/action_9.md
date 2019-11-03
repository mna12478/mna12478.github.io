---
title: Temporal Segment Networks Towards Good Practices for Deep Action Recognition (ECCV 2016, PAMI 2018)
date: 2019-11-01 09:02:23卷积网络对于静态图片的识别
categories: 动作识别
tags:
- ActivityNet
---
&emsp;&emsp;卷积网络在静态图片的识别中取得了很大的成效，但是在动作识别中，相比于传统方法，优势并没有那么明显，本文探索了基于卷积神经网络的动作识别方法，尤其是在有限的训练样本的情况下，提出了时序分割网络（Temporal Segment Network, TSN)，该方法是基于长时时序结构建模。本文的[代码](https://github.com/yjxiong/temporal-segment-networks)已经公开。
&emsp;&emsp;现有的方法多是在剪辑过的视频上进行试验，但是在实际中我们需要在没有剪辑过的 视频（如THUMOS，ActivityNet）进行试验，每个动作可能只占整个视频的一小部分，主要的背景部分可能会干扰动作识别模型的预测。为了解决这个问题，我们需要考虑定位动作的位置，同时避免背景的影响。其次，训练动作识别模型时会遇到一些问题：训练深度卷积网络通常需要大量的训练样本才能达到最优的表现，但是现有公开的动作识别数据库，如UCF-101和HMDB51在规模和复杂度上都还很局限，模型容易过拟合；提取光流特征来获取短时运动信息成为了一个计算的瓶颈。所以，本文从以下几个方面着手：如何高效学习视频表示，从而提取长时时序结构；如何在实际的未剪辑视频中应用模型进行识别；在给定有限的训练样本的情况下如何有效地学习卷积网络，并将模型应用到大规模的数据中。
&emsp;&emsp;我们提出的TSN网络提供了一个简单并且通用的学习动作视频中动作模型的网络，这个模型是基于，我们观测到连续的视频帧是高度冗余的，稀疏并且全局的时序采样策略更适合并且更高效。TSN框架首先使用稀疏采样方案在较长的视频序列上提取短片段snippet，首先将视频划分为固定数量的分段segment，然后从每个片段中随机采样一个片段。之后，使用分段一致性函数来集成采样的snippet中的信息，这样可以建模整个视频的长时时序结构，计算复杂度独立于视频长度，实际上，我们全面研究了不同段数的影响，并提出了五个汇总函数来总结这些采样片段的预测分数，包括三种基本形式：average pooling, max pooling和加权平均，以及两个高级方案：top-K pooling和自适应attention加权。后面两个的设计是为了在训练时自动化突出有判别性的snippet。
&emsp;&emsp;为了将学习之后的模型应用到未剪辑视频，我们设计了一个分级集成策略，叫做M-TWI (Multi-scale Temporal Window Integration)。我们首先将未剪辑的视频分成固定时间的短时间窗的序列，然后为每个窗口独立地进行动作识别，窗口内的动作识别是通过max-pooling窗口中所有snippet-level的识别分数得到的，最终，通过时序segment网络的集成函数，我们使用top-K pooling法或attention权重来集成这些窗口的预测，从而得到video-level的识别结果。由于隐含选择有间隔的判别性动作的能力，同时能抑制噪声背景影响，本文提出的集成模型能有效识别未剪辑的视频。
&emsp;&emsp;为了解决有限数据库的问题，我们首先提出了一个交叉模型初始化方法，将从RGB模式学习到的representation转移到其他模型，如光流；其次，我们提出了一个在fine-tune场景中进行BN的策略，成为partial BM，只有第一个BN层的均值和方差需要自适应更新来控制domain shift。为了全面地使用视频中的视觉内容，我们研究了TSN的四种类型的输入模式，单帧RGB图像、堆叠RGB差异、堆叠光流场合堆叠变形的光流场。将RGB和RGB差异组合，我们构建了最好的实时动作识别系统，能解决现实生活中的很多问题。
&emsp;&emsp;本文的**贡献点有三个**：端到端的视频表示模型TSN，能获取长时时序信息；分级集成策略，在未剪辑的视频中进行动作识别；一系列学习和应用深度动作识别模型的好的实践。在之前论文的基础上，本文的扩展点有：TSN中新的集成函数，能有效突出重要的snippet同时减少背景噪声；通过设计分级集成策略，将原始的动作识别方法扩展到未剪辑视频的分类；对TSN网络进行进一步的探索研究，增加了两个数据库的实验；基于TSN网络，提出了ActivityNet挑战赛2016的解决方案，在未剪辑视频分类的24支队伍中排名第一。
# TSN网络
&emsp;&emsp;以前学习长时时序结构的网络，由于计算量和GPU内存的限制，通常只能处理固定分64~120帧的序列，不太可能学习整个视频。而本文提出的方法没有序列长度的限制，是一个video-level的端到端的结构，能建模整个视频。
## 基于segment采样策略的来源
&emsp;&emsp;以双流网络和C3D为例的网络能操作有限的时序长度，如单帧或者帧的堆叠（如16帧），这些结构缺少将长时时序信息整合到动作模型的学习中的能力。为了建模长时时序结构，有很多方法，如堆叠更多的连续帧（如64帧）或者以固定比例采样更多的帧（如1FPS），尽管类似这种密集或局部采样的方法能帮助缓解双利和C3D这种短时卷积网络的问题，但依然存在计算量和建模方面的问题，从计算的角度说，这些方法会极大地增加卷积网络训练的计算量；从建模的角度说，这些方法的时序覆盖依旧是局部和有限的，如采样的64帧只占了一段10s视频（约300帧）的一小部分。
&emsp;&emsp;另外，尽管这些方法密集记录了视频帧，但是内容的改变却是相对缓慢的，于是我们提出了segment based sampling方法，这是一种稀疏和全局的采样方法，首先，只有一小部分稀疏采样的snippet会用来建模一个动作中的时序结构，通常一次训练迭代的采样帧的数量固定为一个预定义的值，这个值与视频长度无关，这能保证计算消耗是一个常数；其次，这个方法确保了这些采样的snippet会沿着时序维度均匀分布，因此无论动作视频持续多久，我们采样的snippet会一直大致覆盖整个视频的视觉内容。
## 网络结构
&emsp;&emsp;TSN网络的输入是一系列从整个视频中采样的短的snippet，为了是这些采样的snippet能表示整个视频的内容同时保持合理的计算量，segment based sampling方法会首先将视频分成几个等时长的segment，然后从每个segment中采样一个snippet，序列中每个snippet会生成snippet-level的动作类别预测，然后使用一个consensus函数将这些snippet-level的预测集成，并通过softmax得到video-level的分数，这个分数会比原始的snippet-level分数更可靠，因为获取了整个视频的长时信息，在训练阶段，优化目标是video-level的预测。在这里，consensus函数是很重要的，因为它应该具备高度建模能力，同时可微分或者有子梯度，高度建模能力是指将snippet-level的预测集成为video-level的能力。整个网络结构如下图所示。
![](/images/TSN/TSN.png "TSN网络结构")
&emsp;&emsp;整个集成过程可以用如下公式表示，*F*(T<sub>k</sub>; W)表示对短的snippet，即T<sub>k</sub>进行预测的参数为W的卷积网络，会生成类别分数，G为consensus函数，预测函数H为整段视频预测类别，这里用的是softmax函数。
![](/images/TSN/fun_TSN.png "")
&emsp;&emsp;训练期间，结合标准的categorical cross-entropy，最终的损失函数如下所示，C是动作类别的数量，g<sub>j</sub>是G的第j个维度，G=*G*(F(T<sub>1</sub>;W), F(T<sub>2</sub>;W),...,F(T<sub>K</sub>;W))
![](/images/TSN/func_loss.png "")
&emsp;&emsp;损失函数关于W的梯度如下所示，K是TSN网络中segment的数量，以下公式展现了参数更新是通过使用来自所有snippet-level的预测的分段consensus函数G，这样TSN网络就能通过整个视频学习参数，而不是短的snippet。另外，通过为所有视频固定K，我们将稀疏的时序采样聚合来选择少量的snippet，另外，相比于使用密集采样的方法，这能极大地减少对视频帧记性evaluate的卷积网络的计算量。
![](/images/TSN/func_gradient.png "")
## 集成函数（consensus函数）
&emsp;&emsp;这里我们提出了五种consensus策略：max-pooling，average-pooling，top-K pooling，加权平均和attention加权。
### Max-pooling
&emsp;&emsp;这种方法是对采样的snippet每个种类的预测分数进行max-pooling，即g<sub>i</sub>=max<sub>k=1,2,...K</sub>f<sub>i</sub><sup>K</sup>，其中f<sub>i</sub><sup>K</sup>是F<sup>k</sup>=*F*(T<sub>k</sub>; W)的第i个元素，g<sub>i</sub>关于f<sub>i</sub><sup>K</sup>的梯度为：
![](/images/TSN/func_max.png "")
&emsp;&emsp;使用max-pooling的主要原理是为每个动作类别寻找一个最有判别性的snippet，并且使用这种最有力的激活作为video-level的响应，它强调单个snippet，完全忽略其他snippet的响应。所以，这种集成函数鼓励TSN从最有判别性的snippet学习，但是缺少将多个snippet联合建模成video-level动作识别的能力。
### Average-pooling
&emsp;&emsp;对所有的K个f<sub>i</sub><sup>K</sup>求平均，梯度为1/K。这种方法对所有snippet的响应进行平均，使用均值作为video-level的预测值，这种方法联合建模了多个snippet，并且从整个视频中获取视觉信息；另外，对于一些背景复杂的有噪声的视频，一些snippet可能是与动作无关的，平均化这些背景snippet可能会影响最终的识别性能。
### Top-K pooling
&emsp;&emsp;为了均衡max和average pooling，提出了op-K pooling，我们首先为每个动作种类选择K个最有判别性的snippet，然后平均化这些选择的snippet，公式和梯度可以如下表示，这个集成函数能自适应决定有判别性的snippet的子集，具备max和average两种方法的优势，能联合建模多个相关的snippet同时避免背景snippet的影响。
![](/images/TSN/func_top.png "")
![](/images/TSN/func_top-1.png "")
### 线性加权
&emsp;&emsp;这种方法是对每个动作类别的预测采用element-wise的加权线性组合，公式和梯度可以如下表示，*w*<sub>k</sub>是第k个snippet的权重，是自适应更新，这种集成方法的假设是动作可以分解成几个阶段，这些不同的阶段在识别中起到不同的作用，这个集成函数希望能学到一个动作类别不同阶段的权重，可以作为snippet选择的soft版本。
![](/images/TSN/func_weight.png "")
![](/images/TSN/func_weight-1.png "")
### Attention加权
&emsp;&emsp;很明显线性加权是与数据独立的，缺少考虑视频之间差异的能力，因此又提出一种自适应加权的方法，attention加权，这种集成函数希望函数能根据视频内容为每个snippet自动分配一个重要性权重，公式表示和权重如下所示，*A*(T<sub>K</sub>)是snippet T<sub>K</sub>的attention权重，是根据视频内容自适应计算的，对最终的性能很重要。在当前的方法中，我们首先用相同的卷积网咯从每个snippet中提取视觉特征R = *R*(T<sub>K</sub>)，然后生成attention权重*A*(T<sub>K</sub>)。其中w<sup>attn</sup>是attention权重计算函数的参数，可以和网络权重W一起学习，*R*(T<sub>k</sub>)是第k个snippet的视觉特征，目前是最后隐层的激活值，那么*A*(T<sub>k</sub>)关于w<sup>attn</sup>的梯度如下所示，
![](/images/TSN/func_attn.png "")
![](/images/TSN/func_attn-1.png "")
![](/images/TSN/func_attn-2.png "")
![](/images/TSN/func_attn-3.png "")
&emsp;&emsp;有了梯度计算公式后，我们可以与卷积网络参数W一起，通过反向传播学习attention建模参数w<sup>attn</sup>，而且，反向传播的公式可以按如下所示的方式定义。所以，这种方法的优势有两点：增强了基于视频内容自动学习每个snippet重要性的能力；由于attention模型是基于卷积网络的表示R，利用额外的反向传播信息指导ConvNet参数W的学习过程，可以加快训练的收敛速度。
![](/images/TSN/func_attn-3.png "")
## 实际的TSN网络
&emsp;&emsp;实际训练中为了使TSN网络达到最优性能，需要考虑一些训练技巧。
### TSN结构
为了体现方法的通用性，用多种网络结构初始化TSN，为了进行多方面的验证，这里采用了Inception v2，因为这个结构平衡了精度和效率，在ActivityNet挑战赛中，我们探索了更强大的结构，包括Inception v3和ResNet152。
### TSN输入
&emsp;&emsp;相比于图片，视频额外携带的时序维度携带了另一种用于动作理解的重要线索，也就是运动，双流网络中使用密集光流场作为输入证明是有效果的，在这篇文章中，我们从两个方面扩展了这个方法：精度和速度，如下图所示，除了原始的RGB和光流外，我们也探索了另外两种模式：变形的光流和RGB差异。
#### 变形的光流
&emsp;&emsp;变形的光流对相机的运动具有鲁棒性，能帮助集中于人的动作，我们希望这有助于提高运动感知的准确性，从而提高动作识别性能。
#### RGB差异
&emsp;&emsp;双流网络虽然精度高，但影响其应用的问题是光流提取时所消耗的时间，为了解决这个问题，我们构建了一个没有光流的运动表示方法，我们重新回顾了最简单的视觉运动感知线索：堆叠的连续帧之间的RGB像素差异。回顾之前的研究中关于密集光流的工作，像素强度与时间的偏导数在计算光流中起着关键作用，所以假设光流在表示运动特征方面的能力可以通过简单的RGB来学习是合理的。
### TSN训练
&emsp;&emsp;现有的人类标注的动作识别数据库规模较小，在实际中使用这些数据库训练卷积网络很可能会过拟合，为了解决这个问题，我们设计了几种优化训练的策略。
#### 交叉模式初始化
&emsp;&emsp;当目标数据库没有足够的训练样本时，在大规模图像识别数据库如ImageNet中预训练网络参数是有效的策略，因为空间网络以图片作为输入，所以很自然想到用在ImageNet上预训练的模型作为初始化，对于其他输入模式，如光流和RGB差异，我们提出交叉模式初始化策略，首先通过线性变换的方法将光流离散化到[0, 255]，然后在第一层，沿RGB通道平均化预训练的RGB模型的权重，并将平均值复制为时序网络的输入，最后时序网络余下层的权重直接从预训练RGB网络中复制。
#### 正则化
&emsp;&emsp;BN层加速了训练时网络的收敛，同时，由于偏向于目标数据库有限规模训练样本的均值和方差，增加了迁移学习阶段过拟合的风险，因此，经过预训练模型初始化后，固定除了第一个BN层之外的所有BN层的均值和方差。由于光流的贡献不同于RGB图像，所以第一个卷积层的激活值会有不同的分布，并且我们需要重新评估对应的均值和方差，我们把这种策略称为partial BN，同时，我们在全局池化之后添加了额外的dropout层，dropout比例较高0.8，来进一步降低过拟合的影响。
#### 数据增广
&emsp;&emsp;原始的双流网络使用随机crop和水平flip来扩增训练样本，这里我们使用两种新的数据扩增方法：边角crop和scale jittering。边角crop只从边角或者中心来选取图片区域，避免过多地集中于中心区域；在多尺度crop中，我们将ImageNet中使用的scale jittering技术用于动作识别，提出一个有效的scale jittering，将输入图片固定为256\*340，crop的区域的宽度和高度，从{256; 224; 192; 168}中随机选择，最后将crop的区域resize成224\*224，用于网络的训练，实际上，这种方法不仅包含scale jittering，还包括纵横比jittering。
# 用TSN进行动作识别
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
-----------未完待续