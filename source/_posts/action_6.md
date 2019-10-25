---
title: Long-term Recurrent Convolutional Networks for Visual Recognition and Description (CVPR 2015)
date: 2019-10-24 16:47:23
tags:
- 动作识别
- 图片取标题
- 视频描述
- RNN
- LRCNs
---
&emsp;&emsp;本文提出了LRCNs (Long-term Recurrent Convolutional Networks )模型，以端到端训练的方式结合了卷积和RNN，接受可变长的输入和输出，可用于行为识别，图片标题生成和视频描述。
&emsp;&emsp;相比于RNN，LSTM因为加入记忆单元，所以可以学到什么时候忘记以前的隐层状态，给定新的信息后什么时候更新现在的隐层状态。
![](/images/LRCN/LSTM.png "RNN和LSTM结构")
&emsp;&emsp;输入门和遗忘门用于选择性遗忘以前的记忆或者考虑现在的输入，输出门决定记忆单元有多少比例传送给隐层状态。LSTM已经用于语音识别（双向LSTM），即使没有语言模型或语音词典，基于LSTM的模型也可以进行文本翻译，使用encoder-decoder模型将英语翻译为法语，sequence-to-sequence结构。
# LRCN网络结构
&emsp;&emsp;将visual input，可能是一张单独的图片，也可能是视频中的一帧，通过特征转换，得到一个固定长度的特征向量，然后输入到sequence model中。一般sequence model是将输入和前一时刻的隐层转台映射为输出，并更新当前时刻的隐层状态。那么在测试阶段，也应该按序列进行，也就是下图中的Sequence Learning部分，计算方法为h<sub>1</sub>=f<sub>W</sub>(x<sub>1</sub>, h<sub>0</sub>)=f<sub>W</sub>(x<sub>1</sub>, 0)，h<sub>2</sub>=f<sub>W</sub>(x<sub>2</sub>, h<sub>1</sub>)，一直计算到h<sub>T</sub>.
![](/images/LRCN/LRCN.png "LRCN结构")
&emsp;&emsp;本文考虑解决三种视觉方面的问题：行为识别、图片描述和视频表示：
&emsp;&emsp;1、行为识别，序列型输入，固定长度的输出，<x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T</sub>>&emsp;->&emsp;y，输入为任意长度T的视频，目标是从固定的词汇库中预测单个标签，如running, jumping。解决方案是使用late fusion的方法，将每个timestep的预测值合并为单个预测值。
&emsp;&emsp;2、图片描述，固定长度的输入，序列型输出，x&emsp;->&emsp;<y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>T</sub>>，输入为非随时间变化的图片，输出的标签空间更大，更丰富，由任意长度的句子组成。解决方案是在在所有的timestep都复制输入。
&emsp;&emsp;3、视频表示，序列型输入和输出，<x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T</sub>>&emsp;->&emsp;<y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>T'</sub>>，输入和输出都是随时间变化，一般输入和输出的时间步不同。解决方案是基于encoder-decoder结构，第一个序列模型encoder用于将输入序列映射为固定长度的向量；第二个序列模型decoder用于将向量展开成任意长度的序列输出。
![](/images/LRCN/specific.png "三种问题的解决方案")
# 行为识别
&emsp;&emsp;一个clip包含16帧，尝试了LRCN的两个变种：LSTM放在CNN第一个全连接层的后面，即LRCN-fc<sub>6</sub>，另一个将LSTM放在CNN第二个全连接层后面，即LRCN-fc<sub>7</sub>，LRCN在每个timestep预测一个视频的分类，通过取平均得到最终的预测分类。测试时，以stride=8提取16帧的视频clip，并对所有clip取平均。
&emsp;&emsp;另外，也考虑了RGB的输入和光流的输入，将光流计算出来并转换成光流图，将x和y分量围绕128居中并乘以标量，使得光流值落在0到255之间，光流图的第3个通道是通过计算光流幅值得到。CNN模型在ILSVRC-2012上进行预训练，在LSTM模型中，整个视频的分类是由所有视频帧分数取平均得到。
&emsp;&emsp;评估的数据库是UCF-101，LSTM放置位置不同，输入不同，输入在计算最终结果时所占的比例不同。
![](/images/LRCN/activity.png "行为识别结果对比")
# 图片描述
&emsp;&emsp;图片描述只需要一个CNN模型，图片特征和之前的描述单词多作为序列模型的输入，考虑到可能会堆叠LSTM，对于时刻t，输入到最底层的LSTM的是经过embed的前一时刻的ground truth word，对于句子生成，输入变成以前的timestep中模型预测分布中的一个样本，第二个LSTM融合了最底层的LSTM的输出和图片表示，在时刻t生成一个视觉和语言输入的联合表示，后面的LSTM将其下面的LATM的输出进行转换，第四个LSTM的输出作为softmax的输入，生成一个单词的分布。
&emsp;&emsp;在检索和生成任务中评估模型，使用的数据库是Flickr30k和COCO2014，这两个数据库每张图片都有5个句子注释。检索的结果评估是使用第一个检索到的gt图像或标题的median rank，Medr和Recall@K，在前K个结果中检索到正确的标题或图片的数量。在结果中，OxfordNet模型再检索任务中稍好一点，但是它使用了较好的卷积网络。（这里就不放结果了）
# 视频描述
&emsp;&emsp;由于视频描述数据有限，所以使用传统的行为和视频识别方法处理输入，使用LSTM生成句子，有以下几种结构，对于每种结构，我们假定已经基于CRF，有了视频中出现的 物体和动作的预测，在每个timestep将视频看成一个整体。
![](/images/LRCN/video_description.png "视频描述几种结构")
&emsp;&emsp;a、CRF max+基于LSTM的encoder-decoder，首先使用CRF的最大后验概率MAP来识别视频的语义表示，如<person, cut, cutting, board>等，拼接成一个输入的句子 (person cut
cutting board) ，并使用基于词组的统计机器学习翻译（statistical machine translation (SMT)）将其翻译成自然的句子 (a person cuts on the board)，将SMT替换成LSTM。encoder用于将one-hot向量表示的输入句子编码，那么encoder最终的隐层单元一定能记住所有的必要信息，decoder用于将隐层表示解码，每个timestep解码一个单词，encoder和decoder使用相同的LSTM。
&emsp;&emsp;b、CRF max+基于LSTM的decoder，语义表示能编码成一个单个的固定长度的向量，我们在每个时间步骤向LSTM提供完整的视觉输入表示，类似于在图片描述中将整个图片输入到LSTM中。
&emsp;&emsp;c、CRF prob+基于LSTM的decoder，相比于基于词组的SMT，使用LSTM锦绣柠机器翻译的好处是，它可以自然地在训练和测试期间合并概率向量，这使LSTM可以学习视觉生成中的不确定性，而不必依赖MAP估计。结构与b相同，但是将最大预测变成概率分布。
&emsp;&emsp;评估的数据库是TACoS中级数据库，对比的方法是使用CRF max的方法。

