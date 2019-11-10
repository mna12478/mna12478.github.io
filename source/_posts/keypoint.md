---
title: 关键点检测
date: 2018-05-05 10:57:28
categories: 
- 深度学习
- Keypoint
tags:
- 关键点
- 位姿估计
- 动作识别
---
&emsp;&emsp;注：本文介绍的RMPE方法在MPII和MSCOCO这两个数据库上的效果很好，AP超过了16年和17年coco比赛的第一名CMU和旷视，在实际中自己从网上找的图片进行测试，对一些比较简单的，比如人面对镜头整成站立或一些小的形变是比较鲁棒的，但当人存在较大弯曲（如杂技演员），就会发生误检，但实际生活中这种情况一般不多，所以基本上能应用在日常生活的一些图片中。
&emsp;&emsp;关键点检测(Keypoint Detection)，也可以叫位姿估计(Pose Estimation)，是找到人体中包括头，手，肘等关节的位置，然后依次连接得到一个“火柴人”。[COCO中的关键点](http://cocodataset.org/)是包括nose, eyes, ears, shoulders, elbows, wrists, hips, knees, & ankles在内的17个关节，而另一个比赛[AI Challenger](https://challenger.ai/competition/keypoint)中则是14个关节。
![](/images/keypoint/keypoint.png "COCO 2017 Keypoint Detection")
![](/images/keypoint/AI_challenger.png "AI Challenger人体骨骼关键点检测")

<font size=5>一、单人关键点检测</font>
&emsp;&emsp;首先是单人的关键点检测，由于图像中只有人，且只有一个人，所以不需要边界框，这里介绍的单人关键点检测是Stacked Hourglass Networks for Human Pose Estimation，首先将输入经过一个7x7的，stride=2卷积，然后经过一个残差块，再经过一个maxpooling层，将输入的256x256分辨率减小为64x64（主要是为了减少memory），然后通过不断堆叠沙漏结构的网络来实现得到关键点的热力图，如下图所示。
![](/images/keypoint/single_person.png "单人关键点检测网络")
&emsp;&emsp;首先介绍单个沙漏网络，结构如下，其中每一个方块都是一个残差块，整个网络是一个encoder-decoder的结构，encoder中的feature map一方面通过Maxpooling+残差块操作，做前向传播，另一方面经残差块，与decoder上采样之后得到的feature map进行加和操作（对位操作），这里的上采样方法采用的是最近邻采样。
![](/images/keypoint/hourglass.png "沙漏结构")
![](/images/keypoint/res_block.png "残差块结构")
&emsp;&emsp;在得到单个沙漏之后，将沙漏结构堆叠在一起，堆叠方式如下所示，白色的方块均为1x1的卷积层，蓝色的是关键点的预测热力图，热力图之后的1x1卷积是为了将热力图的通道数变换至与输入通道相同，最后将输入，沙漏结构的输出，与热力图变换后的feature map三个加在一起，作为下一个沙漏结构的输入。
![](/images/keypoint/stack_method.png "沙漏结构的堆叠")

<font size=5>二、多人关键点检测</font>
&emsp;&emsp;以上是单人关键点检测的方法，对于多人的关键点检测，就有开头说的那两种框架，这里介绍的是自上而下的方法 RMPE: Regional Multi-Person Pose Estimation。在介绍RMPE之前，先看一下以前的人体检测+单人关键点检测存在的问题,下图是一个关键点检测的示例，从图中可以看出两个问题：定位错误问题和冗余问题。
&emsp;&emsp;首先是定位错误问题，实际上，单人关键点检测对人体检测的边界框的误差很敏感，即使在人体检测的IOU>0.5的情况下也有可能会出错，如下图中，红色的是gt，黄色的是人体检测的边界框，黄色框和红色框的IOU>0.5，从SPPE(Single-person pose estimator)的结果上来看，gt对应的关键点检测的热力图是有激活值的，而黄色的边界框对应的关键点检测的热力图是没有激活值，也就是说没有检测到关键点的。
![](/images/keypoint/problem.png "单人关键点检测对人体检测的边界框很敏感")
&emsp;&emsp;第二个问题是人体检测的冗余问题，同一张图片，可能得到多个人体检测的边界框，就会得到多个人体的位姿，如下图所示。
![](/images/keypoint/redundant.png "人体位姿的冗余问题")
&emsp;&emsp;首先，为了解决第一个问题，加入了STN网络，在得到人体检测的边界框后，使用STN网络提取ROI区域，使边界框更加精确，经STN处理后，输入到单人关键点检测的网络中，将此网络的输出输入到STN的反变换网络SDTN中，之后为了解决冗余的问题，引入了非极大值抑制nms，最后得到最终的多人关键点检测结果。同时，加入一个不包含SDTN网络的并行的单人关键点检测网络作为辅佐，这个部分的网络权重固定，损失是通过将此网络的输出直接与center-located(人在图像的中心位置)的gt pose进行比较来得到的，如果主干网络检测到的位姿不是center-located，说明STN检测到的人不是在中心位置，那么这个分支会返回比较大的loss，使STN网络提取更精确的区域，所以可以看成一个正则化，能帮助避免局部最小值（所谓的局部最小点，就是STN网络没有将位姿转换成提取的人体边界框的中心），只是用来提高STN的对ROI区域提取的准确度，在测试的过程中就不要这个并行的网络了。
![](/images/keypoint/RMPE.png "RMPE网络结构")
![](/images/keypoint/stn.png "MPE网络结构细节")
&emsp;&emsp;其次，为了解决第二个问题，加入了参数化的位姿nms，首先定义两个pose之间的相似度：
![](/images/keypoint/criterion.png "相似度")
&emsp;&emsp;公式中，d为两个pose的距离，即相似度，如果d小于阈值(论文中写的是小于，但实际上我感觉是大于)，则认为两个位姿相似，第i个位姿由于与第j个位姿冗余，所以应该剔除第i个位姿。定义距离d包括两个部分：关键点距离和空间距离，首先关键点距离定义如下，B是以位姿P为中心的盒子，B是以第j个人的第n个关键点k为中心的盒子，其大小为B的1/10，c是置信度，根据这个公式可知，当两个关节都有很高的置信度时，输出应接近1。
![](/images/keypoint/pose_distance.png "关键点距离")
&emsp;&emsp;除关键点的距离外，还有一项是空间距离，定义如下
![](/images/keypoint/spatial_distance.png "空间距离")

&emsp;&emsp;由公式可知，当两个关键点距离很近时，会得到很高的空间距离得分。最终的距离度量是将两项加权组合：
![](/images/keypoint/final_distance.png "距离度量")

&emsp;&emsp;所以这个参数化的nms就是先找到置信度最大的位姿P，抑制那些置信度较高，且与P空间距离很近的pose，至于为什么不直接用置信度最高的pose，我认为可能是人体检测器并不一定检测出一个人，当两个人距离很近，或者部分关节有重合时，在检测一个人的关节时，可能会检测到另一个人的某个关节，所以保留的是那些与pose距离较远，置信度较高的关节。距离公式中共4个参数：sigma1，sigma2，lambda和阈值，这4个参数的求取方法是固定其中两个参数，更改另外两个参数，直至最优。
&emsp;&emsp;解决以上两个问题后，RMPE还有一个在人体检测器检测不完美的人体proposal的情况，仍能得到较好的关键点检测的方法：数据增广。数据增广一个直观的方法是直接使用训练阶段人体检测器生成的边界框，但因为人体检测器只能为每个人生成一个边界框，所以需要大量的生成器。实际上，我们可以用更优化的方法，因为我们已经有每张图片中人体pose的gt和人体检测的边界框，那么我们可以生成大量的与人体检测器的输出分布相同的训练proposal。在实验中发现，不同pose的人体检测的边界框和gt之间的偏移量是不同的，也就是说，某个pose对应的偏移量是服从某个分布P(deltaB|P)的，如果我们能对这种分布建模，那么我们那就能生成许多的训练样本，这些样本是和人体检测器生成的proposal是类似的。而在实际中，直接学习这种分布是很困难的，但是我们可以学习P(deltaB|atom(P))，也就是学习元pose的偏移量的分布。为了得到元pose，先校准所有的躯干，使其长度相同，然后使用k-means方法对校准后的pose进行聚类，计算得到的聚类中心就组成元pose。对于共享同一个元pose的人，计算检测的边界框与gt之间的偏移量，并根据该偏移量方向对应的边界框的gt的side-length归一化该偏移量，经过这些步骤后，偏移量服从某个频率分布，并且我们将数据拟合成高斯混合分布，对于不同的元动作，有不同的高斯混合的参数。

![](/images/keypoint/atomic_pose.png "一些元pose的示例，图片来源于李飞飞的论文Recognizing Human-Object Interactions in Still Images by Modeling the Mutual Context of Objects and Human Poses")
![](/images/keypoint/augmentation.png "不同元pose边界框偏移量不同的高斯分布")

&emsp;&emsp;其他关键点检测的方法：Cascaded Pyramid Network for Multi-Person Pose Estimation(旷视，自下而上的方法), [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://github.com/CMU-Perceptual-Computing-Lab/openpose)(CMU，第一个实时的关键点检测算法)，[Mask RCNN](https://github.com/matterport/Mask_RCNN)(集目标检测，实体分割，关键点检测的万千宠爱于一身)

****************
本次恢复原始文件过程中，与原始文件也存在不同，如本次使用了“一、二”这样的标号，之前是没有的
![](/images/keypoint/former_keypoint.png "之前的版本")


