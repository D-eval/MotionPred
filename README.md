# motion transformer diffusion

## 模型介绍

* `post_mt_diffusion`: 后验Transformer diffusion, 编码了未来信息的 _context_ 和历史姿态同时作为输入。

* `prior_mt_diffusion`: 先验Transformer diffusion, 仅用历史姿态作为输入。

* `psyPred`: 用过去姿态得到未来 _context_ 的分布，从而预测未来的姿态。

## 数据集

从AMASS下载数据集。

## 训练

```
python3 train_mt_multi_diffusion.py
```

## 实时检测

预测曲线

![](./curve.png)

真实曲线

![](./curve_real.png)

## 方法提出

我们先来介绍模型的特殊能力，再介绍整体思路。

* 我们希望提出了一种能够建模不确定性的多步预测方法，其中未来动作的不确定性由扩散模型的采样来决定。
* 我们希望能模拟不同条件下的预测，用条件和历史输入共同决定未来的动作。我们希望有额外条件输入的和没有额外条件输入的情况下，二者无论隐层还是最终输出结果不会相差太多。因此我们受卡尔曼滤波后验修正的启发，提出一种称为后验旋转`PosteriorRotation`的模块。

该模块会将条件输入用一个微旋转矩阵表示，并将该旋转矩阵作用于Decoder的隐层上，对每个时刻的表示作用不同的旋转，从而模拟后验信息对隐层的扰动。

模型的整体思路是，加噪的关节通过decoder中的CrossAttention看到其历史信息，从而去噪。

历史信息需要保留原始数据的动态，同时一定程度上聚合其他关节的信息。这要求我们的encoder不能太复杂，否则，将信息平摊会让decoder的动态表达能力降低。

总之，我们的设计要求是 _去噪要高度依赖历史信息编码_。解决方法是：

* Decoder只保留CrossAttention模块。防止SpatialAttention破坏骨骼间的差异，也防止SelfTemporalAttention破坏时间上的差异，使得生成结果保守，缺乏时间上的动态和空间上的姿态丰富度。

* Encoder要保留历史信息的动态，并适当的融合关节间的信息。

具体介绍每个模块

### TCSA-Encoder

为了达到保留动态、融合全局的目的，我们设计了一种 Temporal Conv Spatial Attention Encoder (TCSA-Encoder)，用于

我们设计的编码器再时间上用一维卷积捕捉动态时序信息，空间上用 Spatial Attention 捕捉骨骼姿态之间的相关信息，允许关节再尽可能保留原始姿态的情况下融合了局部的动态和全局的姿态。

### Posterior-Rotation

#### 介绍

后验编码器要有较强的信息压缩能力，对于Decoder隐层为 $D$ 维的情况，PosteriorEncoder可以将任意长的时间数据压缩成一个 $(C,D)$ 的表示。其中$C$为旋转矩阵参数的个数:

$$ C = \frac{1}{2}D(D-1) $$

由于任意旋转矩阵都可以表示为一个斜对称矩阵$A$的指数矩阵，我们取一个微小旋转矩阵$A$，其指数矩阵可以近似为$R = I+A$。

后验旋转模块从形状为 $(B,T,N,D)$ 的decoder隐层输出中，接收某个时刻某个关节的表示 $z$ ，其形状为 $(D)$，送入CrossAttention作为query，并将PosteriorEncoder得到的形状为 $(C,D)$ 的后验表示作为key，该模块不需要value，或者将1作为value。用query和key求的 $C$ 个相似度作为参数构造斜对称矩阵$A$。随后将 $R=A+I$ 作用在 $z$ 上得到经过后验修正的 $z$ 。

#### 实现

实现的关键在于如何把任意时间长度的数据压缩到 $C$ 。同样需要保留动态信息，我们采用相同的编码器将姿态序列输入 $(B,T,N,d)$ 映射为 $(B,T,N,D)$ 。然后再通过一些其他的方式压缩得到 $(B,C,N,D)$

压缩的关键在于寻找 $C$ 个未来的关键帧作为指引。我们采取了一种token化的方法：定义 $C\times N$ 个可学的 $D$ 维向量作为 query，与形状为 $(B,T,N,D)$ 姿态序列编码进行 CrossAttention 得到了压缩后的 $(B,C,N,D)$。

由于 