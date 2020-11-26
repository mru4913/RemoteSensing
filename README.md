# “华为·昇腾杯”AI+遥感影像-复赛

## 队名：借过一下

---
## 算法思路
本次赛题任务是识别高分辨率光学遥感图像中具有语义信息的各个像元所属的语义类别标签，针对高分辨的图像语义分割任务，我们选用医学影像（通常是大分辨率影像）分割中常用的unet（encode-decode）结构作为基础模型，并根据训练模型在验证集上的性能，修改网络结构；最后，根据线上评测集的评测结果，优化模型后处理方法。


### `模型设计的关键点`：

1. 为学习高分辨率图像的语义信息，采用unet++的nest结构，学习更详细的特征信息；
2. 为增加模型的泛化性，unet++的backbone部分采用[efficient-b4](https://arxiv.org/abs/1905.11946) 预训练参数作为encode结构;
3. 为减少梯度消失现象，unet++的nest和decode的conv-block采用resnet block;
4. 为增加输出特征的感受野信息，unet++的4个decode层级，引入空洞卷积，同时，对unet++的中间层级nest结构均引入attentionunet提出的注意力机制；

### `模型训练的关键点`： 

1. 采用2080ti单卡训练；
2. 损失函数使用CrossEntropyLoss，优化器用adamw，初始学习率0.000125，学习率策略CosineAnnealingWarmRestarts

### `模型推断后处理的关键点`： 

1. 复赛线上的评测集大小分布为256-5000多尺寸图像，为使模型在多分辨率数据集上保持较优的推断性能，采用 `Overlap tiles during inferencing` 方案，主要思路如下： 

    1.1 在2080ti单卡环境，计算模型推断时batchsize最大阈值为多少；

    1.2 计算不同overlap下，不同尺寸的batchsize数值；

    1.3 在时间和精度上，针对不同尺寸的图像选择最优的overlap组合方案；

2. 对于256大小的图像，采用[tta](https://github.com/qubvel/ttach)方法；


---
## 硬件环境
1. 本次参加大赛，用的是`单卡RTX2080ti gpu`训练，训练时由于gpu性能计算有限batchsize=16，算法需要的环境与主办方复赛提供的测试环境一致

---
## 解题思路
1. 针对高分辨图像的语义分割，借鉴医学高分辨影像的病灶分割思路；
2. 针对多尺寸模型的推断，在时间与精度的权衡下，采用上述`Overlap tiles during inferencing` 方案；

--- 
## 模型复现
模型训练过程分两步，如下：
1. 先在划分好的数据集（训练集：验证集=8:2）上训练230epoch模型（原计划训练260，由于时间问题，只训练到230epoch）,获取在验证集上最优的模型文件（./checkpoints/dilate_ags_unet_002_noshare_aug_best_220epoch.pth），该模型线上评测fwiou=0.56040517;
2. fine-tune训练：将步骤1划分的验证集选取19/20加入训练集，只留1/20作为验证集，用最优模型继续再训练，得到最好文件（./checkpoints/dilate_ags_unet_008_aug_epoch220_retrain_epoch=69.pth，原计划训练120epoch，由于时间比较紧，故只训练到69epoch）,该模型结合最后的后处理方案，线上评测fwiou=0.56927503；
```shell
# step 1: split train data and valid data
$ cd utils/
$ python train_valid_split.py # replace your data path 

# step 2: training
$ sh run.sh  

# step 3: retraining
$ sh run.sh  # edit run.sh, comment step1, and uncommet step2
```

