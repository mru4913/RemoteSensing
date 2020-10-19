# RemoteSensing
Remote Sensing Competition 

repo会根据时间推移进行完善。这里先简单书写一下。如果有什么疑问，请随时联系我by mengda_yu@126.com

# Preliminary Contest
## Introduction

初赛主要是采用Unet family， encoder部分引入不同的pretrained backbone， decoder部分加入attention 机制，如SCSE attention block, Loss function主要用了lovasz, cross entropy等常见的计算方式。优化器是Adam，并且运用cosine Annealing去调节学习率。图像增强也是常见的，如90度倍数旋转，水平垂直翻转。之后用SGD进行fine-tuning。在一张11GB的GPU运行2天左右。

## Train

```
python ../baseline/train.py ../train_valid_split/all_train_list.txt ../train_valid_split/all_valid_list.txt --num_epochs 28 --lr 0.0001 --log_num 10 --save_model 1 --img_aug 1 --batch_size 16 --loss ce --weight_decay 0 --model eunet --cuda_device 1 --scheduler cosr --optimizer sgd --pretrained 0 --load e100_eunet1_ce_adam_cosr_e127_lr0001_b16_Aug.pth --name e100_eunet1_ce_sgd_cosr_e28_lr0001_b16_Aug

python ../baseline/train.py ../train_valid_split/all_train_list.txt ../train_valid_split/all_valid_list.txt --num_epochs 28 --lr 0.0001 --log_num 10 --save_model 1 --img_aug 1 --batch_size 16 --loss ce --weight_decay 0 --model eunet --cuda_device 1 --scheduler cosr --optimizer sgd --pretrained 0 --load e100_eunet1_ce_adam_cosr_e127_lr0001_b16_Aug.pth --name e100_eunet1_ce_sgd_cosr_e28_lr0001_b16_Aug
```

#### MODEL URL:https://pan.baidu.com/s/4gkqHIqn




