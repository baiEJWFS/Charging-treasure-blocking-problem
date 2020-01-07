# 机器学习大作业-遮挡问题 #
+ 小组成员：**智相谦ZF1921250** **王思其ZF1921216** **孙境棋ZF1921344** **李雅妍ZF1921105** **刘彦君ZF1921223**
## 简介 ##
+ 在做作业之前，我们大概了解了几种目标检测算法，有（1）two-stage方法，如R-CNN系算法，其主要思路是先通过启发式方法（selective search）或者CNN网络（RPN)产生一系列稀疏的候选框，然后对这些候选框进行分类与回归，two-stage方法的优势是准确度高；（2）one-stage方法，如Yolo和SSD，其主要思路是均匀地在图片的不同位置进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类与回归。
+ 最终我们选择了SSD作为使用的模型，首先划分数据集，训练集和验证集的比例为9:1（5400:600），在GPU上进行训练，最后得到的mAP为0.86。
## 环境  ##
+ **python3.7**
+ **PyTorch 1.1**
+ **GPU**
### 使用 ###
+ 模型文件：Release版本里面 或者 CoverWithCharger/weigths
######  测试:
```
cd CoverWithCharger
python eval_5epoch_for.py --trained_model you_own_model_root
```
###### 训练:
```
cd CoverWithCharger
python train.py --sixray_root data_root --imagesetfile data_set_root
```
### mAP结果 ###

![mAP](/img/result.png)
### 过程曲线 ###

![mAP](/img/loss.png)
