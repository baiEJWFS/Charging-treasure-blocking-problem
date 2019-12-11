# 2019BUAA机器学习大作业-遮挡问题 #
+ 小组成员为**智相谦ZF1921250**&**王思其ZF1921216**&**孙境棋ZF1921344**&**李雅妍ZF1921105**&**刘彦君ZF1921223**.
## Introduce ##
+ 针对此次大作业，我们充分了解了目标检测的相关算法，对主流算法进行了对比，最终我们选取**SSD**作为模型。我们先对数据集进行简单交叉验证(**90%为训练集，10%为测试集**)，并在GPU上进行训练，多次调整参数使其得到更好的效果。经过**8.4K**次迭代，最终得到的mAP为**86.19**
## Detial ##
#### Environment  ####
+ **python3.7**: python使用的版本为3.7
+ **PyTorch 1.1**: pytorch版本为1.1
+ **Work on GPU or CPU**: 使用DistributedDataParallel进行多GPU并行计算。我们没有使用CPU进行测试，但理论上支持CPU.
+ **Hardware**: 2张NVIDIA Tesla V100 32GB
### Usage ###
+ 我们提供的模型在CoverWithCharger/weigths
###### 1. Install pytorch
+ 我们使用的python版本为3.7
+ 你可以点击此处来安装pytorch [this](https://github.com/pytorch/pytorch)
###### 2. Clone the repository:
###### 3. Dataset
+ Our dataset is private
###### 4. Evaluation
```
cd CoverWithCharger
python eval_5epoch_for.py --trained_model you_own_model_root
```
###### 5. Training:
```
cd CoverWithCharger
python train.py --sixray_root data_root --imagesetfile data_set_root
```
### Result ###
我们提供我们的训练数据和结果，如下图所示：
+ 1 .![mAP](_v_images/20191211151141278_22710.png =900x)
+ 2 .![Loss](_v_images/20191211151212042_2386.png =900x)