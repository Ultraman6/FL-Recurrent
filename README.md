<div align="center">
<h1 align="center">Federated Learning Recurrent</h1>

![Language](https://img.shields.io/badge/Language-Python-blue)

</div>

## 基于FedML框架搭建联邦学习经典算法复现环境
本项目旨在修改[FedML框架](FedML_README.md)的部分算法逻辑和环节、加入一些新的算子和组件来复现各大paper的联邦学习算法<br/>

## 主要路径
- 实验运行路径：[FL_recurrent](python/examples/simulation/FL_recurrent)
- 算法API路径：[algorithm](python/fedml/simulation/sp)
- 数据集逻辑路径：[dataset](python/fedml/data)
- 模型逻辑路径：[model](python/fedml/model/model_hub.py)
<br/><br/>

## 复现进度

### FedAB [原文](https://ieeexplore.ieee.org/abstract/document/10092911)
#### 算法实现 
- [ ] [FedAB](python/fedml/simulation/sp/fedab)
- 变量关系：
- [ ] 实验对比算法 [Optimal](python/fedml/simulation/sp/fedavg)、[Bid-First](python/fedml/simulation/sp/fedavg)、[Random](python/fedml/simulation/sp/fedavg)、[ε-Greedy](python/fedml/simulation/sp/fedavg)
#### 配置实现
- [x] 模型 [CNN](python/fedml/model/cnn.py)
- [x] 数据集 [MNIST](python/fedml/data/mnist.py) [FMNIST](python/fedml/data/fmnist.py) [CIFAR10](python/fedml/data/cifar10.py)
#### 实验结果复现
- [ ] Total reward performance under different clients K in each round. (a) K = 10. (b) K = 15. (c) K = 20.
- [ ] Best test accuracy under different budget G and clients K.(a)G = 300. (b) G = 400. (c) G = 500.
- [ ] Test accuracy and loss of all datasets.
- [ ] Individual rationality on client id and

### DDS [原文](https://ieeexplore.ieee.org/abstract/document/10092911)
#### 算法实现 
- [ ] [DDS](python/fedml/simulation/sp/fedab)
- 变量关系：
- [ ] 实验对比算法 [RandFL](python/fedml/simulation/sp/fedavg)、[FixFL](python/fedml/simulation/sp/fedavg)
#### 配置实现
- [x] 模型 [MLP](python/fedml/model/cnn.py)
- [x] 数据集 [MNIST](python/fedml/data/mnist.py) [FMNIST](python/fedml/data/fmnist.py) [CIFAR10](python/fedml/data/cifar10.py)
#### 实验结果复现
- [ ] Total reward performance under different clients K in each round. (a) K = 10. (b) K = 15. (c) K = 20.
- [ ] Best test accuracy under different budget G and clients K.(a)G = 300. (b) G = 400. (c) G = 500.
- [ ] Test accuracy and loss of all datasets.
- [ ] Individual rationality on client id and



## 快速开始
在['python/'](python/)目录下执行：<br/>
Execute in the ['Python/'](python/) directory:<br/>
```bash
pip install .
```
然后执行[FL_recurrent](python/examples/simulation/FL_recurrent)下的各个python文件即可：<br/>
Then execute the various Python files under [FL_recurrent](python/examples/simulation/FL_recurrent) to:<br/>
```bash
torch_fedab_step_by_step_example.py

(使用 --cf 指定并使用特定的外部配置文件)
(Use --cf to specify and use a specific external configuration file)
```
程序会自动绘制图像，并且在控制台输出绘图坐标信息<br/>
借助plt绘图，在图像绘制中途程序处于阻塞状态，预定的全部FL轮次完成后图像会保持输出在用户界面上供保存和查看细节<br/>


