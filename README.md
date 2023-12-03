<div align="center">
<h1 align="center">BEBOCS Algo</h1>
paper：BEBOCS:Bayesian Estimation Based Online Clients Selection

![GitHub](https://img.shields.io/github/license/Sensorjang/BEBOCS_FedML_experiment_SYH)
![Language](https://img.shields.io/badge/Language-Python-blue)

</div>

## 基于FedML框架搭建的BEBOCS算法验证实验
本实验修改了[FedML框架](FedML_README.md)的部分算法逻辑和环节，并且加入了一些新的算子和组件以实现BEBOCS的对比实验的细节<br/>
This experiment modified some algorithm logic and links of the [FedML framework](FedML_README.md), and added some new operators and components to achieve the details of the FARA comparative experiment<br/>

## 实验涉及主要算子和组件信息
- 涉及的算法：BEBOCS(ours)、FedAvg、FedCS
- 涉及的数据集：MNIST、FederatedEMNIST、Cifar10、SVHN
- 涉及的模型：CNN、CNN-Web
<br/><br/>
- Algorithms involved: BEBOCS(ours), FedAvg, FedCS
- Datasets involved: MNIST, FederatedEMNIST, Cifar10, SVHN
- Models involved: CNN, CNN-Web

## 主要路径
- 实验运行路径：[BEBOCS_experiment](python/examples/simulation/bebocs_experiment)
- 算法API路径：[algorithm](python/fedml/simulation/sp)
- 数据集逻辑路径：[dataset](python/fedml/data)
- 模型逻辑路径：[model](python/fedml/model/model_hub.py)
<br/><br/>
- Experimental run path: [BEBOCS_experiment](python/examples/simulation/bebocs_experiment)
- Algorithm API path: [algorithm](python/fedml/simulation/sp)
- Dataset logical path: [dataset](python/fedml/data)
- Model logical path: [model](python/fedml/model/model_hub.py)

## 实验当前进行进度
### 实验启动路径[BEBOCS_experiment](python/examples/simulation/bebocs_experiment)
### 算法逻辑[algorithm](python/fedml/simulation/sp)
- [x] BEBOCS算法实现(初步实现、细节逻辑需修改[BEBOCS](python/fedml/simulation/sp/bebocs))
- [x] FedAvg算法实现
- [x] FedCS算法实现
### 数据集[dataset](python/fedml/data)
- [x] MNIST数据集
- [x] FederatedEMNIST数据集
- [x] Cifar10数据集
- [x] SVHN数据集
### 模型[model](python/fedml/model/model_hub.py)
- [x] MNIST数据集对应的CNN模型
- [x] FederatedEMNIST数据集对应的CNN模型
- [x] Cifar10数据集对应的CNN-Web模型
- [x] SVHN数据集对应的CNN-Web模型
### 备注
- [ ] 部分来自yaml的参数不生效的问题(例如：global_client_num_in_total、global_client_num_per_round等，目前使用临时常量代替)

## 快速开始
在['python/'](python/)目录下执行：<br/>
Execute in the ['Python/'](python/) directory:<br/>
```bash
pip install .
```
然后执行[BEBOCS_experiment](python/examples/simulation/bebocs_experiment)下的各个python文件即可：<br/>
Then execute the various Python files under [BEBOCS_experiment](python/examples/simulation/bebocs_experiment) to:<br/>
```bash
torch_fedavg_step_by_step_example.py
torch_fedcs_step_by_step_example.py
torch_bebocs_step_by_step_example.py

(使用 --cf 指定并使用特定的外部配置文件)
(Use --cf to specify and use a specific external configuration file)
```
程序会自动绘制图像，并且在控制台输出绘图坐标信息<br/>
The program will automatically draw images and output drawing coordinate information on the console<br/>
绘图借助plt，在图像绘制中途程序处于阻塞状态，预定的全部FL轮次完成后图像会保持输出在用户界面上供保存和查看细节<br/>
Drawing with the help of PLT, the program is in a blocked state during the process of image drawing. After all predetermined FL rounds are completed, the image will remain output on the user interface for saving and viewing details<br/>

## License
该项目基于[Apache-2.0 License](LICENSE)许可证开源<br/>
This project is released under the [Apache-2.0 License](LICENSE).<br/>