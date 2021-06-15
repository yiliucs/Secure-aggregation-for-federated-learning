# Aggregation Service for Federated Learning: An Efficient, Secure, and More Resilient Realization
---
## 代码说明
## 1.数据集
### 1.1数据集下载
本次版本的代码添加了一个数据集[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)，CelebA是CelebFaces Attribute的缩写，意即名人人脸属性数据集，其包含10,177个名人身份的202,599张人脸图片，每张图片都做好了特征标记，包含人脸bbox标注框、5个人脸特征点坐标以及40个属性标记，CelebA由香港中文大学开放提供，广泛用于人脸相关的计算机视觉训练任务，可用于人脸属性标识训练、人脸检测训练以及landmark标记等。同时，CelebA也是联邦学习（FL）标准数据集[LEAF](https://leaf.cmu.edu/)中的一部分。由于数据集过大，无法通过PyTorch封装的代码直接获得，获取方式如下：
> [1.My Google Drive](https://drive.google.com/drive/folders/1XegONA2EQzPPO5h-0sUp-hbzGNvyXHTe?usp=sharing) （推荐）
> 
> [2.Google Drive](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8)
> 
> [3.Baidu Drive](https://pan.baidu.com/s/1CRxxhoQ97A5qbsKO7iaAJg)

注意：本文中是利用这些数据做性别分类任务，因此，需要运行*loading_data.py*来获取标签文件，如下：
 ```python=
python loading_data.py
```

---
### 1.2数据集展示
![](https://codimd.xixiaoyao.cn/uploads/upload_7a764e819d2d6efd7b447224bdbc7ca4.png)
For more details of the dataset, please refer to the paper ["Deep Learning Face Attributes in the Wild".](https://liuziwei7.github.io/projects/FaceAttributes.html)

---

### 1.3数据集引用
```
 @inproceedings{liu2015faceattributes,
 title = {Deep Learning Face Attributes in the Wild},
 author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {December},
 year = {2015} 
}
```


## 2.运行说明
- 环境配置
 ```python=
conda env create -f environment.yml
```
注意:请注意服务器上nvidia的版本，请确认一下是否可以安装我这个服务器所带的环境，这个很重要。
### 2.1 Baseline 
- Baseline: 不加Scaling与量化。
 ```python=
python main_fed.py --dataset [dataset_name] --iid --num_channels 1 --model cnn --epochs 200 --gpu [GPU_number]
```
注意：main_fed.py中不包括新加的数据集CelebA，**若要运行CIFAR-10数据集则num_channels=3且模型为AlexNet**。MNIST和CIFAR-10数据集可以不使用GPU。

- CelebA数据集
 ```python=
python main_fed_celeba.py --dataset celeba --iid --num_channels 3 --model cnn --epochs 200 --gpu 1
```
注意：**此程序运行必须要使用GPU且运行时间大于十小时**。如果使用服务器挂载运行，建议使用nobhup来运行。

---

### 2.2 Scaling方案
- Scaling方案
 ```python=
python main_fed_scaling.py --dataset [dataset_name] --iid --num_channels 1 --model cnn --epochs 200 --scaling_factor 10 --gpu [GPU_number]
```

---

### 2.3 Quan方案
- Quan-16bit方案
 ```python=
python main_quan_celeba.py --dataset [dataset_name] --iid --num_channels 1 --model cnn --epochs 200 --bit_width 16 --alpha [0.5 or 0.1] --gpu [GPU_number]
```
- Quan-8bi方案 
```python=
python main_quan_celeba.py --dataset [dataset_name] --iid --num_channels 1 --model cnn --epochs 200 --bit_width 8 --alpha [0.5 or 0.1] --gpu [GPU_number]
```

---
## 3.模型结构、参数量以及大小
```python=
python model_para.py
```
结果如下：
```
Total params: 14,028,106
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 23.34MB
Total MAdd: 1.54GMAdd
Total Flops: 770.57MFlops
Total MemR+W: 100.51MB

resnet18 have 14028106 paramerters in total
```
---

