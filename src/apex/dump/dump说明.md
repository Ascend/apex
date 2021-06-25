# Pytorch module dump功能

## 一、实现原理

### 功能

能dump出规定module epoch和step的每层数据，Dump完成后，可以 remove hook的handle。

依赖Pytorch Hook机制，采用Pytorch Module的forward hook和backward hook分别dump出每个module的前向和反向输出。

### 存放位置

该相关文件放在[Ascend](https://gitee.com/ascend) 社区apex仓库下的apex/dump文件夹。

### 安装包依赖

HDF5支持安装

```
sudo apt-get install libhdf5-dev  # 安装头文件依赖
sudo pip install h5py   # 安装h5py
```

apex包安装

```
pip install --upgrade apex-*.whl
```

### Forward hook

正向数据输出每个在module层：

```python
# 定义 forward hook function
def hook_fn_forward(module, input, output):
    print(module) # 用于区分模块
    print('input', input) # 首先打印出来
    print('output', output)

model = Model()
modules = model.named_children() # 获取子模块
for name, module in modules:
    module.register_forward_hook(hook_fn_forward) #子模块注册
```

### Backward hook

反向梯度数据输出每个层：

```python
def hook_fn_backward(module, grad_input, grad_output):
    print(module) # 为了区分模块
    print('grad_output', grad_output) # 为了符合反向传播的顺序，我们先打印 grad_output
    print('grad_input', grad_input)  # 再打印 grad_input

model = Model()
modules = model.named_children()
for name, module in modules:
    module.register_backward_hook(hook_fn_backward)
```

按照输出和输入， 顺序序列化每个tensor。

### 接口定义

```python
import torch
from torch import nn
import h5py

class HookerAndDumper():
def  __init__(self, file_path = "", is_screen_display=False):
    # 参数初始化
def forward_hook_fn(self, module, input, output):
    # dump前向数据
def backward_hook_fn(self, module, grad_input, grad_output):
    # dump 后向数据
    # epoch 和 step用来标志文件名称
def register_hooks(self, epoch, step):
    # 子模块注册
def remove_hooks(self):
    # remove 所有的handle
def read_h5py_data(self):
    # 读取存储的hdf5文件的存储结构 group dataset
    
```

## 二、功能调用说明

以resnet50为例进行说明

### 导入模块dump.py

从apex模块中导入

```
import apex.dump.dump as dump
或者
from apex.dump.dump import HookerAndDumper
```

注：需要安装h5py包，pip install h5py

### ForwardAndBackwardDumper类初始化

```
fwb = dump.HookerAndDumper(file_path="···/": str) -> HookerAndDumper
```

参数说明：

file_path：设置存储的文件路径，最后以”/”结束，默认存储在当前路径下。

### hook注册

```
fwb.register_hooks(model=model: module, epoch=0: int, step=3: int) -> handle
```

参数说明：

model：计算的model，会对每个子模块进行注册。

epoch：需要dump的epoch位置，需要将该注册函数放在相应epoch位置，该函数会将epoch信息添加到生成的文件名当中。

step：同上，需要dump的step位置，请将该注册函数放在相应step位置，该函数会将step信息添加到生成的文件名当中。

其间，会有子模块注册和Dump数据到hdf5文件的信息在屏幕打印。

### 关闭dump功能

```
fwb.remove_hooks()
```

功能说明：

关闭hook 的handle，请将该函数放在对应epoch，step结束之后。

显示关闭hook和文件成功信息，反馈dump过程中写入的文件路径与文件名信息。

### 读取hdf5文件

```
fwb.read_h5py_data()
```

功能说明：

读取文件，在屏幕端显示hdf5文件格式的group和group下dataset信息，可根据显示的信息对文件内容进行访问，详细访问方式在下章节介绍。

## 三、HDF5格式文件读写方法

###  介绍

HDF5是一种存储相同类型数值的大数组的机制，适用于可被层次性组织且数据集需要被元数据标记的数据模型，常用的接口模块为 h5py，便于进行数据的比对。

**- hdf5 files**： 能够存储两类数据对象 dataset 和 group 的容器，其操作类似 **python 标准的文件**操作；File 实例对象本身就是一个组，以 `/` 为名，是遍历文件的入口。

**- dataset(array-like)**： 可类比为 Numpy 数组，每个数据集都有一个名字（name）、形状（shape） 和类型（dtype），支持切片操作。

**- group(folder-like)**： 可以类比为 字典，它是一种像文件夹一样的容器；group 中可以存放 dataset 或者其他的 group，键就是组成员的名称，值就是组成员对象本身(组或者数据集)。

### 安装

```
pip install h5py
```

### read

```python
>>> import h5py
>>> f = h5py.File('mytestfile.hdf5', 'r')
>>> dset = f['mydataset'] # 通过键值获得dataset或group
>>> dset.shape
>>> dset.dtype
```

```python
>>> import torch
>>> import h5py
>>> torch.set_printoptions(profile="full")
>>> f = h5py.File("0-epoch_3-step_dump.hdf5")
>>> data = f['018_BatchNorm2d/grad_input0']
>>> print(data)
>>> print(data.value)
>>> print(data[:])
```

### Write

```python
>>> import h5py
>>> import numpy as np
>>> f = h5py.File("mytestfile.hdf5", "w")
>>> dset = f.create_dataset("mydataset", (100,), dtype='i') # 直接创建dataset

>>> arr = np.arange(100)
>>> dset = f.create_dataset("init", data=arr) # 直接传入data，无需设置dtype和shape类型，会根据arr类型自动设置

>>> f2 = h5py.File('mydataset2.hdf5', 'a')
>>> grp = f2.create_group("subgroup")
>>> dset2 = grp.create_dataset("another_dataset", (50,), dtype='f') # 在group上创建dataset
```

### 遍历文件

**f.Key()**

```python
import h5py
with h5py.File('cat_dog.h5', "r") as f:
    for key in f.keys():
    	# 若是group对象，则没有value属性的,会包异常。
    	print(f[key], key, f[key].name, f[key].value) 
		# f[key] means a dataset or a group object. 
		# f[key].value visits dataset' value, except group object.
        print(f[key], key, f[key].name) 

```

**f.visit()**

```python
with h5py.File(self.file_path, "r") as f:
        def print_name(name):
            print(name)
            dset=f[name]
            if isinstance(dset, h5py.Dataset): # 判断为Dataset成员类型，不是Group
            	# 可对获取的dataset数据进行相应操作
				print(dset.dtype)  # dset数据类型
                print(dset.shape)  # dset shape格式
                print(dset[:]) # 显示dset数据信息，也可用dset.value
        f.visit(print_name) # 遍历hfd5格式信息传导到输入的函数中作为参数name

```

