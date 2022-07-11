# Ascend apex

## Apex配套软件

| AscendPyTorch版本 | 支持PyTorch版本 | Pytorch Gitee分支名称 | Apex Gitee分支名称 |
| :---------------- | :-------------- | --------------------- | :----------------- |
| 2.0.2             | 1.5.0.post2     | 2.0.2.tr5             | 2.0.2.tr5          |
| 2.0.3             | 1.5.0.post3     | 2.0.3.tr5             | 2.0.3.tr5          |
| 2.0.4             | 1.5.0.post4     | 2.0.4.tr5             | 2.0.4.tr5          |
| 3.0.rc1           | 1.5.0.post5     | v1.5.0-3.0.rc1        | 3.0.rc1            |
| 3.0.rc1           | 1.8.1.rc1       | v1.8.1-3.0.rc1        | 3.0.rc1            |
| 3.0.rc2           | 1.5.0.post6     | v1.5.0-3.0.rc2        | 3.0.rc2            |
| 3.0.rc2           | 1.8.1.rc2       | v1.8.1-3.0.rc2        | 3.0.rc2            |

## 生成全量代码及编译

注：以下描述中的“根目录”指“Ascend apex的根目录”

##### 获取昇腾适配的Ascend apex源码

```
git clone -b master https://gitee.com/ascend/apex.git
```

##### 获取原生apex源代码

从github上获取源代码，在根目录下执行
```
git clone https://github.com/NVIDIA/apex.git
```
进入源代码目录，切换至commitid为4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a的分支
```
cd apex
git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
cd ..
```

##### 生成适配昇腾AI处理器的apex代码

进入到scripts目录，执行
```
bash gen.sh
```
会在根目录下apex目录中生成npu适配全量代码

##### 编译apex的二进制包

1、请确保npu版本的pytorch可以正常使用（否则会影响apex的编译）；

2、进入到根目录下apex目录，执行
```
python3 setup.py --cpp_ext --npu_float_status bdist_wheel
```
生成的二进制包在当前的dist目录下


## 安装

进入dist目录，执行以下命令：
```
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-cp37-cp37m-linux_{arch}.whl arch表示架构，为aarch64或x86_64
```


## 特性
**已支持：**
- [x] O1模式
- [x] O2模式
- [x] 静态 loss scale
- [x] 动态 loss scale
- [x] combine tensors
- [x] combine grad for unscale
- [x] npu fused optimizer: adadelta, adam, adamp, adamw, sgd, lamb, rmsprop, rmsprop_tf
- [x] 动态 loss scale新增dynamic_init_scale, scale_growth_factor, scale_backoff_factor, scale_window等可调参数

**说明：**

当前版本的实现方式为python实现，不支持acl或者cuda优化。


## 使用方法
**自动混合精度：**

请参考https://nvidia.github.io/apex/amp.html

**combine grad for unscale：**

在amp.initialize()中将参数combine_grad设置为True

**npu fused optimizer：**

将原有优化器替换为apex.optimizers.xxx, 其中xxx为融合优化器名称

