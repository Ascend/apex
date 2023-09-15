# Ascend apex

## Apex配套软件

| AscendPyTorch版本 | 支持PyTorch版本 | Pytorch Gitee分支名称 | Apex Gitee分支名称 |
| :---------------- | :--------------- | :--------------------- | :----------------- |
| 2.0.2             | 1.5.0.post2      | 2.0.2.tr5             | 2.0.2.tr5          |
| 2.0.3             | 1.5.0.post3      | 2.0.3.tr5             | 2.0.3.tr5          |
| 2.0.4             | 1.5.0.post4      | 2.0.4.tr5             | 2.0.4.tr5          |
| 3.0.rc1           | 1.5.0.post5      | v1.5.0-3.0.rc1        | v1.5.0-3.0.rc1     |
| 3.0.rc1           | 1.8.1.rc1        | v1.8.1-3.0.rc1        | v1.8.1-3.0.rc1     |
| 3.0.rc2           | 1.5.0.post6      | v1.5.0-3.0.rc2        | v1.5.0-3.0.rc2     |
| 3.0.rc2           | 1.8.1.rc2        | v1.8.1-3.0.rc2        | v1.8.1-3.0.rc2     |
| 3.0.rc3           | 1.5.0.post7      | v1.5.0-3.0.rc3        | v1.5.0-3.0.rc3     |
| 3.0.rc3           | 1.8.1.rc3        | v1.8.1-3.0.rc3        | v1.8.1-3.0.rc3     |
| 3.0.rc3           | 1.11.0.rc1 (beta)| v1.11.0-3.0.rc3       | v1.11.0-3.0.rc3    |
| 3.0.0             | 1.5.0.post8      | v1.5.0-3.0.0          | v1.5.0-3.0.0       |
| 3.0.0             | 1.8.1            | v1.8.1-3.0.0          | v1.8.1-3.0.0       |
| 3.0.0             | 1.11.0.rc2 (beta)| v1.11.0-3.0.0         | v1.11.0-3.0.0      |
| 5.0.rc1           | 1.8.1.post1, 1.11.0  | v1.8.1-5.0.rc1, v1.11.0-5.0.rc1 | 5.0.rc1     |
| 5.0.rc2           | 1.8.1.post2, 1.11.0, 2.0.1.rc1  | v1.8.1-5.0.rc2, v1.11.0-5.0.rc2, v2.0.1-5.0.rc2 | 5.0.rc2     |

## 生成全量代码及编译

##### 获取昇腾适配的Ascend apex源码

```
git clone -b master https://gitee.com/ascend/apex.git
cd apex/
```


##### 编译apex的二进制包

1、请确保torch已安装，setuptools版本小于等于65.7.0（不满足时执行pip install setuptools==41.2.0）

2、执行（支持python3.7-3.10，确保python3.x命令存在）
```
bash scripts/build.sh --python=3.7
```
生成的二进制包在apex/dist目录下


## 安装

进入apex/dist目录，执行以下命令：
```
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl version代表python版本和cpu架构
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

