# Ascend 910 版本apex安装使用说明


## 生成全量代码及编译
1、请确保npu版本的pytorch可以正常使用；

2、进入到build目录，执行
```
bash build.sh
```
会在当前根目录下apex-npu目录中生成npu适配全量代码，同时在dist目录中生成whl包。执行
```
bash build.sh gen
```
只生成全量代码。


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
- [x] npu fused optimizer: adadelta, adam, sgd, lamb

**说明：**

当前版本的实现方式为python实现，不支持acl或者cuda优化。


## 使用方法
**自动混合精度：**

请参考https://nvidia.github.io/apex/amp.html

**combine grad for unscale：**

在amp.initialize()中将参数combine_grad设置为True

**npu fused optimizer：**

将原有优化器替换为apex.optimizers.xxx, 其中xxx为融合优化器名称

