# Apex Patch

## 一、简介

### 1.1 Apex Patch功能介绍

Apex Patch以代码patch的形式发布，用户通过对原始Apex进行patch，可以在华为昇腾AI处理器上，使用Apex的自动混合精度训练功能进行模型训练，提升AI模型的训练效率，同时保持模型的精度和稳定性。此外，Apex-patch额外提供了如梯度融合、融合优化器等，以提升部分场景下模型在昇腾NPU上的训练效率，供用户选择使用。

### 1.2 Apex-patch代码目录说明

   ```
    ├── Apex
        ├──patch
            ├──npu.patch              # Apex-patch对于原生Apex的patch文件，用于原生Apex中混合精度等功能基于昇腾AI处理器的适配
        ├──scripts
            ├──docker/                # 使用Docker构建安装包的脚本与说明文件
            ├──build.sh               # 构建安装包脚本
            ├──gen.sh                 # 使用Apex-patch对官方Apex打patch的脚本
            ├──make_patch.sh          # 生成patch文件脚本
        ├──src
            ├──apex
                ├──contrib/           # 提供Tensor融合的Python API，供融合优化器使用
                ├──optimizers/        # 融合优化器的实现，部分场景下发挥昇腾的算力
            ├──csrc/combine_tensors/  # 提供Tensor融合的C++接口
        ├──tests                      # 测试用例
        ├──LICENSE
   ```

### 1.3 Apex-patch已支持特性

- [x] O1/O2自动混合精度模式
- [x] 动态/静态 loss scale
- [x] combine ddp
- [x] combine grad for unscale
- [x] npu fused optimizer: adadelta, adam, adamp, adamw, sgd, lamb, rmsprop, rmsprop_tf
- [x] 动态 loss scale新增可调参数，如：dynamic_init_scale, scale_growth_factor, scale_backoff_factor, scale_window


## 二、生成全量代码及编译、安装

*推荐使用docker，在容器内编译：[参考链接](https://gitee.com/ascend/apex/tree/master/scripts/docker/README.md)

建议用户以非root用户做环境的安装。若使用容器环境编译，建议使用普通用户，本仓库提供的Dockerfile仅供参考。请用户关注容器挂载目录安全性，避免系统路径，推荐只挂载业务路径，避免不必要的安全问题。

### 2.1 获取昇腾适配的Apex-patch源码

```
git clone -b master https://gitee.com/ascend/apex.git
cd apex/
```

### 2.2 编译apex的二进制包

1、请确保torch已安装，setuptools版本小于等于65.7.0（不满足时执行pip install setuptools==41.2.0）

2、执行（支持python3.7-3.10，确保python3.x命令存在）
```
bash scripts/build.sh --python=3.7
```
生成的二进制包在apex/dist目录下

### 2.3 安装

进入apex/dist目录，执行以下命令：
```
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl version代表python版本和cpu架构
```

如需要保存安装日志，可在pip3 install命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

### 2.4 卸载

Pytorch框架训练环境的卸载可以参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。

Apex及Apex-patch的卸载只需执行命令：

  ```python
  pip3 uninstall apex
  ```

如需要保存卸载日志，可在pip3 install命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

### 2.4 安全加固（可选）

##### 2.4.1 安全风险提示

建议您务必对模型训练相关文件（如数据集、配置文件、源代码、checkpoint等）做好权限管理，避免文件被恶意篡改、破坏业务进行等风险，比如可以控制为同组/其他用户仅有只读权限。

##### 2.4.2 权限风险提示

- Linux 系统的 umask 值建议不低于027。
- 不建议使用管理员账户安装运行，建议安装完成后对安装目录文件做好权限管控，文件及文件夹权限建议设置为 550。
- 如需要保存安装日志，可在安装时对标准输出进行重定向， 重定向文件做好权限管控。
- 运行程序前，建议用户对训练所需文件，模型脚本、权重等做好权限控制等安全措施，权限建议设置为 640。
- 在多用户共享数据集的场景下，请根据需求最小化权限设置所需的文件夹以及文件的读写权限等，避免出现非法访问等安全问题。
- 原生框架中，在使用某些特定特性时，可能会在`~/.cache`文件夹下生成临时文件，建议用户也应对`~/.cache`文件夹做好权限控制，避免安全风险。
- 对于涉及到使用 C++ 动态编译特性的场景，建议打开 ASLR （地址空间配置随机加载）以及对编译后的 SO 文件开启 strip（移除调试符号信息），减少程序的暴露面。 因Apex-patch源码发布由用户自行编译，故需用户自行开启此类选项，开启方法参考下方章节。

##### 2.4.3 打开 ASLR

```
echo 2 > /proc/sys/kernel/randomize_va_space
```

##### 2.4.4 对change_data_ptr等动态编译的so文件开启strip

安装软件后会在site-packages目录下产生change_data_ptr.{version}.so，用户可自行对该动态库开启strip。

```
strip -s /xxx/python3/lib/site-packages/change_data_ptr.{version}.so  # 目录路径仅供参考
```

## 三、使用方法

### 3.1 自动混合精度

使用apex.amp进行混合精度训练，如：
```
model = torch.nn.Linear(D_in, D_out).cuda()
optimzier = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
  scaled_loss.backward()
...
```

详细使用方式请请参考https://nvidia.github.io/apex/amp.html。

### 3.2 使用融合梯度进行scale/unscale

在amp.initialize()中将参数combine_grad设置为True，如：
```
model = torch.nn.Linear(D_in, D_out).cuda()
optimzier = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer, opt_level='O1', combine_grad=True)  # 增加combine_grad参数
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
  scaled_loss.backward()
...
```

### 3.3 融合优化器

将原有优化器替换为apex.optimizers.xxx, 其中xxx为融合优化器名称，如：
```
model = torch.nn.Linear(D_in, D_out).cuda()
optimzier = apex.optimizers.NpuFusedSGD(model.parameters(), lr=1e-3) # 使用apex.optimizers.NpuFusedSGD

model, optimizer = amp.initialize(model, optimizer, opt_level='O1', combine_grad=True)
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
  scaled_loss.backward()
...
```

## 四、接口清单

原生API及参数说明请参考`https://nvidia.github.io/apex/amp.html`， 这里仅对Apex-patch新增接口、新增参数进行说明。

### 4.1 apex.amp

> 4.1.1 apex.amp.initialize(models, optimizers=None, enabled=True, opt_level="O1", cast_model_type=None, patch_torch_functions=None, keep_batchnorm_fp32=None, master_weights=None, loss_scale=None, cast_model_outputs=None, num_losses=1, verbosity=1, dynamic_init_scale=2.**16, scale_growth_factor=2., scale_backoff_factor=0.5, scale_window=2000, min_loss_scale=None, max_loss_scale=2.**24, combine_grad=None, combine_ddp=None, ddp_replica_count=4, user_cast_preferred=None, check_combined_tensors=None)

接口说明：

根据选择的opt_level等配置初始化模型、优化器，也可开启融合梯度优化、融合数据并行优化等。amp.initialize 应在构建完模型和优化器后调用，但应在通过任何 DistributedDataParallel 装饰器装饰模型之前调用。目前，amp.initialize 只应调用一次，尽管它可以处理任意数量的模型和优化器。

新增参数说明：

- dynamic_init_scale - 动态loss scale初始值（默认2**16）
- scale_growth_factor - loss scale增长系数（默认2）
- scale_backoff_factor - loss scale回退系数（默认0.5）
- scale_window - loss scale窗口（默认2000）
- combine_grad - 梯度融合开关 （默认None）
- combine_ddp - 融合分布式数据并行 （默认None）
- ddp_replica_count - DDP融合梯度副本数（默认4）
- user_cast_preferred - O1模式下优先选择用户注册的精度模式（默认None）
- check_combined_tensors - tensor融合检查（默认None）

约束条件：

启用融合功能（combine_grad/combine_ddp）后，在创建融合张量时会申请融合后张量大小的内存，device内存不足时不建议使用。融合张量内存与原张量共享内存，若更改其一的内存地址，将破坏共享内存机制，可以引起精度异常等问题，使用时须用户自行保证共享内存不被破坏。

示例：
```
model, optim = apex.amp.initialize(model, optim, opt_level="O3", keep_batchnorm_fp32=True, ddp_replica_count=8)
```

> 4.1.2 apex.amp.scale_loss(loss, optimizers, loss_id=0, model=None, delay_unscale=False, delay_overflow_check=False)

接口说明：

使用混合精度时对loss进行scale，避免在低精度模式下梯度溢出。

API及参数说明请参考`https://nvidia.github.io/apex/amp.html`，Apex-patch中修改了接口内部实现，以保证在NPU上功能正常，在开启融合功能时使用融合张量进行scale/unscale以提升训练效率。

### 4.2 apex.optimizers

接口说明：

融合优化器算法实现上等价于torch中的优化器实现，在梯度更新阶段利用Tensor融合技术，使用融合的梯度和参数进行更新，以提升部分场景下昇腾训练服务器上模型训练的效率。

示例：

将torch.optim.XXX替换为apex.optimizers.NpuFusedXXX，如：
```
opt = apex.optimizers.NpuFusedSGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
```

已支持的融合优化器及参数说明如下：

> 4.2.1 class apex.optimizers.NpuFusedSGD(params, lr=required， momentum=MOMENTUM_MIN, dampening=DAMPENING_DEFAULT, weight_decay=WEIGHT_DECAY_MIN, nesterov=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率
- momentum - 动量（默认值：0.0）
- dampening - 阻尼系数（默认值：0.0）
- weight_decay - 权重衰减（默认值：0.0）
- nesterov - 使用nesterov动量（默认值：False）

> 4.2.2 class NpuFusedAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- betas -  用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
- eps - 防止除0，提高数值稳定性 （默认值：1e-8）
- weight_decay - 权重衰减（默认值：0）
- amsgrad - 是否使用AMSGrad（默认值：False）

> 4.2.3 class NpuFusedAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- betas -  用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
- eps - 防止除0，提高数值稳定性 （默认值：1e-8）
- weight_decay - 权重衰减（默认值：0）
- amsgrad - 是否使用AMSGrad（默认值：False）

> 4.2.4 class NpuFusedAdamP(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- betas -  用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
- eps - 分母防除0项，提高数值稳定性 （默认值：1e-8）
- weight_decay - 权重衰减（默认值：0）
- delta - 余弦相似度阈值（默认值：0.1）
- wd_ratio - 权重衰减动态调整速率（默认值：0.1）
- nesterov - 使用nesterov动量（默认值：False）

> 4.2.5 class NpuFusedBertAdam(params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.99, e=1e-6, weight_decay=0.01, max_grad_norm=-1)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- warmup - t_total的warmup比例（默认值：-1，表示不进行warmup）
- t_total - 学习率调整的步数（默认值：-1，表示固定学习率）
- schedule - 学习率warmup策略（默认值：'warmup_linear'）
- b1 - Adams b1（默认值：0.9）
- b2 - Adams b2（默认值：0.99）
- e - Adams epsilon（默认值：1e-6）
- weight_decay - 权重衰减（默认值：0.01）
- max_grad_norm - 最大梯度正则（默认值：1.0，-1表示不做裁剪）

> 4.2.6 class NpuFusedAdadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- rho - 梯度的均方差系数（默认值：0.9）
- eps - 分母防除0项，提高数值稳定性（默认值：1e-6）
- weight_decay - 权重衰减（默认值：0）

> 4.2.7 class Lamb(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- betas -  用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
- eps - 分母防除0项，提高数值稳定性（默认值：1e-8）
- weight_decay - 权重衰减（默认值：0）
- adam - 将strust_ratio设置为1，退化为Adam（默认值：False）

> 4.2.8 class NpuFusedLamb(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False, use_global_grad_norm=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率。（默认值：1e-3）
- betas -  用于计算梯度及其平方的运行平均值的系数。 （默认值：（0.9，0.999））
- eps - 分母防除0项，提高数值稳定性（默认值：1e-8）
- weight_decay - 权重衰减（默认值：0）
- adam - 将strust_ratio设置为1，退化为Adam（默认值：False）
- use_global_grad_norm - 使用全局梯度正则（默认值：False）

> 4.2.9 class NpuFusedRMSprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率。（默认值：1e-3）
- alpha - 平滑常量（默认值：0.99）
- eps - 分母防除0项，提高数值稳定性（默认值：1e-8）
- weight_decay - 权重衰减（默认值：0）
- momentum - 动量因子（默认值：0）
- centered - 计算中心RMSProp（默认值：False）

> 4.2.10 class NpuFusedRMSpropTF(params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False, decoupled_decay=False, lr_in_momentum=True)

参数说明：

- params - 模型参数或模型参数组
- lr - 学习率（默认值：1e-3）
- alpha - 平滑常量（默认值：0.9）
- eps - 分母防除0项，提高数值稳定性（默认值：1e-10）
- weight_decay - 权重衰减（默认值：0）
- momentum - 动量因子（默认值：0）
- centered -  计算中心RMSProp（默认值：False）
- decoupled_decay - 权重衰减仅作用于参数（默认值：False）
- lr_in_momentum - 计算动量buffer时使用lr（默认值：True）

融合优化器约束条件：

启用融合优化器（如apex.optimizers.NpuFusedSGD）后，在创建融合张量时会申请融合后张量大小的内存，device内存不足时不建议使用。融合张量内存与原张量共享内存，若更改其一的内存地址，将破坏共享内存机制，可以引起精度异常等问题，使用时须用户自行保证共享内存不被破坏。

### 4.3 其它

> 4.3.1 model.zero_grad(set_to_none: bool = False)

在torch>2.x版本后，该接口参数`set_to_none`默认值变化为True，由于融合功能要求模型梯度内存不能发生变化或释放，在开启融合功能时Apex-patch通过Monkey Patch的方式将该接口参数`set_to_none`强制设置为False。

> 4.3.2 apex.optimizers.NpuFusedXXX.clip_optimizer_grad_norm_fused(max_norm, norm_type=2)

在使用融合优化器时，该接口等价于`torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)`

### 4.4 C++侧接口

> 4.4.1 change_data_ptr(at::Tensor dst, at::Tensor src, int offset)

接口说明：

目标Tensor地址重指向源Tensor相对起始地址的某个偏移，用于实现Tensor融合时的Storage内存共享。该接口为Apex-patch内部使用接口，不对用户开放。

参数说明：

- dst - 用于地址重指的目标Tensor，重指后对应的Storage内存将被释放
- src - 用于地址重指的源Tensor，目标Tensor将被指向源Tensor的某个偏移
- offset - 目标Tensor指向源Tensor起始地址的偏移量


## 五、Apex-patch配套软件

| AscendPyTorch版本 | 支持PyTorch版本 | Pytorch Gitee分支名称 | Apex Gitee分支名称 |
| :---------------- | :--------------- | :--------------------- | :----------------- |
| 2.0.2             | 1.5.0.post2      | 2.0.2.tr5             | 2.0.2.tr5          |
| 2.0.3             | 1.5.0.post3      | 2.0.3.tr5             | 2.0.3.tr5          |
| 2.0.4             | 1.5.0.post4      | 2.0.4.tr5             | 2.0.4.tr5          |
| 3.0.rc1           | 1.5.0.post5      | v1.5.0-3.0.rc1        | v1.5.0-3.0.rc1     |
| 3.0.rc1           | 1.8.1.rc1        | v1.8.1-3.0.rc1        | v1.8.1-3.0.rc1     |
| 3.0.rc2           | 1.5.0.post6      | v1.5.0-3.0.rc2        | v1.5.0-3.0.rc2     |
| 3.0.rc2           | 1.8.1.rc2        | v1.8.1-3.0.rc2        | v1.8.1-3.0.rc2     |
| 3.0.rc3           | 1.5.0.post7      | v1.5.0-3.0.rc3        | v1.5.0-3.0.rc3     |
| 3.0.rc3           | 1.8.1.rc3        | v1.8.1-3.0.rc3        | v1.8.1-3.0.rc3     |
| 3.0.rc3           | 1.11.0.rc1 (beta)| v1.11.0-3.0.rc3       | v1.11.0-3.0.rc3    |
| 3.0.0             | 1.5.0.post8      | v1.5.0-3.0.0          | v1.5.0-3.0.0       |
| 3.0.0             | 1.8.1            | v1.8.1-3.0.0          | v1.8.1-3.0.0       |
| 3.0.0             | 1.11.0.rc2 (beta)| v1.11.0-3.0.0         | v1.11.0-3.0.0      |
| 5.0.rc1           | 1.8.1.post1, 1.11.0  | v1.8.1-5.0.rc1, v1.11.0-5.0.rc1 | 5.0.rc1     |
| 5.0.rc2           | 1.8.1.post2, 1.11.0, 2.0.1.rc1  | v1.8.1-5.0.rc2, v1.11.0-5.0.rc2 | 5.0.rc2 |
| 5.0.rc3           | 1.11.0, 2.0.1.rc1  | v1.11.1-5.0.rc3, v2.0.1-5.0.rc3 | 5.0.rc3 |

## 六、附录

##### 6.1 公网地址

|      类型      |          文件名          |             公网IP地址/公网URL地址/域名/邮箱地址             |          用途说明           |
| :------------: | :----------------------: | :----------------------------------------------------------: | :-------------------------: |
|  开源代码  |      scripts/build.sh       |          https://github.com/NVIDIA/apex.git          | 构建脚本中拉取原生Apex源码，结合Ascend Apex patch构建安装包 |

##### 6.2 通信矩阵

本插件不涉及端口开放、监听等相关行为，相关端口行为由用户在模型脚本调用Pytorch原生接口开启，具体通信矩阵可参考 [torch_npu](https://gitee.com/ascend/pytorch/)，建议用户注意做好安全防护。

##### 6.3 资源使用

建议您根据自身运行环境资源状况，进行训练配置的设定与数据集的准备，若与资源状况不匹配，比如数据集的size超出内存容量/NPU存储容量等，那么Pytorch及相关三方库进程会直接退出，并自动释放占用的资源。
