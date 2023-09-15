# Ascend apex


## Full Code Generation and Compilation


**Obtain the apex source code.**

Run the following command in the root directory:
```
git clone -b master https://gitee.com/ascend/apex.git
cd apex/
```

**Compile the binary package of apex.**

1. Ensure that torch is installed and the version of setuptools is less than or equal to 65.7.0 (otherwise run: pip install setuptools==41.2.0).

2. Run the following command(python3.7-3.10 is supported):
```
bash scripts/build.sh --python=3.7
```
The generated binary package is stored in the current **dist** directory.


## Installation

Run the following command:
```
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl *version* indicates the python version and cpu architecture.
```


## Features
**Supported features:**

- [x] O1 mode
- [x] O2 mode
- [x] Static loss scale
- [x] Dynamic loss scale
- [x] combine tensors
- [x] combine grad for unscale
- [x] npu fused optimizer: adadelta, adam, adamp, adamw, sgd, lamb, rmsprop, rmsprop_tf
- [x] Adjustable parameters such as **dynamic_init_scale**, **scale_growth_factor**, **scale_backoff_factor**, and **scale_window** are added for dynamic loss scale.

**Note:**

In the current version, apex is implemented using Python and does not support AscendCL or CUDA optimization.


## Method of Use
**Mixed precision:**

For details, see https://nvidia.github.io/apex/amp.html.

**combine grad for unscale: **

In **amp.initialize()**, set **combine_grad** to **True**.

**npu fused optimizer: **

Replace the original optimizer with **apex.optimizers.xxx**, where *xxx* indicates the name of the fusion optimizer.

