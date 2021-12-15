# Ascend apex


## Full Code Generation and Compilation

Note: The root directory in the following description refers to the root directory of Ascend apex.

**Obtain the native apex source code.**

Obtain the source code from GitHub and run the following command in the root directory:
```
git clone https://github.com/NVIDIA/apex.git
```
Go to the source code directory and switch to the branch whose **commitid** is **4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a**.
```
cd apex
git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
cd ..
```

**Generate the apex code adapted to Ascend AI Processors.**

Go to the **scripts** directory and run the following command:
```
bash gen.sh
```
The full code adapted to NPUs is generated in the **apex** directory under the root directory.

**Compile the binary package of apex.**

1. Ensure that PyTorch of the NPU version can be properly used. Otherwise, the apex compilation will be affected.

2. Go to the **apex** directory under the root directory and run the following command:
```
python3 setup.py --cpp_ext --npu_float_status bdist_wheel
```
The generated binary package is stored in the current **dist** directory.


## Installation

Go to the **dist** directory and run the following command:
```
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-cp37-cp37m-linux_{arch}.whl *arch* indicates the architecture, which can be AArch64 or x86_64.
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

