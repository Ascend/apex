### Build apex from Docker container

**Clone apex**

```Shell
git clone -b master https://gitee.com/ascend/apex.git
```

**Build docker image**

```Shell
cd apex/scripts/docker/{arch} # {arch} for X86 or ARM
docker build -t manylinux-builder:v1 .
```
**Enter docker Container**

```Shell
docker run -it -v /{code_path}/apex:/home/apex manylinux-builder:v1 bash
# {code_path} is the apex source code path
```
**Install torch**

Take Python 3.8 as an example
```Shell
pip3.8 install torch
# Reference url: https://pytorch.org/get-started/previous-versions/
```
**Compile apex**
```Shell
cd /home/apex
bash scripts/build.sh --python=3.8
```
