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

Take Python 3.8 and torch 2.1.0 as an example
```Shell
pip3.8 install torch==2.1.0
# Reference url: https://pytorch.org/get-started/previous-versions/
# Please modify the installation commands based on the torch version
```
**Compile apex**
```Shell
cd /home/apex
bash scripts/build.sh --python=3.8
```
