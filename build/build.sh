#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..
APEX_DIR=$ROOT_DIR/apex-npu
APEX_PKG=apex.tar.gz

function main()
{
    cd $ROOT_DIR || exit
    # get ori apex
    if [ -f $ROOT_DIR/$APEX_PKG ];then
        echo "detect $APEX_PKG exist, skip download"
    else
        wget https://ascend-ptadapter.obs.cn-north-4.myhuaweicloud.com/pytorch-v1.5.0/$APEX_PKG --no-check-certificate
    fi

    if [ $? != 0 ]; then
        echo "Failed to wget source code of apex, check network."
        exit 1
    fi

    # mkdir apex
    if [ -d $APEX_DIR ];then
        echo "$APEX_DIR exists, if nothing to backup, please remove it"
        exit 1
    fi

    # unpack
    tar -xf $APEX_PKG
    mv apex $APEX_DIR

    echo "download and unpack $APEX_PKG success"

    # patch
    cp $ROOT_DIR/patch/npu.patch $APEX_DIR
    cd $APEX_DIR || exit
    patch -p1 <npu.patch
    rm -rf npu.patch
    cp -r $ROOT_DIR/src/* $APEX_DIR
    cp -r $ROOT_DIR/tests/* $APEX_DIR/tests
    
    if [[ $1 = "gen" ]];then
        exit 0
    fi
    # build
    python3 setup.py --cpp_ext --npu_float_status bdist_wheel
    cp $APEX_DIR/dist/apex-* $ROOT_DIR/dist
}

main "$@"

