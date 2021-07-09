#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..
APEX_DIR=$ROOT_DIR/apex

function main()
{
    cd $ROOT_DIR || exit

    # patch
    cp $ROOT_DIR/patch/npu.patch $APEX_DIR
    cd $APEX_DIR || exit
    patch -p1 <npu.patch
    rm -rf npu.patch
    cp -r $ROOT_DIR/src/* $APEX_DIR
    cp -r $ROOT_DIR/tests/* $APEX_DIR/tests
}

main "$@"

