#!/bin/bash
# 用法：bash run_experiments.sh <train|test> <实验名>
#
# 实验名：
#   baseline  无任何改进
#   A_only    仅ArcFace
#   B_only    仅小波
#   C_only    仅傅里叶
#   AB        ArcFace + 小波
#   AC        ArcFace + 傅里叶
#   BC        小波 + 傅里叶
#   ABC       三个全开

MODE=$1
EXP=$2

case $EXP in
    baseline) A=False; B=False; C=False ;;
    A_only)   A=True;  B=False; C=False ;;
    B_only)   A=False; B=True;  C=False ;;
    C_only)   A=False; B=False; C=True  ;;
    AB)       A=True;  B=True;  C=False ;;
    AC)       A=True;  B=False; C=True  ;;
    BC)       A=False; B=True;  C=True  ;;
    ABC)      A=True;  B=True;  C=True  ;;
    *) echo "未知实验名: $EXP"; exit 1 ;;
esac

echo "实验: $EXP | ArcFace=$A | Wavelet=$B | Fourier=$C"

ARGS="--use_arcface $A --use_wavelet $B --use_fourier $C"

export PYTHONPATH=$PYTHONPATH:$(pwd)

if [ "$MODE" = "train" ]; then
    CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1 \
    torchrun --nproc_per_node=1 --master_port=4326 \
        scripts/T2V_train.py $ARGS

elif [ "$MODE" = "test" ]; then
    CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1 \
    torchrun --nproc_per_node=1 --master_port=4327 \
        scripts/T2V_test.py $ARGS

else
    echo "用法: bash run_experiments.sh <train|test> <实验名>"
fi