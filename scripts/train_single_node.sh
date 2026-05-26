#!/usr/bin/env bash
# 单机多卡训练 MixFormer (Alibaba UserBehavior 数据集).
#
# 用法:
#   bash scripts/train_single_node.sh                       # 用默认参数 (8 GPU, config=default)
#   NPROC=4 CONFIG=medium bash scripts/train_single_node.sh # 用 4 卡跑 medium 配置
#
# 环境变量 (均可覆盖):
#   NPROC            : 单机使用的 GPU 数 (默认 8)
#   CONFIG           : default | medium (默认 default; 论文等比缩水版本)
#   DATA_DIR         : 预处理后的数据目录 (默认 data/alibaba_real)
#   BATCH_SIZE       : 每卡 batch (默认 4096)
#   EPOCHS           : 训练轮数 (默认 1; CTR 通常 1 epoch 即可)
#   AMP              : 1=开启混合精度 (默认 1)
#   NUM_WORKERS      : DataLoader workers (默认 4; 显存紧张时改 0)
#   SAVE_DIR         : 检查点保存目录 (默认 checkpoints_single_${CONFIG}_${NPROC}gpu)
#   MASTER_PORT      : torchrun 监听端口 (默认 29500)
set -euo pipefail

NPROC=${NPROC:-8}
CONFIG=${CONFIG:-default}
DATA_DIR=${DATA_DIR:-data/alibaba_real}
BATCH_SIZE=${BATCH_SIZE:-4096}
EPOCHS=${EPOCHS:-1}
AMP=${AMP:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_DIR=${SAVE_DIR:-checkpoints_single_${CONFIG}_${NPROC}gpu}
MASTER_PORT=${MASTER_PORT:-29500}

AMP_FLAG=""
[[ "$AMP" == "1" ]] && AMP_FLAG="--amp"

echo "[train_single_node] NPROC=$NPROC CONFIG=$CONFIG BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS AMP=$AMP"
echo "[train_single_node] DATA_DIR=$DATA_DIR SAVE_DIR=$SAVE_DIR"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    --data_dir "${DATA_DIR}" \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_dir "${SAVE_DIR}" \
    ${AMP_FLAG}
