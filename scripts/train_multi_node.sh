#!/usr/bin/env bash
# 多机多卡训练 MixFormer (AWS p5.48xlarge × N + EFA).
#
# 在每个节点上分别执行此脚本; 通过 NODE_RANK 区分主从.
#
# 用法 (双机为例):
#   节点 0 (master, 私有 IP=172.31.8.251):
#     MASTER_ADDR=172.31.8.251 NODE_RANK=0 NNODES=2 bash scripts/train_multi_node.sh
#   节点 1 (worker):
#     MASTER_ADDR=172.31.8.251 NODE_RANK=1 NNODES=2 bash scripts/train_multi_node.sh
#
# 必填环境变量:
#   MASTER_ADDR : rank-0 节点的私有 IP
#   NODE_RANK   : 当前节点编号, [0, NNODES-1]
#   NNODES      : 节点总数
#
# 可选环境变量:
#   NPROC_PER_NODE : 每节点 GPU 数 (默认 8)
#   CONFIG         : default | medium (默认 default)
#   DATA_DIR       : 预处理后的数据目录 (默认 data/alibaba_real)
#   BATCH_SIZE     : 每卡 batch (默认 4096)
#   EPOCHS         : 训练轮数 (默认 1)
#   AMP            : 1=开启混合精度 (默认 1)
#   NUM_WORKERS    : DataLoader workers (默认 4; GDR=off 时建议 0 防 fork ENOMEM)
#   SAVE_DIR       : 检查点保存目录 (默认 checkpoints_multi_${CONFIG}_${WORLD_SIZE}gpu)
#   MASTER_PORT    : torchrun 端口 (默认 29500)
#   GDR_MODE       : on | off (默认 on; off 模拟客户 GDR 失效场景)
#                    - on : 走 GPUDirect RDMA (dmabuf), 期望性能
#                    - off: NCCL_NET_GDR_LEVEL=0, 退到 host bounce buffer (慢 2x+)
#   IFNAME         : NCCL bootstrap 走的网卡名 (默认 enp71s0; AWS p5 上是 ENA 主网卡)
set -euo pipefail

: "${MASTER_ADDR:?需要设置 MASTER_ADDR=<rank-0 私有 IP>}"
: "${NODE_RANK:?需要设置 NODE_RANK}"
: "${NNODES:?需要设置 NNODES}"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CONFIG=${CONFIG:-default}
DATA_DIR=${DATA_DIR:-data/alibaba_real}
BATCH_SIZE=${BATCH_SIZE:-4096}
EPOCHS=${EPOCHS:-1}
AMP=${AMP:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
GDR_MODE=${GDR_MODE:-on}
MASTER_PORT=${MASTER_PORT:-29500}
IFNAME=${IFNAME:-enp71s0}

WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
SAVE_DIR=${SAVE_DIR:-checkpoints_multi_${CONFIG}_${WORLD_SIZE}gpu_gdr${GDR_MODE}}

# ---------- EFA / libfabric ----------
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_HUGE_PAGE=1

# ---------- NCCL ----------
export NCCL_SOCKET_IFNAME="${IFNAME}"
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}
# aws-ofi-nccl 插件 (容器内默认装在 /opt/aws-ofi-nccl/lib/)
[[ -f /opt/aws-ofi-nccl/lib/libnccl-net.so ]] && \
    export NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net.so

if [[ "$GDR_MODE" == "on" ]]; then
    # GPUDirect RDMA (dmabuf) — 期望路径
    unset FI_HMEM_CUDA_USE_GDRCOPY            # 让 libfabric 自动用 gdrcopy
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
    export NCCL_NET_GDR_READ=1
    echo "[train_multi_node] GDR ON   (NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL)"
elif [[ "$GDR_MODE" == "off" ]]; then
    # 模拟客户故障场景: 关闭整条 GDR 数据路径
    export FI_HMEM_CUDA_USE_GDRCOPY=0
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_NET_GDR_READ=0
    # GDR=off 时 NCCL 占用 host 内存大, fork DataLoader workers 容易 ENOMEM
    NUM_WORKERS=0
    echo "[train_multi_node] GDR OFF  (强制 host bounce buffer; num_workers 降为 0)"
else
    echo "GDR_MODE 必须是 on 或 off, 实际: $GDR_MODE" >&2
    exit 1
fi

AMP_FLAG=""
[[ "$AMP" == "1" ]] && AMP_FLAG="--amp"

echo "[train_multi_node] NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "[train_multi_node] MASTER=$MASTER_ADDR:$MASTER_PORT IFNAME=$IFNAME"
echo "[train_multi_node] CONFIG=$CONFIG BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS AMP=$AMP NUM_WORKERS=$NUM_WORKERS"
echo "[train_multi_node] DATA_DIR=$DATA_DIR SAVE_DIR=$SAVE_DIR"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train.py \
    --data_dir "${DATA_DIR}" \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_dir "${SAVE_DIR}" \
    ${AMP_FLAG}
