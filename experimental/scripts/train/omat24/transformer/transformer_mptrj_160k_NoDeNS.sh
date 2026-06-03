#!/usr/bin/env bash
set -euo pipefail

NNODES=1
NPROC_PER_NODE=4
RDZV_PORT=29500

REMOTE_PREFIX=''

###########################
#       Script
###########################

MAIN_PATH="my_main.py"

LOG_DIR="/home/ryoji/equivarient/equiformer_v3/logs/omat24/transformer"

CONFIG_PATH="experimental/configs/omat24/mptrj/experiments/direct/transformer_160k_NoDeNS.yml"
IDENTIFIER="transformer_mptrj_direct_160k_NoDeNS"

PROJECT="equiformer_v3_mptrj"


REMOTE_SCRIPT="$MAIN_PATH \
    --num-gpus ${NPROC_PER_NODE} \
    --num-nodes ${NNODES} \
    --mode train \
    --amp \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --seed 1 \
    --identifier $IDENTIFIER \
    --optim.num_workers=0 \
"

echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "SCRIPT: $REMOTE_SCRIPT"
echo

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv-id=10 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${RDZV_PORT} \
    ${REMOTE_SCRIPT}