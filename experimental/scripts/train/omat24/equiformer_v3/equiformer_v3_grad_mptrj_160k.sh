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

LOG_DIR="models/omat24/equiformer_v3"

CONFIG_PATH="experimental/configs/omat24/mptrj/experiments/gradient/equiformer_v3_160k.yml"
IDENTIFIER="mptrj_grad_160k"

PROJECT="equiformer_v3_mptrj"

# REMOVED --amp so that we can run gradient fine-tuning in FP32

REMOTE_SCRIPT="$MAIN_PATH \
    --num-gpus ${NPROC_PER_NODE} \
    --num-nodes ${NNODES} \
    --mode train \
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