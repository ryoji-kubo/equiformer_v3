MAIN_PATH="my_main.py"
LOG_DIR="/home/ryoji/equiformer_v3/models/oc20/equiformer_v3/200k"
CONFIG_PATH="experimental/configs/oc20/200k/equiformer_v3/experiments/base_N@8-L@6-C@128-attn-hidden@64-ffn@512-envelope-num-rbf@128_merge-layer-norm_gates2-gridmlp_use-gate-force-head_wd@1e-3-grad-clip@100_lin-ref-e@4.yml"
IDENTIFIER="base"


python -u -m torch.distributed.launch --nproc_per_node=4 $MAIN_PATH \
    --num-gpus 4 \
    --amp \
    --mode train \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --seed 1 \
    --identifier $IDENTIFIER \
    --optim.eval_every=10 \
    --optim.batch_size=8 \
    --optim.eval_batch_size=16 \
    --optim.grad_accumulation_steps=1