MAIN_PATH="my_main.py"
LOG_DIR="models/oc20/equiformer_v2/200k"
CONFIG_PATH="experimental/configs/oc20/200k/equiformer_v2/experiments/eqV2_57M_inf.yml"
IDENTIFIER="base"


python -u -m torch.distributed.launch --nproc_per_node=4 $MAIN_PATH \
    --num-gpus 4 \
    --amp \
    --mode validate \
    --checkpoint checkpoints/oc20/equiformer_v2/200k/eqv2_57M_200k.pt \
    --config-yml $CONFIG_PATH \
    --run-dir $LOG_DIR \
    --print-every 200 \
    --seed 1 \
    --identifier $IDENTIFIER \
    --optim.eval_every=5000 \
    --optim.batch_size=8 \
    --optim.eval_batch_size=16 \
    --optim.grad_accumulation_steps=1