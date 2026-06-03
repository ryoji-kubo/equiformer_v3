CHECKPOINT_PATH="/home/ryoji/equivarient/equiformer_v3/models/omat24/equiformer_v3/checkpoints/2026-05-28-13-37-04-mptrj_grad_160k/best_checkpoint_no-torch-compile.pt"
OUTPUT_DIR="/home/ryoji/equivarient/equiformer_v3/results/matbench_discovery/discovery_results/equiformer_v3/2026-05-28-13-37-04-mptrj_grad_160k/all"
DATA_PATH="/home/ryoji/equivarient/equiformer_v3/dataset/matbench_discovery/WBM_IS2RE.aselmdb"

for i in $(seq 0 15); do
    device=$((i % 4))
    CUDA_VISIBLE_DEVICES=$device python experimental/tasks/matbench_discovery/test_discovery.py \
        --checkpoint-path $CHECKPOINT_PATH \
        --output-path $OUTPUT_DIR \
        --data-path $DATA_PATH \
        --num-jobs 16 \
        --job-index $i \
        &
done
# pkill -f "python.*test_discovery.py"