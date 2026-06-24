CHECKPOINT_PATH="/home/ryoji/equivarient/equiformer_v3/logs/omat24/equiformer_v3/checkpoints/2026-05-28-13-37-04-mptrj_grad_160k/best_checkpoint_no-torch-compile.pt"
OUTPUT_DIR="/home/ryoji/equivarient/equiformer_v3/results/matbench_discovery/discovery_results/equiformer_v3/2026-05-28-13-37-04-mptrj_grad_160k/all/kappa_2"

python experimental/tasks/matbench_discovery/kappa_run_single_relaxation.py --checkpoint-path $CHECKPOINT_PATH --output-dir $OUTPUT_DIR