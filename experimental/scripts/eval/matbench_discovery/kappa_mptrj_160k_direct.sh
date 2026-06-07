CHECKPOINT_PATH="/home/ryoji/equiformer_v3/models/omat24/equiformer_v3/checkpoints/2026-05-26-13-52-00-mptrj_direct_160k/2026-05-26-13-52-00-mptrj_direct_160k/best_checkpoint_no-torch-compile.pt"
OUTPUT_DIR="/home/ryoji/equiformer_v3/results/matbench_discovery/discovery_results/equiformer_v3/2026-05-26-13-52-00-mptrj_direct_160k_eval2/all/kappa_2"

python experimental/tasks/matbench_discovery/kappa_run_single_relaxation.py --checkpoint-path $CHECKPOINT_PATH --output-dir $OUTPUT_DIR