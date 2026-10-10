#!/usr/bin/env bash
# E001 run: $1 = tag (p1_bf16 | p3_int4 | p2_int8), rest = extra args. [ai-scaffold]
set -x
source "$(dirname "$0")/env.sh"
source "$CONDA_SH" && conda activate oft
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
export PYTHONPATH=$LIBERO_PLUS_ROOT MUJOCO_GL=egl ORIG_TASK_IDS=0 E001_METRICS=$LOG_DIR/e001-$1-metrics.json
tag=$1; shift
cd $OPENVLA_OFT_ROOT
nvidia-smi --query-gpu=memory.used --format=csv
( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > $LOG_DIR/e001-$tag-nvsmi.txt &
MON=$!
/usr/bin/time -v python $REPO_ROOT/scripts/run_oft_original.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial --num_trials_per_task 3 \
  --local_log_dir $LOG_DIR/e001-$tag-rollouts "$@"
kill $MON
echo "NVSMI_PEAK_MIB $(sort -n $LOG_DIR/e001-$tag-nvsmi.txt | tail -1)"
echo RUN_DONE
