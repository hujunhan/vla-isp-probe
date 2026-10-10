#!/usr/bin/env bash
# E002 stage runner: $1 = suite (libero_spatial|libero_object|libero_goal|libero_10), $2 = csv tag.
# Env passthrough: CATEGORIES, TASK_IDS, LANG_MODE. [ai-scaffold]
set -x
source "$(dirname "$0")/env.sh"
source "$CONDA_SH" && conda activate oft
export PYTHONPATH=$LIBERO_PLUS_ROOT MUJOCO_GL=egl
suite=$1; tag=$2; ck=${suite#libero_}
export OUT_CSV=$REPO_ROOT/results/${RESULTS_DIR:-e002}/$tag.csv
cd $OPENVLA_OFT_ROOT
python $REPO_ROOT/scripts/run_oft_libero_plus.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-$ck \
  --task_suite_name $suite --num_trials_per_task 1 --load_in_4bit True \
  --local_log_dir $LOG_DIR/e002-$tag
echo "STAGE_EXIT $?"
