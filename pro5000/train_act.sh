#!/bin/bash
# Train one ACT model on pro5000.
#   usage: tools/train_act.sh <ACT_bi|ACT_mix|ACT_fmix> <gpu index>
# Hyper-parameters are the same as the earlier act_01_ACT_300k run on wxr5 (only the data differ).
set -e
NAME=$1; GPU=$2
A="${ACT_DIR:-<PRO5000_ACT_DIR>}"   # ACT working directory (datasets/, tools/, outputs/, .venv)
HF_USER="${HF_USER:-<HF_USER>}"      # only used as the dataset name; data are read from $A/datasets
case $NAME in
  ACT_bi)   DS=01_ACT_fused; EPS=$($A/.venv/bin/python -c "import json;print(json.load(open(\"$A/tools/episode_lists.json\"))[\"ACT_bi\"])") ;;
  ACT_mix)  DS=01_ACT_mixR;  EPS=$($A/.venv/bin/python -c "import json;print(json.load(open(\"$A/tools/episode_lists.json\"))[\"ACT_mix\"])") ;;
  ACT_fmix) DS=01_ACT_mixR;  EPS="" ;;
  *) echo "unknown model $NAME"; exit 1 ;;
esac
export HF_HOME=$A/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DIR=$A/outputs/wandb_dirs/$NAME
mkdir -p $A/outputs/train $A/outputs/logs $WANDB_DIR
cd $A
ARGS=(--dataset.repo_id=$HF_USER/$DS --dataset.root=$A/datasets/$DS
      --policy.type=act --policy.device=cuda --policy.push_to_hub=false
      --output_dir=$A/outputs/train/$NAME --job_name=$NAME
      --steps=300000 --batch_size=8 --num_workers=8 --save_freq=20000 --log_freq=200 --seed=1000
      --wandb.enable=true --wandb.mode=offline)
[ -n "$EPS" ] && ARGS+=(--dataset.episodes="${EPS// /}")
echo "[$(date "+%F %T")] $NAME on GPU $GPU: ${ARGS[*]}"
CUDA_VISIBLE_DEVICES=$GPU exec $A/.venv/bin/lerobot-train "${ARGS[@]}"
