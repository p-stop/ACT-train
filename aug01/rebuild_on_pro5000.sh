#!/bin/bash
# Rebuild 01_ACT_fused / 01_ACT_mixR on pro5000 from the original 01_ACT with the same scripts and seeds as on bimanual.
set -e
A="${ACT_DIR:-<PRO5000_ACT_DIR>}"; PY=$A/.venv/bin/python; S=$A/datasets/01_ACT; W=$A/datasets/aug_work_01
cd $A/tools/aug01; mkdir -p $W
$PY analyze.py $S $W/analysis > $W/analyze.log 2>&1
$PY clean_check.py $W/analysis --write > $W/clean.log 2>&1
$PY build.py fused --pairing same --keep_all --src $S --analysis $W/analysis --out $A/datasets/01_ACT_fused --repo_id <HF_USER>/01_ACT_fused > $W/fused_same.log 2>&1 &
$PY build.py mixr --src $S --analysis $W/analysis --out $A/datasets/01_ACT_mixR --repo_id <HF_USER>/01_ACT_mixR > $W/mixr.log 2>&1 &
wait
echo ALL_DONE >> $W/clean.log
