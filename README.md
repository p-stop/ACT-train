# ACT-train

Data analysis, data augmentation and training scripts for an ACT policy (LeRobot **v0.6.1**, not included — install it with
`pip install "lerobot[training]==0.6.1"`) on a dual-arm Piper robot.

Task: the right arm puts an orange sponge into the right plate, then the left arm puts a wooden block into the left plate
(dataset `<HF_USER>/01_ACT`: 52 demos, 30 fps, 14-D joints + grippers, cameras `left_wrist`, `right_wrist`, `scene_fixed`).

## What is here

| Directory | Contents |
|---|---|
| `analysis/` | Diagnosis of the trained ACT: per-episode timelines and grasp poses (`stats.py`), key-frame montages (`frames.py`), object positions by colour segmentation (`seg.py`), counterfactual image edits (`cf.py`) and input-swap tests (`swap.py`, `swap2.py`). Finding: the left arm identified the episode from the right half of the scene image instead of looking at the block; the right arm localised the coaster rather than the sponge. |
| `aug01/` | Augmented datasets built from the original one. `analyze.py` (arm-swept regions per phase), `clean_check.py`, `overlay.py`, `build.py mixr` (replace the parked right arm / plate / empty coaster region with another episode's), `build.py fused` (play the left-arm and right-arm phases at the same time, split by a dividing line, with per-frame arm-distance collision screening), `preview.py`, `riskframes.py`, `push_hf.py` (private HF upload with the `v3.0` tag), `rebuild_on_pro5000.sh`. |
| `dataset_preview/` | Browser-based dataset viewer / checker (stdlib + numpy + pyarrow): synchronized videos, joint curves, episode marks, **dividing-line labelling** on the scene camera, and **collision-risk flags** for fused datasets. |
| `pro5000/` | Training launcher `train_act.sh` (models `ACT_bi`, `ACT_mix`, `ACT_fmix`), `episode_lists.json`, pinned dependency list `constraints_from_wxr5.txt`, and the environment notes `环境配置说明.md`. |
| `verify/` | Cross-machine consistency checks (frame data digest, decoded-frame thumbnails). |

## Pipeline

```bash
A=<PRO5000_ACT_DIR>            # working dir with .venv/, datasets/01_ACT/, tools/
PY=$A/.venv/bin/python
$PY aug01/analyze.py     $A/datasets/01_ACT $A/datasets/aug_work_01/analysis
$PY aug01/clean_check.py $A/datasets/aug_work_01/analysis --write
$PY aug01/build.py mixr  --src $A/datasets/01_ACT --analysis $A/datasets/aug_work_01/analysis --out $A/datasets/01_ACT_mixR  --repo_id <HF_USER>/01_ACT_mixR
$PY aug01/build.py fused --pairing same --keep_all [--seams <labels.json>] \
                         --src $A/datasets/01_ACT --analysis $A/datasets/aug_work_01/analysis --out $A/datasets/01_ACT_fused --repo_id <HF_USER>/01_ACT_fused
ACT_DIR=$A HF_USER=<HF_USER> bash pro5000/train_act.sh ACT_bi 0    # or ACT_mix / ACT_fmix
```

`train_act.sh` expects `episode_lists.json` in `$ACT_DIR/tools/`. The preview tool is started with
`bash dataset_preview/dataset_preview.sh` from a project directory laid out like `<ROBOT_PROJECT_DIR>` (see its README).

## Placeholders to replace

| Placeholder | Meaning |
|---|---|
| `<HF_USER>` | Hugging Face user that owns the datasets |
| `<PRO5000_ACT_DIR>` | ACT working directory on the training server |
| `<ROBOT_PROJECT_DIR>`, `<ROBOT_IP>` | project directory and LAN address of the robot machine (for the preview tool) |
| `<WORK_DIR>` | a local working directory used in the analysis notes |

The scripts that need them read them from environment variables (`ACT_DIR`, `HF_USER`) or command-line arguments.
