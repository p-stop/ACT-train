---
license: mit
task_categories: [robotics]
tags: [LeRobot]
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# 01_ACT_fused

Synthetic simultaneous bimanual demonstrations built from `<HF_USER>/01_ACT` (52 sequential demos: right arm puts the orange
sponge into the right plate, then the left arm puts the wooden block into the left plate).

Each output episode i plays the **left-arm phase** of source episode i and the **right-arm phase** of the same episode at the
same time. Joint state / action and the wrist cameras come from their own phase; the scene image takes everything left of a
dividing line from the left-arm phase and everything right of it from the right-arm phase (automatic line per episode).

- `meta/aug_manifest.json`: for every output episode the source episodes, the dividing line, and a collision screening
  (minimum distance between the two arms in the scene image, `arm_min_dist_px`, at `arm_min_dist_t` seconds).
- Episodes marked unusable after visual review (likely arm collision): **7, 10, 11, 15, 25, 44, 45**.
  Model `ACT_bi` is trained on the remaining 45 episodes.

Generated with `tools/aug01/build.py fused --pairing same --keep_all`.
