---
license: mit
task_categories: [robotics]
tags: [LeRobot]
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# 01_ACT_mixR

`<HF_USER>/01_ACT` (52 demos) plus 2 augmented copies of each (156 episodes in total).

In the augmented copies, from the moment the right arm has parked, the region of the parked right arm, the blue plate with the
sponge and the empty coaster is replaced by the same region from another episode's parked frame (the left arm's own pixels are
kept wherever it passes). Joint data and wrist cameras are unchanged. This breaks the shortcut where the left arm identified
the episode from the right side of the scene instead of looking at the wooden block.

`meta/aug_manifest.json` lists for every output episode whether it is an original (`orig`) or a copy (`mixr`, with `src` and `donor`).

Generated with `tools/aug01/build.py mixr`.
