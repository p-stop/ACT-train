"""Digest of a lerobot dataset's frame data (states, actions, episode index) for cross-machine comparison."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
root = Path(sys.argv[1])
d = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("chunk-*/file-*.parquet"))])
h = hashlib.md5()
for c in ("observation.state", "action"):
    h.update(np.stack(d[c].values).astype(np.float32).tobytes())
h.update(d["episode_index"].values.astype(np.int64).tobytes())
info = json.loads((root / "meta/info.json").read_text())
print(root.name, "episodes", info["total_episodes"], "frames", info["total_frames"], "data md5", h.hexdigest())
