"""Inspect / clean the swept masks: keep only big connected blobs, report automatic-line margins per episode."""
import sys
from pathlib import Path
import cv2, numpy as np
from common import best_line, profile
d = Path(sys.argv[1]); write = len(sys.argv) > 2 and sys.argv[2] == "--write"


def big(m, frac=0.03, minpx=800):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    tot = st[1:, 4].sum() if n > 1 else 0
    keep = [i for i in range(1, n) if st[i, 4] >= max(minpx, frac * tot)]
    return np.isin(lab, keep)


res = []
for f in sorted(d.glob("ep_*.npz")):
    z = dict(np.load(f)); ep = int(f.stem[3:])
    raw = (z["swL"].sum(), z["swR"].sum(), z["parkR"].sum())
    for k in ("swL", "swR", "parkR"):
        z[k] = big(z[k])
    A = profile(*[z[k] for k in ("swL", "obj_wood", "obj_green")])
    B = profile(*[z[k] for k in ("swR", "parkR", "obj_sponge", "obj_blue")])
    m, line = best_line(A, B)
    res.append(m)
    print(f"ep {ep:2d}  kept swL {z['swL'].sum() / max(raw[0], 1):.2f} swR {z['swR'].sum() / max(raw[1], 1):.2f} parkR {z['parkR'].sum() / max(raw[2], 1):.2f}   auto line {line} margin {m:6.1f}", flush=True)
    if write:
        np.savez_compressed(f, **z)
print("margin: min %.1f  median %.1f  <0: %d  <8: %d" % (min(res), np.median(res), sum(r < 0 for r in res), sum(r < 8 for r in res)))
