"""Small thumbnails of sampled frames (all cameras) to compare two encodings of the same dataset."""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, sys.argv[3])
from common import Source, CAMS
src = Source(sys.argv[1]); out = {}
for ep in range(0, src.n_episodes(), max(1, src.n_episodes() // 8)):
    n = src.length(ep)
    for c in CAMS:
        r = src.reader(ep, c)
        for i in (0, n // 3, 2 * n // 3, n - 1):
            out[f"{ep}_{c[-11:]}_{i}"] = cv2.resize(r.get(i), (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
        r.close()
np.savez_compressed(sys.argv[2], **out); print(len(out), "frames")
