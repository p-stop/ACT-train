"""Scene frames of a fused dataset at the moment the two arms come closest (from aug_manifest.json)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from common import Source, SCENE
src = Source(sys.argv[1]); eps = [int(x) for x in sys.argv[3].split(",")]
man = {e["out_ep"]: e for e in json.loads((src.root / "meta/aug_manifest.json").read_text())["episodes"]}
tiles = []
for ep in eps:
    e = man[ep]; i = int(e["arm_min_dist_t"] * 30)
    r = src.reader(ep, SCENE); img = cv2.cvtColor(r.get(i), cv2.COLOR_RGB2BGR); r.close()
    xt, xb = e["line"]; cv2.line(img, (int(xt), 0), (int(xb), 479), (0, 255, 0), 1)
    cv2.rectangle(img, (0, 0), (330, 22), (0, 0, 0), -1)
    cv2.putText(img, f"#{ep} t={e['arm_min_dist_t']}s min {e['arm_min_dist_px']}px", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    tiles.append(cv2.resize(img, (480, 360)))
while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
