"""Draw arm-swept regions and objects on the first frame of each episode (for checking and for the seam tool)."""
import sys
from pathlib import Path
import cv2, numpy as np
d = Path(sys.argv[1])
tiles = []
for f in sorted(d.glob("ep_*.npz")):
    z = np.load(f); img = cv2.imread(str(f).replace(".npz", "_first.jpg")).astype(np.float32)
    def tint(m, bgr, a=0.45):
        img[m] = img[m] * (1 - a) + np.array(bgr, np.float32) * a
    tint(z["swL"], (255, 120, 0)); tint(z["swR"], (0, 140, 255)); tint(z["parkR"], (200, 0, 200), 0.6)
    img = img.astype(np.uint8)
    for k, c in (("obj_wood", (255, 255, 0)), ("obj_sponge", (0, 0, 255)), ("obj_green", (0, 200, 0)), ("obj_blue", (255, 0, 0))):
        cs, _ = cv2.findContours(z[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, c, 2)
    cv2.imwrite(str(f).replace(".npz", "_overlay.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    t = cv2.resize(img, (320, 240)); cv2.rectangle(t, (0, 0), (34, 18), (0, 0, 0), -1)
    cv2.putText(t, f.stem[3:].lstrip("0") or "0", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(t)
while len(tiles) % 8: tiles.append(np.full_like(tiles[0], 255))
rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
cv2.imwrite(str(d / "montage_overlay.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
