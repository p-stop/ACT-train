import sys
from pathlib import Path
import cv2, numpy as np
d = Path(sys.argv[1])
DISK = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
dil = lambda m, r: cv2.dilate(m.astype(np.uint8), DISK(r)).astype(bool) if r else m.astype(bool)
dist = lambda a, b: float(cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)[b].min()) if b.any() and a.any() else 999
Z = [dict(np.load(d / f"ep_{e:03d}.npz")) for e in range(52)]
park = [cv2.morphologyEx(z["parkR"].astype(np.uint8), cv2.MORPH_CLOSE, DISK(15)).astype(bool) for z in Z]
print("ep  park->wood park->green sponge->wood blue->wood blue->green  (px, own episode)")
for e, z in enumerate(Z):
    print(e, *[round(x) for x in (dist(park[e], z["obj_wood"]), dist(park[e], z["obj_green"]), dist(z["obj_sponge"], z["obj_wood"]), dist(z["obj_blue"], z["obj_wood"]), dist(z["obj_blue"], z["obj_green"]))])
