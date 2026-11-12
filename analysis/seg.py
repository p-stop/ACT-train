# colour segmentation on 320x240 tiles of the first-frame scene montage
import numpy as np, pandas as pd, sys
from PIL import Image
import cv2
K=np.ones((3,3),np.uint8)
M = np.asarray(Image.open("m_scene_fixed_first.jpg").convert("HSV")).astype(int)
R = np.asarray(Image.open("m_scene_fixed_first.jpg").convert("RGB")).astype(int)
W, H, cols = 320, 240, 8
st = pd.read_csv("episodes.csv")
def blobs(mask, minpx):
    n, lab, stt, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = [(cen[i][0], cen[i][1], stt[i][4]) for i in range(1, n) if stt[i][4] >= minpx]
    return sorted(out, key=lambda b: -b[2])
rows = []
for ep in st.ep:
    x0, y0 = (ep % cols) * W, (ep // cols) * H
    hsv = M[y0:y0 + H, x0:x0 + W]; rgb = R[y0:y0 + H, x0:x0 + W]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    roi = np.zeros_like(h, bool); roi[40:190, 20:300] = True  # table area, excludes grippers at bottom
    red = roi & ((h < 8) | (h > 245)) & (s > 140) & (v > 120)
    brown = roi & (h >= 8) & (h <= 22) & (s > 90) & (v > 60) & (v < 200) & ~red
    brown = cv2.morphologyEx(brown.astype(np.uint8), cv2.MORPH_OPEN, K).astype(bool)
    green = roi & (h > 50) & (h < 110) & (s > 40)
    blue = roi & (h > 115) & (h < 150) & (s > 60)
    r = {"ep": ep}
    sp = blobs(red, 15); co = blobs(cv2.morphologyEx((brown | red).astype(np.uint8), cv2.MORPH_CLOSE, K, iterations=2), 60)
    if sp: r["sp_x"], r["sp_y"], r["sp_px"] = sp[0]
    co = sorted(co[:2], key=lambda b: b[0])
    if len(co) == 2:
        r["coL_x"], r["coL_y"], r["coL_px"] = co[0]; r["coR_x"], r["coR_y"], r["coR_px"] = co[1]
    g = blobs(green, 100); b = blobs(blue, 100)
    if g: r["gp_x"], r["gp_y"] = g[0][:2]
    if b: r["bp_x"], r["bp_y"] = b[0][:2]
    rows.append(r)
df = pd.DataFrame(rows)
# the sponge coaster is the one nearest to the sponge
def near(r):
    dL = np.hypot(r.sp_x - r.coL_x, r.sp_y - r.coL_y); dR = np.hypot(r.sp_x - r.coR_x, r.sp_y - r.coR_y)
    return pd.Series({"sponge_on": "L" if dL < dR else "R", "sp_dx": r.sp_x - (r.coR_x if dR <= dL else r.coL_x), "sp_dy": r.sp_y - (r.coR_y if dR <= dL else r.coL_y)})
df = pd.concat([df, df.apply(near, axis=1)], axis=1)
df.to_csv("positions.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(1).to_string())
print(df.describe().round(1).to_string())
