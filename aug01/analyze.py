"""Per-episode scene analysis: arm-swept regions per phase and object regions (full-res masks, saved as npz)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from common import Source, SCENE, H, W, phases

src = Source(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
K3 = np.ones((3, 3), np.uint8)
DARK = 60


def dark(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < DARK


def objects(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    roi = np.zeros((H, W), bool); roi[80:380, 40:600] = True
    red = roi & ((h < 8) | (h > 245)) & (s > 140) & (v > 120)
    brown = roi & (h >= 8) & (h <= 22) & (s > 90) & (v > 60) & (v < 200) & ~red
    brown = cv2.morphologyEx(brown.astype(np.uint8), cv2.MORPH_OPEN, K3).astype(bool)
    green = roi & (h > 50) & (h < 110) & (s > 40)
    blue = roi & (h > 115) & (h < 150) & (s > 60)
    def comps(mask, minpx):
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        return sorted([(st[i][4], lab == i) for i in range(1, n) if st[i][4] >= minpx], key=lambda t: -t[0])
    co = comps(cv2.morphologyEx((brown | red).astype(np.uint8), cv2.MORPH_CLOSE, K3, iterations=3), 400)[:2]
    if len(co) == 2 and co[1][0] < 0.4 * co[0][0]:
        co = co[:1]
    if len(co) == 1 and red.any():
        # two coasters touching: split the blob by distance to the sponge centre vs the far end of the blob
        m = co[0][1]; ys, xs = np.where(m); ry, rx = [c.mean() for c in np.where(red & m)]
        far = np.argmax((ys - ry) ** 2 + (xs - rx) ** 2); fy, fx = ys[far], xs[far]
        for _ in range(10):  # 2-means
            near_s = (ys - ry) ** 2 + (xs - rx) ** 2 < (ys - fy) ** 2 + (xs - fx) ** 2
            ry, rx, fy, fx = ys[near_s].mean(), xs[near_s].mean(), ys[~near_s].mean(), xs[~near_s].mean()
        a = np.zeros_like(m); a[ys[near_s], xs[near_s]] = True
        b = np.zeros_like(m); b[ys[~near_s], xs[~near_s]] = True
        co = [(a.sum(), a), (b.sum(), b)]
    co = sorted([m for _, m in co], key=lambda m: np.where(m)[1].mean())
    g = comps(green, 400); b = comps(blue, 400)
    fill = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return {"wood": fill(co[0]) if len(co) == 2 else None, "sponge": fill(co[1]) if len(co) == 2 else None,
            "green": fill(g[0][1]) if g else None, "blue": fill(b[0][1]) if b else None}


summary = {}
for ep in range(src.n_episodes()):
    ph = phases(src, ep)
    r = src.reader(ep, SCENE)
    f0 = r.get(0).copy(); d0 = cv2.dilate(dark(f0).astype(np.uint8), K3, iterations=2).astype(bool)
    cntR = np.zeros((H, W), np.uint16); cntL = np.zeros((H, W), np.uint16)
    fpark = None; dpark = None; fL = None
    for i in range(0, ph["end"] + 1):
        img = r.get(i)
        if i == ph["R_park"]:
            fpark = img.copy(); dpark = cv2.dilate(dark(fpark).astype(np.uint8), K3, iterations=2).astype(bool)
        if i == ph["L_start"]:
            fL = img.copy()
        if ph["R_start"] - 5 <= i <= ph["R_park"]:
            cntR += dark(img) & ~d0
        elif i > ph["R_park"]:
            cntL += dark(img) & ~dpark
    r.close()
    clean = lambda c: cv2.morphologyEx((c >= 3).astype(np.uint8), cv2.MORPH_OPEN, K3).astype(bool)
    swR, swL = clean(cntR), clean(cntL)
    parkR = cv2.morphologyEx((dark(fpark) & ~d0).astype(np.uint8), cv2.MORPH_OPEN, K3).astype(bool)
    ob = objects(f0)
    if any(v is None for v in ob.values()):
        print(f"ep {ep}: object segmentation failed: {[k for k, v in ob.items() if v is None]}", flush=True)
        ob = {k: (v if v is not None else np.zeros((H, W), bool)) for k, v in ob.items()}
    np.savez_compressed(out / f"ep_{ep:03d}.npz", swR=swR, swL=swL, parkR=parkR, **{f"obj_{k}": v for k, v in ob.items()})
    cv2.imwrite(str(out / f"ep_{ep:03d}_first.jpg"), cv2.cvtColor(f0, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / f"ep_{ep:03d}_park.jpg"), cv2.cvtColor(fpark, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    summary[ep] = {**ph, "swR_px": int(swR.sum()), "swL_px": int(swL.sum()), "parkR_px": int(parkR.sum())}
    print(ep, summary[ep], flush=True)
(out / "phases.json").write_text(json.dumps(summary, indent=1))
