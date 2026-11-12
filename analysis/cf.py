# Counterfactual probe of the trained ACT: edit the scene image (move / remove objects) and see
# whether the predicted base yaw (joint1) of the acting arm follows, compared with the slope seen in the demos.
import sys, json, numpy as np, pandas as pd, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
CK = sys.argv[1]
torch.manual_seed(0)
pol = get_policy_class("act").from_pretrained(CK).to("cuda").eval()
pre, post = make_pre_post_processors(pol.config, pretrained_path=CK)
ds = LeRobotDataset("<HF_USER>/01_ACT")
st = pd.read_csv("episodes.csv"); P = pd.read_csv("positions.csv")
ep_from = ds.meta.episodes["dataset_from_index"]
A_all = np.stack(ds.hf_dataset.with_format("numpy")["action"])

@torch.no_grad()
def predict(item, scene):
    obs = {k: item[k].unsqueeze(0) for k in ("observation.state", "observation.images.left_wrist", "observation.images.right_wrist")}
    obs["observation.images.scene_fixed"] = scene.unsqueeze(0)
    obs["task"] = [item["task"]]
    b = pre(obs)
    pol.reset()
    ch = pol.predict_action_chunk(b)  # (1,100,14) normalized
    return post(ch).squeeze(0).cpu().numpy()

def disk(img, cx, cy, r):
    H, W = img.shape[1:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

def table_fill(img, cx, cy, r):
    # median colour of a ring just outside the patch
    m = disk(img, cx, cy, r + 14) & ~disk(img, cx, cy, r + 6)
    col = img[:, m].median(dim=1).values
    out = img.clone(); d = disk(img, cx, cy, r)
    out[:, d] = col[:, None] + 0.01 * torch.randn(3, int(d.sum()))
    return out

def move(img, cx, cy, r, dx, dy=0, keep_src=False, mask=None):
    m = disk(img, cx, cy, r) if mask is None else mask
    out = img.clone() if keep_src else table_fill(img, cx, cy, r) if mask is None else img.clone()
    ys, xs = torch.where(m)
    out[:, ys + dy, xs + dx] = img[:, ys, xs]
    return out

def red_mask(img):
    r, g, b = img
    return (r > 0.55) & (r - g > 0.3) & (r - b > 0.25)

res = []
DX = 40  # full-res pixels (= 20 px in the 320x240 tiles used for positions.csv)
for _, s in st.iterrows():
    e = int(s.ep); p = P[P.ep == e].iloc[0]
    if np.isnan(p.coL_x): continue
    for arm, f, j1, tgt in (("R", int(s.R_move_start_s * 30), 7, (p.coR_x, p.coR_y)), ("L", int(s.L_move_start_s * 30), 0, (p.coL_x, p.coL_y))):
        if arm == "L" and f < s.R_release_s * 30: continue  # eps with early jitter of the left arm
        if arm == "R" and e == 1: continue
        g = int(ep_from[e]) + f
        it = ds[g]; sc = it["observation.images.scene_fixed"]
        cx, cy, R = 2 * tgt[0], 2 * tgt[1], 32
        gt = A_all[min(g + 99, int(ep_from[e]) + int(ds.meta.episodes["length"][e]) - 1), j1] - A_all[g, j1]
        row = {"ep": e, "arm": arm, "gt_dj1": gt}
        base = predict(it, sc); row["base"] = base[99, j1] - base[0, j1]
        for sgn in (-1, 1):
            row[f"obj+coaster_{sgn:+d}"] = predict(it, move(sc, cx, cy, R, sgn * DX))[99, j1] - base[99, j1]
        if arm == "R":
            rm = red_mask(sc) & disk(sc, cx, cy, R)
            # sponge alone moved, coaster stays (sponge hole filled with coaster colour)
            for sgn in (-1, 1):
                img = sc.clone(); ring = disk(sc, cx, cy, R - 6) & ~rm
                img[:, rm] = sc[:, ring].median(dim=1).values[:, None]
                ys, xs = torch.where(rm); img[:, ys, xs + sgn * DX] = sc[:, ys, xs]
                row[f"sponge_only_{sgn:+d}"] = predict(it, img)[99, j1] - base[99, j1]
                # coaster alone moved, sponge stays where it was (on bare table)
                img = move(sc, cx, cy, R, sgn * DX)
                img[:, rm] = sc[:, rm]  # sponge pasted back at the original spot
                row[f"coaster_only_{sgn:+d}"] = predict(it, img)[99, j1] - base[99, j1]
            # coaster removed, sponge stays
            img = table_fill(sc, cx, cy, R); img[:, rm] = sc[:, rm]
            row["no_coaster"] = predict(it, img)[99, j1] - base[99, j1]
        else:
            # the now-empty sponge coaster moved instead of the wood block
            ox, oy = 2 * p.coR_x, 2 * p.coR_y
            for sgn in (-1, 1):
                row[f"empty_coaster_{sgn:+d}"] = predict(it, move(sc, ox, oy, R, sgn * DX))[99, j1] - base[99, j1]
        res.append(row)
    print(e, flush=True)
df = pd.DataFrame(res); df.to_csv("counterfactual.csv", index=False)
# expected slope from the demos: gt joint1 change over the same 100 steps vs target x (tile px)
for arm, col in (("R", "coR_x"), ("L", "coL_x")):
    d = df[df.arm == arm].merge(P[["ep", col, col.replace("_x", "_y")]], on="ep")
    X = np.c_[d[col], d[col.replace("_x", "_y")], np.ones(len(d))]
    b, *_ = np.linalg.lstsq(X, d.gt_dj1, rcond=None)
    print(f"\n== {arm} arm, n={len(d)}; demo slope d(joint1)/dx = {b[0]:.4f} rad per tile px -> expected for +{DX//2} tile px: {b[0]*DX/2:+.3f} rad")
    print("model fit on training frames: corr(base, gt) = %.3f" % np.corrcoef(d.base, d.gt_dj1)[0, 1])
    print(d.filter(regex="_[+-]1$|no_coaster").describe().loc[["mean", "std"]].round(3).to_string())
