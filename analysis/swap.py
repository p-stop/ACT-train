# Which input does the left arm read its target from?  At the start of the left phase, take episode e and
# replace ONE input with the one from donor episode d (same phase). If the prediction then follows d's demo,
# that input carries the information.
import sys, numpy as np, pandas as pd, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
CK = sys.argv[1]
pol = get_policy_class("act").from_pretrained(CK).to("cuda").eval()
pre, post = make_pre_post_processors(pol.config, pretrained_path=CK)
ds = LeRobotDataset("<HF_USER>/01_ACT")
st = pd.read_csv("episodes.csv")
ep_from = ds.meta.episodes["dataset_from_index"]; ep_len = ds.meta.episodes["length"]
A = np.stack(ds.hf_dataset.with_format("numpy")["action"])
K = ["observation.state", "observation.images.left_wrist", "observation.images.right_wrist", "observation.images.scene_fixed"]
@torch.no_grad()
def predict(o):
    b = pre({**{k: o[k].unsqueeze(0) for k in K}, "task": [o["task"]]}); pol.reset()
    return post(pol.predict_action_chunk(b)).squeeze(0).cpu().numpy()
eps = [int(s.ep) for _, s in st.iterrows() if s.L_move_start_s * 30 > s.R_release_s * 30]
frames = {}
for e in eps:
    s = st[st.ep == e].iloc[0]; g = int(ep_from[e]) + int(s.L_move_start_s * 30)
    frames[e] = (g, ds[g])
rng = np.random.default_rng(0)
LJ = slice(0, 6)
def gt(g, e): return A[min(g + 99, int(ep_from[e]) + int(ep_len[e]) - 1), LJ] - A[g, LJ]
rows = []
variants = {"state_R(right arm joints)": lambda o, d: {**o, "observation.state": torch.cat([o["observation.state"][:7], d["observation.state"][7:]])},
            "state_L(left arm joints)": lambda o, d: {**o, "observation.state": torch.cat([d["observation.state"][:7], o["observation.state"][7:]])},
            "scene image": lambda o, d: {**o, "observation.images.scene_fixed": d["observation.images.scene_fixed"]},
            "right_wrist image": lambda o, d: {**o, "observation.images.right_wrist": d["observation.images.right_wrist"]},
            "left_wrist image": lambda o, d: {**o, "observation.images.left_wrist": d["observation.images.left_wrist"]}}
for e in eps:
    for d in rng.choice([x for x in eps if x != e], 4, replace=False):
        ge, oe = frames[e]; gd, od = frames[d]
        own, don = gt(ge, e), gt(gd, d)
        if np.linalg.norm(own - don) < 0.15: continue
        for name, f in variants.items():
            p = predict(f(oe, od)); p = p[99, LJ] - p[0, LJ]
            # where between own (0) and donor (1) does the prediction land?
            t = float(np.dot(p - own, don - own) / np.dot(don - own, don - own))
            rows.append({"ep": e, "donor": int(d), "swap": name, "toward_donor": t})
df = pd.DataFrame(rows); df.to_csv("swap.csv", index=False)
print("left arm, first 100 predicted steps, 6 joints. 0 = keeps its own demo, 1 = switches to the donor demo")
print(df.groupby("swap").toward_donor.describe()[["count", "mean", "25%", "50%", "75%"]].round(2).to_string())
