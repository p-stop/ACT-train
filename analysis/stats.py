import sys, json, numpy as np, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]
d = pd.read_parquet(f"{D}/data/chunk-000/file-000.parquet")
A = np.stack(d.action.values); S = np.stack(d["observation.state"].values)
ep = d.episode_index.values
FPS = 30
rows = []; grasp = {"L": [], "R": []}; start = []
def segs(mask):
    m = np.r_[0, mask.astype(int), 0]; dm = np.diff(m)
    return list(zip(np.where(dm == 1)[0], np.where(dm == -1)[0]))
for e in np.unique(ep):
    i = ep == e; a = A[i]; s = S[i]; n = len(a)
    r = {"ep": int(e), "len_s": n / FPS}
    start.append(s[0])
    for k, sl, g in (("L", slice(0, 6), 6), ("R", slice(7, 13), 13)):
        v = np.linalg.norm(np.diff(a[:, sl], axis=0), axis=1) * FPS  # rad/s
        mov = v > 0.05
        mv = np.where(mov)[0]
        r[f"{k}_move_start_s"] = mv[0] / FPS if len(mv) else None
        r[f"{k}_move_end_s"] = mv[-1] / FPS if len(mv) else None
        r[f"{k}_moving_frac"] = mov.mean()
        closed = a[:, g] < 0.035
        sg = [(b, c) for b, c in segs(closed) if c - b > 5]
        r[f"{k}_n_close"] = len(sg)
        r[f"{k}_close_segs"] = [(round(b / FPS, 1), round(c / FPS, 1)) for b, c in sg]
        # empty grasp: measured opening < 8mm while closed (sponge squeezes, so only for L meaningful)
        r[f"{k}_min_state_grip_in_close"] = [round(float(s[b:c, g].min()) * 1000, 1) for b, c in sg]
        if sg:
            b = sg[0][0]
            grasp[k].append(np.r_[e, s[b, sl]])
            r[f"{k}_first_close_s"] = b / FPS
            r[f"{k}_release_s"] = sg[0][1] / FPS
    # overlap: is the other arm moving while one arm works?
    vL = np.linalg.norm(np.diff(a[:, 0:6], axis=0), axis=1) * FPS
    vR = np.linalg.norm(np.diff(a[:, 7:13], axis=0), axis=1) * FPS
    r["both_moving_frac"] = float(((vL > 0.05) & (vR > 0.05)).mean())
    r["static_frac"] = float(((vL < 0.05) & (vR < 0.05)).mean())
    # L moves before R finished release?
    if "R_release_s" in r and r["L_move_start_s"] is not None:
        r["L_start_minus_R_release_s"] = r["L_move_start_s"] - r["R_release_s"]
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/episodes.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
print(df[["ep","len_s","R_move_start_s","R_first_close_s","R_release_s","R_n_close","L_move_start_s","L_first_close_s","L_release_s","L_n_close","L_start_minus_R_release_s","both_moving_frac","static_frac"]].round(1).to_string())
print(df[["ep","R_close_segs","R_min_state_grip_in_close","L_close_segs","L_min_state_grip_in_close"]].to_string())
np.set_printoptions(precision=3, suppress=True, linewidth=200)
st = np.array(start); print("start pose std", st.std(0))
for k in "LR":
    g = np.array(grasp[k]); print(k, "grasp n", len(g), "joint std", g[:, 1:].std(0), "range", np.ptp(g[:, 1:], 0))
    np.save(f"{OUT}/grasp_{k}.npy", g)
