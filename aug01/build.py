"""Build augmented datasets from 01_put_blocks_into_plate.

  mixr  : every original episode + K copies whose scene image, from the moment the right arm is parked, shows another
          episode's right side: the region of the parked right arm, the blue plate with the sponge and the empty coaster
          (of both episodes, dilated) is taken from the donor's frame at the moment its right arm parked (the right side is
          static from then on). Wherever this episode's own left arm is in that region, its own pixels are kept.
  fused : left-arm phase of episode i and right-arm phase of episode j played at the same time
          (left of the dividing line from i, right of it from j; wrist cameras and joints from their own source).
          --pairing same (j = i) or cross (j random, K per i).

Dividing lines are (xt, xb): x at the top / bottom image row. fused uses the hand-labelled lines from
--seams (JSON {"<ep>": [xt, xb]}) where present, otherwise the automatic line of that episode.
"""
import argparse, json, shutil, time
from pathlib import Path
import cv2
import numpy as np
from common import (Source, CAMS, SCENE, H, W, phases, line_mask, blend, best_line, margin, profile,
                    rgb_encoder, user_features)

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["mixr", "fused"])
ap.add_argument("--src", required=True); ap.add_argument("--analysis", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--repo_id", required=True)
ap.add_argument("--pairing", choices=["same", "cross"], default="same")
ap.add_argument("--k", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min_margin", type=float, default=8.0, help="fused: px the arms must stay clear of the line (cross pairs); mixr: clearance of the donor region from the wood block / green plate")
ap.add_argument("--seams", default=None); ap.add_argument("--lead", type=int, default=15)
ap.add_argument("--limit", type=int, default=None, help="only the first N output episodes (smoke test)")
ap.add_argument("--plan_only", action="store_true")
ap.add_argument("--keep_all", action="store_true", help="fused same: keep every episode even if no line separates the arms")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
src = Source(args.src); N = src.n_episodes()
PH = [phases(src, e) for e in range(N)]
Z = [dict(np.load(Path(args.analysis) / f"ep_{e:03d}.npz")) for e in range(N)]
# per-episode profiles of what has to stay left / right of the dividing line
PL = [profile(z["swL"], z["obj_wood"], z["obj_green"]) for z in Z]          # left arm (left phase) + its objects
PLo = [profile(z["obj_wood"], z["obj_green"]) for z in Z]                   # left-side objects only
PRpark = [profile(z["parkR"], z["obj_sponge"], z["obj_blue"]) for z in Z]   # parked right arm + right-side objects
PRsw = [profile(z["swR"], z["parkR"], z["obj_sponge"], z["obj_blue"]) for z in Z]  # right arm (right phase) + objects
U = lambda *ps: (np.maximum.reduce([p[0] for p in ps]), np.minimum.reduce([p[1] for p in ps]))


DISK = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
dil = lambda m, r: cv2.dilate(m.astype(np.uint8), DISK(r)).astype(bool)
def arm_silhouette(e):
    """whole parked right arm: dark blobs of the parked frame touching the moved part, convex hull per blob"""
    img = cv2.cvtColor(cv2.imread(str(Path(args.analysis) / f"ep_{e:03d}_park.jpg")), cv2.COLOR_BGR2RGB)
    dk = cv2.morphologyEx(dark(img).astype(np.uint8), cv2.MORPH_OPEN, DISK(1))
    n, lab = cv2.connectedComponents(dk)
    seed = dil(Z[e]["parkR"], 3)
    out = np.zeros((H, W), np.uint8)
    for k in set(np.unique(lab[seed & (lab > 0)])):
        pts = cv2.findNonZero((lab == k).astype(np.uint8))
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool) | Z[e]["parkR"]


def dark(img, thr=60):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < thr


PARK = [dil(arm_silhouette(e), 10) | dil(Z[e]["obj_blue"], 6) for e in range(N)]
LEFTARM = [dil(z["swL"], 10) for z in Z]   # where this episode's left arm can be during the left phase
COASTER = [dil(z["obj_sponge"], 4) for z in Z]   # the empty sponge coaster (the sponge is in the plate by then)
LEFTOBJ = [dil(z["obj_wood"] | z["obj_green"], 4) for z in Z]


def clearance(a, b):
    return float(cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)[b].min()) if a.any() and b.any() else 999.0


def mixr_region(e, d):
    """region taken from the donor: parked right arm + blue plate of both episodes; each empty sponge coaster only
    if it is clear of the wood block / green plate of both. Returns (mask, clearance of the arm/plate part, info)."""
    lo = LEFTOBJ[e] | LEFTOBJ[d]
    m = PARK[e] | PARK[d]
    c = clearance(m, lo)
    used = {}
    for who, x in (("own_coaster", e), ("donor_coaster", d)):
        used[who] = clearance(COASTER[x], lo) >= 4
        if used[who]:
            m = m | COASTER[x]
    return m, c, used





def sets_fused(i, j):
    return U(PL[i], PLo[j]), U(PRsw[j], PRpark[i])


# ---------------- plan ----------------
plan = []  # dicts: kind, e/i/j, line, margin
if args.mode == "mixr":
    for e in range(N):
        plan.append({"kind": "orig", "src": e})
        ok = [(mixr_region(e, d)[1], d) for d in range(N) if d != e]
        ok = [c for c in ok if c[0] >= args.min_margin]
        ok = [c for c in ok if c[0] >= args.min_margin]
        pick = [ok[i] for i in rng.choice(len(ok), min(args.k, len(ok)), replace=False)] if ok else []
        if len(pick) < args.k:
            print(f"ep {e}: only {len(ok)} donors whose right side stays >= {args.min_margin}px clear of the wood block / green plate", flush=True)
        for m, d in pick:
            plan.append({"kind": "mixr", "src": e, "donor": int(d), "clearance": round(m, 1), **mixr_region(e, d)[2]})
else:
    seams = json.loads(Path(args.seams).read_text()) if args.seams and Path(args.seams).exists() else {}
    seams = seams.get("seams", seams)  # the preview tool stores {"dataset", "updated", "seams": {...}}
    skip = {int(k) for k, v in seams.items() if v == "skip"}
    seams = {k: v for k, v in seams.items() if v != "skip"}
    if skip: print(f"marked as not to be used for fusing: {sorted(skip)}", flush=True)
    auto = {}
    for e in range(N):
        m, line = best_line(*sets_fused(e, e)); auto[e] = (list(line), m)
    own = lambda e: (list(seams[str(e)]), "hand") if str(e) in seams else (auto[e][0], "auto")
    for i in range(N):
        if i in skip: continue
        js = [i] if args.pairing == "same" else list(rng.permutation([j for j in range(N) if j != i and j not in skip]))
        got = 0
        for j in js:
            A, B = sets_fused(i, j)
            opts = [own(i)] + ([own(j)] if j != i else [])
            scored = sorted(((margin(*l, A, B), l, how, who) for (l, how), who in zip(opts, ("left", "right"))), key=lambda t: -t[0])
            m, line, how, who = scored[0]
            if m < (args.min_margin if args.pairing == "cross" else 0) and not args.keep_all:
                if args.pairing == "same":
                    print(f"ep {i}: {how} line {line} does not separate the arms (margin {m:.1f}); skipped", flush=True)
                continue
            plan.append({"kind": "fused", "left": i, "right": int(j), "line": list(map(int, line)), "line_from": f"{how}:{who}", "margin": round(m, 1)})
            got += 1
            if args.pairing == "same" or got >= args.k: break
        if args.pairing == "cross" and got < args.k:
            print(f"ep {i}: only {got} right partners", flush=True)
print(f"plan: {len(plan)} episodes", flush=True)
if args.limit: plan = plan[:args.limit]
out = Path(args.out)
if args.plan_only:
    print(json.dumps(plan[:10], indent=0)); raise SystemExit

# ---------------- write ----------------
from lerobot.datasets.lerobot_dataset import LeRobotDataset
if out.exists(): shutil.rmtree(out)
ds = LeRobotDataset.create(args.repo_id, fps=30, features=user_features(src), root=out, robot_type=src.info["robot_type"],
                           use_videos=True, rgb_encoder=rgb_encoder(), image_writer_threads=8)
TASK_FUSED = "put the orange sponge into the right plate and the wooden block into the left plate at the same time"
alpha_cache = {}
def alpha(line):
    k = tuple(line)
    if k not in alpha_cache: alpha_cache[k] = line_mask(*k)
    return alpha_cache[k]

t0 = time.time(); nfr = 0
for n, p in enumerate(plan):
    if p["kind"] in ("orig", "mixr"):
        e = p["src"]; rows = src.rows(e); T = src.length(e)
        rd = {c: src.reader(e, c) for c in CAMS}
        if p["kind"] == "mixr":
            d = p["donor"]; pe = PH[e]
            r = src.reader(d, SCENE); dpark = r.get(PH[d]["R_park"]).astype(np.float32); r.close()
            region = cv2.GaussianBlur(mixr_region(e, d)[0].astype(np.float32), (0, 0), 3)[..., None]
            ref_dark = None
        for t in range(T):
            fr = {c: rd[c].get(t) for c in CAMS}
            if p["kind"] == "mixr" and t >= pe["R_park"]:
                if ref_dark is None:  # this episode's frame with the right arm just parked and the left arm at home
                    ref_dark = dil(dark(fr[SCENE]), 3)
                own_arm = dil(dark(fr[SCENE]) & ~ref_dark & LEFTARM[e], 12)  # where the left arm is right now: keep own pixels
                a = region * (1 - cv2.GaussianBlur(own_arm.astype(np.float32), (0, 0), 3)[..., None])
                fr[SCENE] = (fr[SCENE] * (1 - a) + dpark * a + 0.5).astype(np.uint8)
            ds.add_frame({"observation.state": src.state[rows][t], "action": src.action[rows][t], **fr, "task": src.task})
        for r in rd.values(): r.close()
    else:
        i, j = p["left"], p["right"]; pi, pj = PH[i], PH[j]
        Ls, Le = max(pi["L_start"] - args.lead, 0), pi["end"]
        Rs, Re = max(pj["R_start"] - args.lead, 0), min(pj["R_park"] + args.lead, pj["end"])
        T = max(Le - Ls, Re - Rs) + 1
        ri, rj = src.rows(i), src.rows(j)
        li = {c: src.reader(i, c) for c in ("observation.images.left_wrist", SCENE)}
        rjr = {c: src.reader(j, c) for c in ("observation.images.right_wrist", SCENE)}
        a = alpha(p["line"])
        # collision screening in the scene image: left arm (from i) vs right arm (from j) at the same output time
        regL = dil(Z[i]["swL"], 10); regR = dil(Z[j]["swR"] | Z[j]["parkR"], 10)
        ys, xs = np.mgrid[0:H, 0:W]; side = xs - (p["line"][0] + (p["line"][1] - p["line"][0]) * ys / (H - 1))
        refL = refR = None; best = (np.inf, 0); close_frames = 0; crossL = crossR = 0.0
        for t in range(T):
            u, v = min(Ls + t, Le), min(Rs + t, Re)
            if t % 2 == 0:
                fl, fr_ = li[SCENE].get(u), rjr[SCENE].get(v)
                if refL is None:  # both arms at home at the first output frame
                    refL, refR = dil(dark(fl), 3), dil(dark(fr_), 3)
                mL = cv2.morphologyEx((dark(fl) & ~refL & regL).astype(np.uint8), cv2.MORPH_OPEN, DISK(2)).astype(bool)
                mR = cv2.morphologyEx((dark(fr_) & ~refR & regR).astype(np.uint8), cv2.MORPH_OPEN, DISK(2)).astype(bool)
                if mL.any() and mR.any():
                    dmap = cv2.distanceTransform((~mR).astype(np.uint8), cv2.DIST_L2, 3)
                    dmin = float(dmap[mL].min())
                    if dmin < best[0]: best = (dmin, t)
                    close_frames += 2 * (dmin < 30)
                if mL.any(): crossL = max(crossL, float(side[mL].max()))
                if mR.any(): crossR = max(crossR, float(-side[mR].min()))
            st = np.concatenate([src.state[ri][u][:7], src.state[rj][v][7:]])
            ac = np.concatenate([src.action[ri][u][:7], src.action[rj][v][7:]])
            ds.add_frame({"observation.state": st, "action": ac,
                          "observation.images.left_wrist": li["observation.images.left_wrist"].get(u),
                          "observation.images.right_wrist": rjr["observation.images.right_wrist"].get(v),
                          SCENE: blend(li[SCENE].get(u), rjr[SCENE].get(v), a), "task": TASK_FUSED})
        for r in list(li.values()) + list(rjr.values()): r.close()
    ds.save_episode()
    if p["kind"] == "fused":
        p["arm_min_dist_px"] = round(best[0], 1) if np.isfinite(best[0]) else None
        p["arm_min_dist_t"] = round(best[1] / 30, 2); p["arm_close_s"] = round(close_frames / 30, 2)
        p["left_over_line_px"] = round(crossL, 1); p["right_over_line_px"] = round(crossR, 1)
    p["out_ep"] = n; p["length"] = T; nfr += T
    print(f"[{n + 1}/{len(plan)}] {p}  {nfr / (time.time() - t0):.0f} fps", flush=True)
ds.finalize()
(out / "meta" / "aug_manifest.json").write_text(json.dumps({"args": vars(args), "episodes": plan}, indent=1))
print("done", out, flush=True)
