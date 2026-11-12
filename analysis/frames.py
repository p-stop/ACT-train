# grab key frames per episode from the concatenated videos, save montages
import sys, numpy as np, pandas as pd, av
from PIL import Image, ImageDraw
D, OUT = sys.argv[1], sys.argv[2]
e = pd.read_parquet(f"{D}/meta/episodes/chunk-000/file-000.parquet")
st = pd.read_csv(f"{OUT}/episodes.csv")
FPS = 30
keys = {"first": lambda r: 0, "R_grasp": lambda r: int(r.R_first_close_s * FPS) if r.ep != 1 else int(10.5 * FPS),
        "L_start": lambda r: int(r.R_release_s * FPS) + 60, "L_grasp": lambda r: int(r.L_first_close_s * FPS),
        "last": lambda r: None}
want = {}
for _, r in st.iterrows():
    n = int(e.length[r.ep])
    for k, f in keys.items():
        fi = f(r); fi = n - 1 if fi is None else fi
        want[(int(r.ep), k)] = fi
for cam in ["scene_fixed", "left_wrist", "right_wrist"]:
    c = f"videos/observation.images.{cam}"
    tgt = {}
    for (ep, k), fi in want.items():
        g = int(round(e[f"{c}/from_timestamp"][ep] * FPS)) + fi
        tgt.setdefault((int(e[f"{c}/file_index"][ep]), g), []).append((ep, k))
    got = {}
    for fidx in sorted({f for f, _ in tgt}):
        t = {g: v for (f, g), v in tgt.items() if f == fidx}
        with av.open(f"{D}/{c}/chunk-000/file-{fidx:03d}.mp4") as v:
            for i, fr in enumerate(v.decode(video=0)):
                if i in t:
                    im = fr.to_image()
                    for ek in t[i]: got[ek] = im
                if i > max(t): break
    for k in keys:
        W, H, cols = 320, 240, 8
        rows = (len(st) + cols - 1) // cols
        M = Image.new("RGB", (cols * W, rows * H), "white")
        for ep in st.ep:
            im = got[(ep, k)].resize((W, H)); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 34, 20], fill="black"); d.text((4, 4), str(ep), fill="yellow")
            M.paste(im, ((ep % cols) * W, (ep // cols) * H))
        M.save(f"{OUT}/m_{cam}_{k}.jpg", quality=85)
    # full-res sample
    for ep in (0, 25, 51):
        got[(ep, "first")].save(f"{OUT}/full_{cam}_ep{ep}_first.png")
        got[(ep, "L_start")].save(f"{OUT}/full_{cam}_ep{ep}_Lstart.png")
import pickle; pickle.dump({k: np.asarray(v) for k, v in got.items()}, open(f"{OUT}/scene_frames.pkl", "wb"))
print("ok")
