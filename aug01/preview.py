"""Contact sheets of a generated dataset: scene camera at several moments for a few episodes, plus wrist cameras."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from common import Source, CAMS, SCENE

src = Source(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
man = json.loads((src.root / "meta/aug_manifest.json").read_text())["episodes"]
want = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
if want is None:
    aug = [p["out_ep"] for p in man if p["kind"] != "orig"]
    want = [aug[i] for i in np.linspace(0, len(aug) - 1, min(8, len(aug))).astype(int)]
FR = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
TW, TH = 320, 240


def label(img, txt):
    img = img.copy(); cv2.rectangle(img, (0, 0), (8 + 7 * len(txt), 18), (0, 0, 0), -1)
    cv2.putText(img, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1); return img


rows, wrows = [], []
for ep in want:
    p = next(x for x in man if x["out_ep"] == ep)
    n = src.length(ep); idx = [int(f * (n - 1)) for f in FR]
    tag = f"#{ep} " + (f"src{p['src']} donor{p['donor']}" if p["kind"] == "mixr" else f"L{p['left']} R{p['right']}" if p["kind"] == "fused" else f"orig{p['src']}")
    rd = {c: src.reader(ep, c) for c in CAMS}
    tiles, wt = [], []
    for k, i in enumerate(idx):
        fr = {c: rd[c].get(i) for c in CAMS}
        t = cv2.resize(cv2.cvtColor(fr[SCENE], cv2.COLOR_RGB2BGR), (TW, TH))
        tiles.append(label(t, (tag + " " if k == 0 else "") + f"t={i / 30:.1f}s"))
        if k in (1, 3):
            for c in CAMS[:2]:
                wt.append(label(cv2.resize(cv2.cvtColor(fr[c], cv2.COLOR_RGB2BGR), (TW // 2, TH // 2)), f"{c.split('.')[-1][:5]} {i / 30:.1f}s"))
    for r in rd.values(): r.close()
    rows.append(np.hstack(tiles)); wrows.append(np.hstack(wt))
cv2.imwrite(str(out / "scene.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(str(out / "wrist.jpg"), np.vstack(wrows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("preview", out, want)
