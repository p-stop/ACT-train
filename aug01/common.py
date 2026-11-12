"""Shared helpers for building augmented versions of the 01_put_blocks_into_plate dataset."""
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

FPS = 30
H, W = 480, 640
CAMS = ["observation.images.left_wrist", "observation.images.right_wrist", "observation.images.scene_fixed"]
SCENE = "observation.images.scene_fixed"


class Source:
    """Read-only access to a LeRobot v3 dataset on disk."""

    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.eps = pd.read_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet")
        d = pd.read_parquet(self.root / "data/chunk-000/file-000.parquet")
        self.state = np.stack(d["observation.state"].values).astype(np.float32)
        self.action = np.stack(d["action"].values).astype(np.float32)
        self.ep_index = d["episode_index"].values
        self.task = pd.read_parquet(self.root / "meta/tasks.parquet").index[0]

    def n_episodes(self):
        return len(self.eps)

    def length(self, ep):
        return int(self.eps["length"][ep])

    def rows(self, ep):
        a = int(self.eps["dataset_from_index"][ep])
        return slice(a, a + self.length(ep))

    def reader(self, ep, cam):
        return FrameReader(self, ep, cam)


class FrameReader:
    """Sequential RGB frame access for one episode/camera. get(i) must be called with non-decreasing i."""

    def __init__(self, src, ep, cam):
        c = f"videos/{cam}"
        f = int(src.eps[f"{c}/file_index"][ep]); ch = int(src.eps[f"{c}/chunk_index"][ep])
        self.t0 = float(src.eps[f"{c}/from_timestamp"][ep])
        self.n = src.length(ep)
        self.cont = av.open(str(src.root / c / f"chunk-{ch:03d}/file-{f:03d}.mp4"))
        st = self.cont.streams.video[0]; st.thread_type = "AUTO"
        self.cont.seek(int(max(self.t0 - 0.5, 0) / st.time_base), stream=st, backward=True, any_frame=False)
        self.it = self.cont.decode(st)
        self.idx, self.cur = -1, None

    def get(self, i):
        i = min(max(i, 0), self.n - 1)
        if i < self.idx:
            raise ValueError("FrameReader only goes forward")
        while self.idx < i:
            fr = next(self.it)
            k = int(round((fr.time - self.t0) * FPS))
            if k < 0:
                continue
            self.cur = fr.to_ndarray(format="rgb24"); self.idx = k
        return self.cur

    def close(self):
        self.cont.close()


def phases(src, ep, thr=0.05, sustain=5):
    """Robust phase times (frame indices) from the commanded joints."""
    a = src.action[src.rows(ep)]
    vL = np.r_[0, np.linalg.norm(np.diff(a[:, 0:6], axis=0), axis=1) * FPS] > thr
    vR = np.r_[0, np.linalg.norm(np.diff(a[:, 7:13], axis=0), axis=1) * FPS] > thr
    run = lambda v: np.convolve(v.astype(int), np.ones(sustain, int), "full")[sustain - 1:len(v) + sustain - 1] >= sustain
    R_start = int(np.argmax(run(vR)))
    R_park = int(np.where(vR)[0][-1]) + 1
    sL = run(vL); sL[:R_park] = False
    L_start = int(np.argmax(sL))
    return {"R_start": R_start, "R_park": R_park, "L_start": L_start, "end": len(a) - 1}


# ---------- dividing lines: x = xt + (xb - xt) * y / (H - 1) ----------
def line_side(xt, xb, ys, xs):
    """signed distance-ish (pixels along x) of points to the line, negative = left of it."""
    return xs - (xt + (xb - xt) * ys / (H - 1))


def line_mask(xt, xb, feather=6):
    """alpha in [0,1]: 0 = left of the line (keep left source), 1 = right of it (take right source)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = line_side(xt, xb, ys, xs)
    return np.clip(d / feather + 0.5, 0, 1)[..., None].astype(np.float32)


def blend(left_img, right_img, alpha):
    return (left_img * (1 - alpha) + right_img * alpha + 0.5).astype(np.uint8)


ROWS = np.arange(0, H, 4, dtype=np.float32)
GRID = np.arange(120, 561, 4, dtype=np.float32)
_XL = GRID[:, None, None] + (GRID[None, :, None] - GRID[:, None, None]) * ROWS[None, None, :] / (H - 1)


def profile(*masks):
    """per row (every 4th): rightmost and leftmost pixel of the union of masks (-inf / +inf for empty rows)."""
    m = np.zeros((H, W), bool)
    for x in masks:
        m |= x
    r = m[::4]
    any_ = r.any(1)
    right = np.where(any_, W - 1 - np.argmax(r[:, ::-1], 1), -np.inf).astype(np.float32)
    left = np.where(any_, np.argmax(r, 1), np.inf).astype(np.float32)
    return right, left


def margin(xt, xb, A, B):
    """A, B: profiles (right, left) of what must be left / right of the line. Positive = clear by that many px."""
    xl = xt + (xb - xt) * ROWS / (H - 1)
    return float(min((xl - A[0]).min(), (B[1] - xl).min()))


def best_line(A, B):
    m = np.minimum((_XL - A[0]).min(-1), (B[1] - _XL).min(-1))
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return float(m[i, j]), (int(GRID[i]), int(GRID[j]))


def mask_points(mask, stride=4):
    ys, xs = np.where(mask[::stride, ::stride])
    return np.stack([ys * stride, xs * stride], 1).astype(np.float32)


def rgb_encoder():
    from lerobot.configs.video import RGBEncoderConfig
    return RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast", g=2)


def user_features(src):
    out = {}
    for k, v in src.info["features"].items():
        if k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        out[k] = {kk: vv for kk, vv in v.items() if kk != "info"}
    return out
