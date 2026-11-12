#!/usr/bin/env python3
"""LeRobot v3 数据集预览与上传前检查（只读）。

用法:
    python server.py [数据集目录或其上级目录 ...] [--port 8899] [--host 0.0.0.0]

只依赖 numpy 和 pyarrow（lerobot 环境里都有）；有 ffprobe 时额外核对视频文件时长。
程序不会改动数据集目录里的任何文件；网页上的「标记」和「分界线」存在 ~/.cache/lerobot_preview/ 下。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
MARKS_DIR = Path.home() / ".cache" / "lerobot_preview" / "marks"
SEAMS_DIR = Path.home() / ".cache" / "lerobot_preview" / "seams"
SEAM_HINTS: Path | None = None  # aug01 analysis 目录（--seam-hints），里面有每个回合的手臂活动区域
INDEX_COLS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
NUMERIC_DTYPES = {"float32", "float64", "float16", "int64", "int32", "int16", "int8", "uint8", "bool"}

ROOTS: list[Path] = []
_lock = threading.Lock()
_check_cache: dict[str, tuple[tuple, dict]] = {}
_probe_cache: dict[tuple, dict | None] = {}


# ----------------------------------------------------------------------------- 数据集发现


def find_datasets() -> list[Path]:
    found: list[Path] = []
    for root in ROOTS:
        if (root / "meta" / "info.json").is_file():
            found.append(root)
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            p = Path(dirpath)
            depth = len(p.relative_to(root).parts)
            if (p / "meta" / "info.json").is_file():
                found.append(p)
                dirnames[:] = []
                continue
            if depth >= 3:
                dirnames[:] = []
            dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")]
    seen, out = set(), []
    for p in found:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def empty_dirs() -> list[str]:
    """根目录下没有 meta/info.json 的子目录（例如录到一半的空数据集），列出来提醒。"""
    out = []
    for root in ROOTS:
        if root.is_dir() and not (root / "meta" / "info.json").is_file():
            for d in sorted(root.iterdir()):
                if d.is_dir() and not d.name.startswith(".") and not (d / "meta" / "info.json").is_file():
                    if not any((d / sub / "meta" / "info.json").is_file() for sub in _subdirs(d)):
                        out.append(str(d))
    return out


def _subdirs(d: Path):
    try:
        return [x.name for x in d.iterdir() if x.is_dir()]
    except OSError:
        return []


def resolve_dataset(ds: str) -> Path:
    p = Path(ds).resolve()
    if p not in find_datasets():
        raise ApiError(404, f"不是已扫描到的数据集: {ds}")
    return p


def dir_size(p: Path) -> tuple[int, int]:
    total, n = 0, 0
    for dirpath, dirnames, filenames in os.walk(p):
        for f in filenames:
            try:
                total += os.stat(os.path.join(dirpath, f)).st_size
                n += 1
            except OSError:
                pass
    return total, n


def signature(p: Path) -> tuple:
    sig = []
    for sub in ("meta", "data", "videos"):
        for dirpath, _, filenames in os.walk(p / sub):
            for f in filenames:
                try:
                    st = os.stat(os.path.join(dirpath, f))
                    sig.append((dirpath, f, st.st_mtime_ns, st.st_size))
                except OSError:
                    pass
    return tuple(sorted(sig))


# ----------------------------------------------------------------------------- 读元数据


class ApiError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status
        self.msg = msg


def load_info(p: Path) -> dict:
    with open(p / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(p: Path) -> dict[int, str]:
    f = p / "meta" / "tasks.parquet"
    if not f.is_file():
        return {}
    t = pq.read_table(f)
    names = t.column_names
    str_col = next((c for c in names if pa.types.is_string(t.schema.field(c).type) or pa.types.is_large_string(t.schema.field(c).type)), None)
    if str_col is None or "task_index" not in names:
        return {}
    return {int(i): s for i, s in zip(t.column("task_index").to_pylist(), t.column(str_col).to_pylist())}


def load_episodes(p: Path) -> list[dict]:
    files = sorted((p / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    rows: list[dict] = []
    for f in files:
        try:
            names = pq.read_schema(f).names
            cols = [c for c in names if not c.startswith("stats/")]
            rows.extend(pq.read_table(f, columns=cols).to_pylist())
        except (pa.ArrowInvalid, OSError) as e:
            raise ApiError(409, f"{f.relative_to(p)} 读不了。这个数据集可能还在生成（lerobot 全部写完才补全这个文件），"
                                f"等生成结束后点「刷新」；如果没有在生成，就是文件损坏了。（{e}）") from None
    rows.sort(key=lambda r: r.get("episode_index", -1))
    return rows


def video_keys(info: dict) -> list[str]:
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def numeric_features(info: dict) -> list[str]:
    out = []
    for k, v in info.get("features", {}).items():
        if k in INDEX_COLS or v.get("dtype") not in NUMERIC_DTYPES:
            continue
        out.append(k)
    return out


def feature_names(info: dict, key: str) -> list[str]:
    ft = info["features"][key]
    shape = ft.get("shape") or [1]
    n = int(np.prod(shape))
    names = ft.get("names")
    if isinstance(names, dict):  # 老格式 {"motors": [...]}
        names = next(iter(names.values()), None)
    if isinstance(names, list) and len(names) == n and all(isinstance(x, str) for x in names):
        return names
    return [f"{key}[{i}]" for i in range(n)]


def fmt_path(tpl: str, **kw) -> str:
    return tpl.format(**kw)


def data_file_of(info: dict, ep: dict) -> str:
    return fmt_path(info["data_path"], chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])


def video_file_of(info: dict, ep: dict, key: str) -> str:
    return fmt_path(
        info["video_path"], video_key=key,
        chunk_index=ep[f"videos/{key}/chunk_index"], file_index=ep[f"videos/{key}/file_index"],
    )


def col_to_numpy(col: pa.ChunkedArray, n_rows: int) -> np.ndarray:
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_fixed_size_list(arr.type) or pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
        dim = len(flat) // n_rows if n_rows else 0
        return flat.reshape(n_rows, dim)
    return arr.to_numpy(zero_copy_only=False).astype(np.float64).reshape(n_rows, 1)


def ffprobe(path: Path) -> dict | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key in _probe_cache:
        return _probe_cache[key]
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=duration:stream=codec_name,width,height,avg_frame_rate,pix_fmt", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        j = json.loads(out.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
        res = {
            "duration": float(j.get("format", {}).get("duration", "nan")),
            "codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
            "pix_fmt": s.get("pix_fmt"),
            "fps": float(num) / float(den or 1) if float(den or 1) else None,
            "error": out.stderr.strip()[:300] or None,
        }
    except Exception as e:  # noqa: BLE001
        res = {"error": str(e)}
    _probe_cache[key] = res
    return res


# ----------------------------------------------------------------------------- API: 概览


def api_datasets() -> dict:
    out = []
    for p in find_datasets():
        item = {"path": str(p), "name": p.name}
        try:
            info = load_info(p)
            item.update(
                version=info.get("codebase_version"), episodes=info.get("total_episodes"),
                frames=info.get("total_frames"), fps=info.get("fps"), robot_type=info.get("robot_type"),
            )
        except Exception as e:  # noqa: BLE001
            item["error"] = f"info.json 读不了: {e}"
        for f in sorted((p / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
            try:
                pq.read_schema(f)
            except Exception:  # noqa: BLE001  正在生成的数据集：lerobot 最后才写完这个文件
                item["incomplete"] = True
                break
        out.append(item)
    return {"roots": [str(r) for r in ROOTS], "datasets": out, "empty_dirs": empty_dirs()}


def api_dataset(p: Path) -> dict:
    info = load_info(p)
    tasks = load_tasks(p)
    eps = load_episodes(p) if str(info.get("codebase_version", "")).startswith("v3") else []
    vkeys = video_keys(info)
    fps = info.get("fps") or 30
    episodes = []
    for e in eps:
        item = {
            "index": e.get("episode_index"), "length": e.get("length"),
            "tasks": list(e.get("tasks") or []),
            "data_file": data_file_of(info, e) if "data/chunk_index" in e else None,
            "from_index": e.get("dataset_from_index"), "to_index": e.get("dataset_to_index"),
            "videos": {},
        }
        for k in vkeys:
            if f"videos/{k}/chunk_index" in e:
                item["videos"][k] = {
                    "file": video_file_of(info, e, k),
                    "from": e.get(f"videos/{k}/from_timestamp"), "to": e.get(f"videos/{k}/to_timestamp"),
                }
        episodes.append(item)
    size, nfiles = dir_size(p)
    stats_keys = []
    if (p / "meta" / "stats.json").is_file():
        try:
            stats_keys = list(json.loads((p / "meta" / "stats.json").read_text()).keys())
        except Exception:  # noqa: BLE001
            stats_keys = []
    numeric = numeric_features(info)
    return {
        "path": str(p), "name": p.name, "info": info, "tasks": tasks, "episodes": episodes,
        "video_keys": vkeys, "numeric_features": numeric,
        "names": {k: feature_names(info, k) for k in numeric},
        "size": size, "n_files": nfiles, "fps": fps, "stats_keys": stats_keys,
        "has_readme": (p / "README.md").is_file(),
        "marks": load_marks(p),
        "seams": load_seams(p), "seams_file": str(seams_file(p)),
        "seam_hints": hints_dir(p) is not None,
        "aug": load_aug(p), "arm_near_px": ARM_NEAR_PX, "arm_risk_px": ARM_RISK_PX,
    }


# ----------------------------------------------------------------------------- API: 单个回合


def api_episode(p: Path, ep_idx: int) -> dict:
    info = load_info(p)
    eps = {e["episode_index"]: e for e in load_episodes(p)}
    if ep_idx not in eps:
        raise ApiError(404, f"没有回合 {ep_idx}")
    e = eps[ep_idx]
    f = p / data_file_of(info, e)
    if not f.is_file():
        raise ApiError(404, f"数据文件不存在: {f.relative_to(p)}")
    numeric = numeric_features(info)
    cols = [c for c in numeric + ["timestamp", "frame_index", "index"] if c in pq.read_schema(f).names]
    t = pq.read_table(f, columns=cols + ["episode_index"], filters=[("episode_index", "=", ep_idx)])
    n = t.num_rows
    out: dict = {"episode_index": ep_idx, "n": n, "series": {}}
    for c in cols:
        a = col_to_numpy(t.column(c), n)
        if c in ("timestamp", "frame_index", "index"):
            out[c] = clean(a[:, 0])
        else:
            out["series"][c] = [clean(a[:, j]) for j in range(a.shape[1])]
    return out


def clean(a: np.ndarray) -> list:
    a = np.asarray(a, dtype=np.float64)
    if np.isfinite(a).all():
        return np.round(a, 6).tolist()
    return [round(float(x), 6) if math.isfinite(x) else None for x in a]


# ----------------------------------------------------------------------------- API: 上传前检查


class Report:
    def __init__(self):
        self.items: list[dict] = []
        self.per_ep: dict[int, list[dict]] = {}

    def add(self, level: str, title: str, detail: str = "", episodes=None, ep_details: dict | None = None):
        eps = sorted({int(x) for x in (episodes or [])})
        item = {"level": level, "title": title, "detail": detail, "episodes": eps}
        self.items.append(item)
        for ep in eps:
            self.per_ep.setdefault(ep, []).append(
                {"level": level, "title": title, "detail": (ep_details or {}).get(ep, "")}
            )


def api_check(p: Path) -> dict:
    sig = signature(p)
    with _lock:
        hit = _check_cache.get(str(p))
        if hit and hit[0] == sig:
            return hit[1]
    t0 = time.time()
    r = Report()
    try:
        run_checks(p, r)
        aug_checks(p, r)
    except Exception as e:  # noqa: BLE001
        r.add("error", "检查程序自己出错了，下面的结果可能不完整", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    order = {"error": 0, "warn": 1, "info": 2, "ok": 3}
    r.items.sort(key=lambda x: order.get(x["level"], 9))
    res = {
        "items": r.items,
        "per_episode": {str(k): v for k, v in r.per_ep.items()},
        "counts": {lv: sum(1 for x in r.items if x["level"] == lv) for lv in order},
        "seconds": round(time.time() - t0, 2),
        "ffprobe": bool(shutil.which("ffprobe")),
    }
    with _lock:
        _check_cache[str(p)] = (sig, res)
    return res


def run_checks(p: Path, r: Report):  # noqa: C901
    # ---------- info.json
    try:
        info = load_info(p)
    except Exception as e:  # noqa: BLE001
        r.add("error", "meta/info.json 读不了", str(e))
        return
    ver = str(info.get("codebase_version", ""))
    if not ver.startswith("v3"):
        r.add("error", f"数据集格式是 {ver or '未知'}，本工具只检查 v3.0",
              "lerobot 0.6.x 只认 v3.0。旧版可以用 lerobot 自带的 convert_dataset_v21_to_v30.py 转换。")
        return
    r.add("ok", f"格式版本 {ver}")
    missing = [k for k in ("fps", "features", "data_path", "total_episodes", "total_frames", "chunks_size") if k not in info]
    if missing:
        r.add("error", "info.json 缺少必要字段", "、".join(missing))
        return
    if video_keys(info) and "video_path" not in info:
        r.add("error", "info.json 有视频特征但没有 video_path")
        return
    if not info.get("robot_type"):
        r.add("warn", "info.json 里没有 robot_type", "Hub 上的数据集卡片和筛选会用到它。")
    fps = float(info["fps"])

    # ---------- tasks / episodes 元数据
    tasks = load_tasks(p)
    if not (p / "meta" / "tasks.parquet").is_file():
        r.add("error", "缺少 meta/tasks.parquet")
    elif not tasks:
        r.add("error", "meta/tasks.parquet 里没有读到任务")
    if info.get("total_tasks") is not None and tasks and info["total_tasks"] != len(tasks):
        r.add("error", "任务数不一致", f"info.json 写 {info['total_tasks']} 个，tasks.parquet 里有 {len(tasks)} 个")

    try:
        eps = load_episodes(p)
    except Exception as e:  # noqa: BLE001
        r.add("error", "meta/episodes 读不了", str(e))
        return
    if not eps:
        r.add("error", "meta/episodes 里没有任何回合", "数据集可能还没录完或者录制时没有正常保存（finalize）。")
        return
    ep_ids = [e["episode_index"] for e in eps]
    n_eps = len(eps)
    if n_eps != info["total_episodes"]:
        r.add("error", "回合数不一致", f"info.json 写 {info['total_episodes']} 条，meta/episodes 里有 {n_eps} 条")
    dup = sorted({x for x in ep_ids if ep_ids.count(x) > 1})
    if dup:
        r.add("error", "回合编号重复", f"重复的编号: {dup}", dup)
    if sorted(ep_ids) != list(range(n_eps)):
        gaps = sorted(set(range(max(ep_ids) + 1)) - set(ep_ids))
        r.add("error", "回合编号不是从 0 开始连续的", f"缺少: {gaps[:30]}{' …' if len(gaps) > 30 else ''}")
    else:
        r.add("ok", f"{n_eps} 个回合，编号 0–{n_eps - 1} 连续")

    lengths = np.array([e.get("length") or 0 for e in eps])
    if int(lengths.sum()) != info["total_frames"]:
        r.add("error", "总帧数不一致", f"info.json 写 {info['total_frames']}，各回合 length 加起来是 {int(lengths.sum())}")

    splits = info.get("splits") or {}
    if not splits:
        r.add("warn", "info.json 里没有 splits")
    elif splits.get("train") and splits["train"] != f"0:{n_eps}" and len(splits) == 1:
        r.add("warn", "splits 和回合数对不上", f"train = {splits['train']}，回合数是 {n_eps}")

    task_names = set(tasks.values())
    bad_task = {}
    for e in eps:
        unknown = [t for t in (e.get("tasks") or []) if t not in task_names]
        if unknown:
            bad_task[e["episode_index"]] = "；".join(unknown)
    if bad_task:
        r.add("error", "回合元数据里的任务不在 tasks.parquet 里", "", bad_task.keys(), bad_task)
    if tasks:
        cnt = {t: 0 for t in tasks.values()}
        for e in eps:
            for t in e.get("tasks") or []:
                if t in cnt:
                    cnt[t] += 1
        unused = [t for t, c in cnt.items() if c == 0]
        if unused:
            r.add("warn", "有任务没有被任何回合用到", "；".join(unused))

    # ---------- 文件引用
    referenced: set[str] = set()
    data_files: dict[str, list[dict]] = {}
    for e in eps:
        df = data_file_of(info, e)
        data_files.setdefault(df, []).append(e)
        referenced.add(df)
    missing_data = {df: es for df, es in data_files.items() if not (p / df).is_file()}
    if missing_data:
        bad = [e["episode_index"] for es in missing_data.values() for e in es]
        r.add("error", "数据文件缺失", "、".join(list(missing_data)[:10]), bad)

    vkeys = video_keys(info)
    video_files: dict[str, list[tuple[dict, str]]] = {}
    for e in eps:
        for k in vkeys:
            if f"videos/{k}/chunk_index" not in e:
                r.add("error", f"回合元数据缺少视频 {k} 的位置信息", "", [e["episode_index"]])
                continue
            vf = video_file_of(info, e, k)
            video_files.setdefault(vf, []).append((e, k))
            referenced.add(vf)

    existing = set()
    for sub in ("data", "videos"):
        for f in (p / sub).rglob("*"):
            if f.is_file():
                existing.add(str(f.relative_to(p)))
    extra = sorted(existing - referenced)
    if extra:
        r.add("warn", f"有 {len(extra)} 个文件没有被任何回合引用",
              "上传时会一起传上去，白占空间（可能是删回合或中断录制留下的）：\n" + "\n".join(extra[:20]) + ("\n…" if len(extra) > 20 else ""))
    missing_v = [vf for vf in video_files if not (p / vf).is_file()]
    if missing_v:
        bad = sorted({e["episode_index"] for vf in missing_v for e, _ in video_files[vf]})
        r.add("error", "视频文件缺失", "\n".join(missing_v[:10]), bad)

    other = []
    for f in p.iterdir():
        if f.name not in ("meta", "data", "videos", "README.md", ".gitattributes"):
            other.append(f.name + ("/" if f.is_dir() else ""))
    if "images/" in other:
        other.remove("images/")
        r.add("warn", "数据集目录里有 images/ 目录",
              "这是 lerobot 录制时暂存图片帧的地方，编码成视频后会删掉。正在录制时出现是正常的，等录完再检查；"
              "录完了还在，说明有回合没编码完或者录制中断了。不要把它传上去。")
    if other:
        r.add("warn", "数据集目录里有额外的文件或目录", "上传整个目录时会一起传上去: " + "、".join(sorted(other)))
    hidden = [str(f.relative_to(p)) for f in p.rglob(".*") if f.name not in (".gitattributes",)]
    if hidden:
        r.add("warn", "有隐藏文件", "\n".join(hidden[:10]))

    # ---------- 视频片段
    seg_bad, seg_det = [], {}
    for vf, uses in video_files.items():
        for e, k in uses:
            fr, to = e.get(f"videos/{k}/from_timestamp"), e.get(f"videos/{k}/to_timestamp")
            if fr is None or to is None:
                continue
            nfr = (to - fr) * fps
            if abs(nfr - e["length"]) > 1.01:
                seg_bad.append(e["episode_index"])
                seg_det[e["episode_index"]] = f"{k.split('.')[-1]}: 视频片段 {nfr:.1f} 帧，数据 {e['length']} 帧"
    if seg_bad:
        r.add("error", "视频片段长度和数据帧数对不上", "按 (to_timestamp − from_timestamp) × fps 算。", seg_bad, seg_det)

    if shutil.which("ffprobe"):
        short_bad, short_det, fmt_warn = [], {}, []
        for vf, uses in video_files.items():
            if not (p / vf).is_file():
                continue
            pr = ffprobe(p / vf)
            if not pr or pr.get("duration") is None or not math.isfinite(pr.get("duration", float("nan"))):
                bad = sorted({e["episode_index"] for e, _ in uses})
                r.add("error", f"视频文件解析失败: {vf}", (pr or {}).get("error") or "", bad)
                continue
            k = uses[0][1]
            vinfo = info["features"][k].get("info", {}) or {}
            exp = (vinfo.get("video.codec"), vinfo.get("video.width"), vinfo.get("video.height"))
            got = (pr.get("codec"), pr.get("width"), pr.get("height"))
            if exp[0] and (exp[0] != got[0] and not (exp[0] == "av1" and got[0] in ("av1", "libdav1d"))) or \
               (exp[1] and exp[1] != got[1]) or (exp[2] and exp[2] != got[2]):
                fmt_warn.append(f"{vf}: info.json 写 {exp[0]} {exp[1]}×{exp[2]}，实际 {got[0]} {got[1]}×{got[2]}")
            if pr.get("fps") and abs(pr["fps"] - fps) > 0.5:
                fmt_warn.append(f"{vf}: 视频帧率 {pr['fps']:.2f}，数据集 fps {fps:g}")
            for e, kk in uses:
                to = e.get(f"videos/{kk}/to_timestamp")
                if to is not None and to > pr["duration"] + 1.5 / fps:
                    short_bad.append(e["episode_index"])
                    short_det[e["episode_index"]] = f"{kk.split('.')[-1]}: 需要到 {to:.2f}s，{vf.split('/')[-1]} 只有 {pr['duration']:.2f}s"
        if short_bad:
            r.add("error", "视频文件比元数据记录的短", "读到这些回合的最后几帧时会报错。", short_bad, short_det)
        if fmt_warn:
            r.add("warn", "视频编码或尺寸和 info.json 不一致", "\n".join(fmt_warn[:15]))
        if not short_bad and not fmt_warn and video_files:
            r.add("ok", f"{len(video_files)} 个视频文件都能解析，时长覆盖所有回合")
        browser_bad = sorted({vf for vf in video_files if (pr := _probe_cache.get(_pkey(p / vf))) and pr.get("codec") not in ("h264", "vp9", "av1", None)})
        if browser_bad:
            r.add("info", "部分视频浏览器可能放不了", "编码不是 h264/vp9/av1，预览页可能黑屏，但不影响数据本身: " + "、".join(browser_bad[:5]))
    else:
        r.add("info", "本机没有 ffprobe，跳过了视频文件时长核对")

    # ---------- 逐帧数据
    numeric = [k for k in numeric_features(info)]
    need = numeric + ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    tables = []
    for df in sorted(data_files):
        if (p / df).is_file():
            names = pq.read_schema(p / df).names
            miss = [c for c in need if c not in names]
            if miss:
                r.add("error", f"数据文件缺少列: {df}", "、".join(miss), [e["episode_index"] for e in data_files[df]])
            tables.append(pq.read_table(p / df, columns=[c for c in need if c in names]))
    if not tables:
        return
    try:
        t = pa.concat_tables(tables, promote_options="default")
    except TypeError:
        t = pa.concat_tables(tables, promote=True)
    n = t.num_rows
    if n != info["total_frames"] and not missing_data:
        r.add("error", "数据文件总行数和 total_frames 不一致", f"数据 {n} 行，info.json 写 {info['total_frames']}")
    epcol = col_to_numpy(t.column("episode_index"), n)[:, 0].astype(np.int64)
    order = np.argsort(epcol, kind="stable")
    epcol = epcol[order]
    arrs = {}
    for c in need:
        if c in t.column_names and c != "episode_index":
            arrs[c] = col_to_numpy(t.column(c), n)[order]
    extra_eps = sorted(set(np.unique(epcol).tolist()) - set(ep_ids))
    if extra_eps:
        r.add("error", "数据文件里有元数据中不存在的回合", f"{extra_eps[:20]}")

    bounds = {}
    uniq, starts, counts = np.unique(epcol, return_index=True, return_counts=True)
    for u, s, c in zip(uniq, starts, counts):
        bounds[int(u)] = (int(s), int(s + c))

    # 全数据集的每维范围，用来定相对阈值
    rng = {}
    for c in numeric:
        if c in arrs:
            a = arrs[c]
            with np.errstate(invalid="ignore"):
                rng[c] = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)

    names_of = {c: feature_names(info, c) for c in numeric}
    len_bad, len_det = [], {}
    idx_bad, idx_det = [], {}
    fi_bad, fi_det = [], {}
    ts_bad, ts_det = [], {}
    nan_bad, nan_det = [], {}
    frozen_bad, frozen_det = [], {}
    jump_bad, jump_det = [], {}
    const_bad, const_det = [], {}
    taskidx_bad = []
    hashes: dict[str, list[int]] = {}
    state_key = "observation.state" if "observation.state" in arrs else next((c for c in numeric if c.startswith("observation")), None)
    action_key = "action" if "action" in arrs else None
    tol = 1e-4  # 和 lerobot 读数据时的 tolerance_s 默认值一致
    gripper_dims = [j for j, nm in enumerate(names_of.get(action_key, [])) if "gripper" in nm.lower()] if action_key else []
    for e in eps:
        ei = e["episode_index"]
        if ei not in bounds:
            if (p / data_file_of(info, e)).is_file():
                r.add("error", "回合在数据文件里没有任何行", "", [ei])
            continue
        s, en = bounds[ei]
        L = en - s
        if L != e["length"]:
            len_bad.append(ei)
            len_det[ei] = f"元数据 {e['length']} 帧，数据 {L} 行"
        if "index" in arrs:
            idx = arrs["index"][s:en, 0]
            fr, to = e.get("dataset_from_index"), e.get("dataset_to_index")
            if fr is not None and (idx[0] != fr or idx[-1] != to - 1 or np.any(np.diff(idx) != 1)):
                idx_bad.append(ei)
                idx_det[ei] = f"元数据 {fr}–{to - 1}，数据 {int(idx[0])}–{int(idx[-1])}"
        if "frame_index" in arrs:
            fi = arrs["frame_index"][s:en, 0]
            if not np.array_equal(fi, np.arange(L)):
                fi_bad.append(ei)
                fi_det[ei] = f"应为 0–{L - 1}，实际 {int(fi.min())}–{int(fi.max())}"
        if "timestamp" in arrs:
            ts = arrs["timestamp"][s:en, 0]
            d = np.diff(ts)
            badd = np.where(np.abs(d - 1 / fps) > tol)[0]
            if len(badd) or (L and abs(ts[0]) > tol):
                ts_bad.append(ei)
                if len(badd):
                    j = int(badd[0])
                    ts_det[ei] = f"{len(badd)} 处间隔不是 1/fps，第一处在第 {j} 帧: {ts[j]:.4f}→{ts[j + 1]:.4f}"
                else:
                    ts_det[ei] = f"第一帧时间戳是 {ts[0]:.4f}，不是 0"
        if "task_index" in arrs and tasks:
            ti = np.unique(arrs["task_index"][s:en, 0]).astype(int)
            if any(int(x) not in tasks for x in ti):
                taskidx_bad.append(ei)
        msgs_nan = []
        for c in numeric:
            if c not in arrs:
                continue
            a = arrs[c][s:en]
            bad_cols = np.where(~np.isfinite(a).all(axis=0))[0]
            if len(bad_cols):
                msgs_nan.append(f"{c}: " + "、".join(names_of[c][j] for j in bad_cols[:5]))
        if msgs_nan:
            nan_bad.append(ei)
            nan_det[ei] = "；".join(msgs_nan)
        if state_key and L > 1:
            st = arrs[state_key][s:en]
            same = np.all(st[1:] == st[:-1], axis=1)
            run_len, best, best_end = 0, 0, 0
            for i, v in enumerate(same):
                run_len = run_len + 1 if v else 0
                if run_len > best:
                    best, best_end = run_len, i + 1
            if best + 1 >= fps:
                b0 = best_end - best
                moving = False
                if action_key and arrs[action_key].shape[1] == st.shape[1]:
                    # 状态一般比动作晚几帧，去掉最后 5 帧；动作在这段里的变化要超过该维全数据集范围的 5% 才算「在动」
                    act = arrs[action_key][s + b0:s + max(b0 + 1, best_end - 5)]
                    with np.errstate(invalid="ignore"):
                        span = (np.nanmax(act, axis=0) - np.nanmin(act, axis=0)) / np.where(rng[action_key] > 0, rng[action_key], np.inf)
                    moving = bool(np.nanmax(span, initial=0) > 0.05)
                if moving:
                    frozen_bad.append(ei)
                    frozen_det[ei] = f"第 {b0}–{best_end} 帧（{(best + 1) / fps:.1f} 秒）{state_key} 完全不变，但 action 在动：可能是状态话题断了"
            hashes.setdefault(hashlib.md5(np.ascontiguousarray(st).tobytes()).hexdigest(), []).append(ei)
            if L >= fps:
                with np.errstate(invalid="ignore"):
                    const = np.where((np.nanmax(st, axis=0) - np.nanmin(st, axis=0) == 0) & (rng[state_key] > 0))[0]
                if len(const):
                    const_bad.append(ei)
                    const_det[ei] = "整个回合不变: " + "、".join(names_of[state_key][j] for j in const[:6])
        if action_key and L > 1:
            act = arrs[action_key][s:en]
            with np.errstate(invalid="ignore"):
                rel = np.abs(np.diff(act, axis=0)) / np.where(rng[action_key] > 0, rng[action_key], np.inf)
            rel[:, gripper_dims] = 0  # 夹爪动作本来就是开合切换，跳变是正常的
            if np.nanmax(rel, initial=0) > 0.25:
                j, dim = np.unravel_index(np.nanargmax(rel), rel.shape)
                jump_bad.append(ei)
                jump_det[ei] = (f"{names_of[action_key][dim]} 在第 {j}→{j + 1} 帧跳了 {abs(act[j + 1, dim] - act[j, dim]):.3f}"
                                f"（全数据集范围的 {rel[j, dim] * 100:.0f}%）")

    if len_bad:
        r.add("error", "回合帧数和元数据 length 不一致", "", len_bad, len_det)
    if idx_bad:
        r.add("error", "全局 index 和元数据 dataset_from/to_index 对不上", "", idx_bad, idx_det)
    if fi_bad:
        r.add("error", "frame_index 不是从 0 连续递增", "", fi_bad, fi_det)
    if ts_bad:
        r.add("error", "时间戳不均匀", f"lerobot 加载数据时按 1/fps ± {tol} 秒核对，不通过会直接报错。", ts_bad, ts_det)
    elif "timestamp" in arrs:
        r.add("ok", "所有回合的时间戳都是均匀的 1/fps，frame_index 连续")
    if taskidx_bad:
        r.add("error", "task_index 不在 tasks.parquet 里", "", taskidx_bad)
    if nan_bad:
        r.add("error", "数值里有 NaN 或 Inf", "", nan_bad, nan_det)
    if frozen_bad:
        r.add("warn", "状态长时间完全冻结而动作在变", "", frozen_bad, frozen_det)
    if const_bad:
        r.add("warn", "有关节在整个回合里读数完全不变", "别的回合里这些关节是动的。如果这个回合本来就只用了一只手可以忽略；否则可能是那只手的状态没读到。", const_bad, const_det)
    if jump_bad:
        r.add("warn", "动作有大跳变", "相邻两帧之间变化超过这一维全数据集范围的 25%（夹爪不算）。可能是遥操作跟丢后又找回来，机械臂会猛地一动。", jump_bad, jump_det)
    dups = [v for v in hashes.values() if len(v) > 1]
    if dups and (p / "meta" / "aug_manifest.json").is_file():  # 合成数据集：混合副本和原始条目共用关节数据，是预期的
        r.add("info", "有回合的状态数据完全相同（合成数据集里的混合副本，属正常）", "；".join("、".join(map(str, v)) for v in dups),
              [x for v in dups for x in v])
    elif dups:
        r.add("warn", "有回合的状态数据完全相同（疑似重复保存）", "；".join("、".join(map(str, v)) for v in dups),
              [x for v in dups for x in v])

    # 长度
    if len(lengths):
        med = float(np.median(lengths))
        short = {e["episode_index"]: f"{e['length']} 帧 = {e['length'] / fps:.1f} 秒" for e in eps if e["length"] < max(fps * 2, med / 3)}
        long_ = {e["episode_index"]: f"{e['length'] / fps:.1f} 秒，中位数 {med / fps:.1f} 秒" for e in eps if e["length"] > med * 3}
        if short:
            r.add("warn", "回合特别短", f"短于 2 秒或不到中位数（{med / fps:.1f} 秒）的三分之一，可能是误触录下的。", short.keys(), short)
        if long_:
            r.add("warn", "回合特别长", "超过中位数的 3 倍，可能是忘了结束录制。", long_.keys(), long_)

    # 全局不变的维度
    for c in numeric:
        if c in rng:
            z = np.where(rng[c] == 0)[0]
            if len(z):
                r.add("info", f"{c} 有 {len(z)} 维在整个数据集里恒定不变",
                      "、".join(names_of[c][j] for j in z) + "。如果这一维本来就不用（比如固定的关节）可以忽略；否则检查数据来源。")

    # ---------- stats.json
    sf = p / "meta" / "stats.json"
    if not sf.is_file():
        r.add("error", "缺少 meta/stats.json", "训练时归一化要用。可以用 lerobot-edit-dataset --operation.type recompute_stats 重算。")
    else:
        try:
            stats = json.loads(sf.read_text())
        except Exception as e:  # noqa: BLE001
            r.add("error", "meta/stats.json 读不了", str(e))
            stats = None
        if stats is not None:
            miss = [k for k in list(numeric) + vkeys if k not in stats]
            if miss:
                r.add("error", "stats.json 缺少特征的统计量", "、".join(miss) +
                      "\n训练时这些特征无法归一化。可以用 lerobot-edit-dataset --operation.type recompute_stats 重算。")
            stale = []
            for c in numeric:
                if c not in stats or c not in arrs:
                    continue
                a = arrs[c]
                for stat, val in (("min", np.nanmin(a, axis=0)), ("max", np.nanmax(a, axis=0)), ("mean", np.nanmean(a, axis=0))):
                    if stat not in stats[c]:
                        continue
                    sv = np.asarray(stats[c][stat], dtype=np.float64).reshape(-1)
                    if sv.shape != val.shape:
                        stale.append(f"{c}.{stat}: 维数 {sv.shape[0]}，数据 {val.shape[0]}")
                        continue
                    diff = np.abs(sv - val) / (rng[c] + 1e-6)  # 相对这一维的取值范围
                    if np.nanmax(diff) > 1e-3:
                        j = int(np.nanargmax(diff))
                        stale.append(f"{c}.{stat}[{names_of[c][j]}]: stats.json {sv[j]:.5g}，按数据算 {val[j]:.5g}")
            if stale:
                r.add("warn", "stats.json 和实际数据对不上（可能删改回合后没重算）",
                      "\n".join(stale[:12]) + ("\n…" if len(stale) > 12 else "") +
                      "\n可以用 lerobot-edit-dataset --operation.type recompute_stats 重算。")
            elif not miss:
                r.add("ok", "stats.json 齐全，和实际数据一致")

    # ---------- 杂项
    if not (p / "README.md").is_file():
        r.add("info", "没有 README.md", "用 lerobot 的 push_to_hub 上传时会自动生成数据集卡片；用 hf upload 直接传目录则不会。")
    big = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file() and f.stat().st_size > 4.5 * 1024 ** 3]
    if big:
        r.add("warn", "有超过 4.5 GB 的单个文件", "Hub 对单文件大小有限制：\n" + "\n".join(big))


def _pkey(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


# ----------------------------------------------------------------------------- 标记


def marks_file(p: Path) -> Path:
    h = hashlib.sha1(str(p).encode()).hexdigest()[:10]
    return MARKS_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', p.name)}_{h}.json"


def load_marks(p: Path) -> dict:
    f = marks_file(p)
    if f.is_file():
        try:
            return json.loads(f.read_text()).get("marks", {})
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_marks(p: Path, marks: dict) -> dict:
    clean_marks = {}
    for k, v in (marks or {}).items():
        if not str(k).isdigit() or not isinstance(v, dict):
            continue
        bad, note = bool(v.get("bad")), str(v.get("note") or "")[:2000]
        if bad or note:
            clean_marks[str(int(k))] = {"bad": bad, "note": note}
    MARKS_DIR.mkdir(parents=True, exist_ok=True)
    f = marks_file(p)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"dataset": str(p), "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "marks": clean_marks},
                              ensure_ascii=False, indent=1))
    tmp.replace(f)
    return {"marks": clean_marks, "file": str(f)}


# ----------------------------------------------------------------------------- 合成数据（tools/aug01/build.py 生成的 meta/aug_manifest.json）
ARM_NEAR_PX, ARM_RISK_PX = 30, 10  # 场景画面上两臂最近距离：< 10 像素可能相撞，10–30 像素离得较近


def load_aug(p: Path) -> dict:
    """每个输出回合的来源和碰撞筛查结果；不是合成数据集时返回 {}。"""
    f = p / "meta" / "aug_manifest.json"
    if not f.is_file():
        return {}
    try:
        eps = json.loads(f.read_text()).get("episodes", [])
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for e in eps:
        if "out_ep" not in e:
            continue
        d = e.get("arm_min_dist_px")
        risk = None if "arm_min_dist_px" not in e else "risk" if d is not None and d < ARM_RISK_PX else "near" if d is not None and d < ARM_NEAR_PX else "ok"
        out[str(e["out_ep"])] = {**e, "risk": risk}
    return out


def aug_checks(p: Path, r: Report):
    aug = load_aug(p)
    if not aug:
        return
    has = [v for v in aug.values() if v.get("risk")]
    if not has:
        r.add("info", "合成数据集", f"共 {len(aug)} 条，来源见 meta/aug_manifest.json")
        return
    det = lambda v: f"两臂最近 {v['arm_min_dist_px']} 像素（第 {v['arm_min_dist_t']} 秒），距离小于 {ARM_NEAR_PX} 像素共 {v['arm_close_s']} 秒；左臂取自 #{v.get('left')}，右臂取自 #{v.get('right')}"
    risk = {int(k): det(v) for k, v in aug.items() if v["risk"] == "risk"}
    near = {int(k): det(v) for k, v in aug.items() if v["risk"] == "near"}
    if risk:
        r.add("warn", f"两臂在画面上可能相撞（最近不到 {ARM_RISK_PX} 像素）",
              "按场景相机画面里两只手的黑色轮廓逐帧计算，没有考虑高度：画面上挨着不一定真撞，请看视频判断。", risk.keys(), risk)
    if near:
        r.add("info", f"两臂在画面上离得较近（{ARM_RISK_PX}–{ARM_NEAR_PX} 像素）", "同上，二维画面上的距离。", near.keys(), near)
    over = {int(k): f"左臂越过 {v['left_over_line_px']} 像素，右臂越过 {v['right_over_line_px']} 像素"
            for k, v in aug.items() if max(v.get("left_over_line_px") or 0, v.get("right_over_line_px") or 0) > 3}
    if over:
        r.add("info", "手臂越过了分界线（越线的部分在画面里会被截断）", "数值是越线最多时超出的像素。", over.keys(), over)


# ----------------------------------------------------------------------------- 分界线（双臂合成数据用）
# 每个回合一条直线，用场景画面上第一行和最后一行的 x 坐标 [xt, xb] 表示：线左边归左臂，右边归右臂。


def seams_file(p: Path) -> Path:
    h = hashlib.sha1(str(p).encode()).hexdigest()[:10]
    return SEAMS_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', p.name)}_{h}.json"


def load_seams(p: Path) -> dict:
    f = seams_file(p)
    if f.is_file():
        try:
            return json.loads(f.read_text()).get("seams", {})
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_seams(p: Path, seams: dict) -> dict:
    clean_seams = {}
    for k, v in (seams or {}).items():
        if str(k).isdigit() and v == "skip":  # 这一条不参与合成
            clean_seams[str(int(k))] = "skip"
            continue
        if not str(k).isdigit() or not isinstance(v, (list, tuple)) or len(v) != 2:
            continue
        xt, xb = (float(x) for x in v)
        if not (math.isfinite(xt) and math.isfinite(xb)) or not (-2000 < xt < 4000 and -2000 < xb < 4000):
            continue
        clean_seams[str(int(k))] = [round(xt, 1), round(xb, 1)]
    SEAMS_DIR.mkdir(parents=True, exist_ok=True)
    f = seams_file(p)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"dataset": str(p), "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "seams": clean_seams},
                              ensure_ascii=False, indent=1))
    tmp.replace(f)
    return {"seams": clean_seams, "file": str(f)}


def hints_dir(p: Path) -> Path | None:
    """--seam-hints 目录里的 source.txt 指向这个数据集时，才有手臂活动区域提示。"""
    if SEAM_HINTS is None or not (SEAM_HINTS / "source.txt").is_file():
        return None
    try:
        src = Path((SEAM_HINTS / "source.txt").read_text().strip()).expanduser().resolve()
    except OSError:
        return None
    return SEAM_HINTS if src == p.resolve() else None


def api_seamhint(p: Path, ep: int) -> dict:
    """这个回合左臂（加木块杯垫、绿盘）和右臂（加停靠位置、海绵杯垫、蓝盘）占过的像素，以及自动给出的分界线。"""
    d = hints_dir(p)
    f = d / f"ep_{ep:03d}.npz" if d else None
    if f is None or not f.is_file():
        raise ApiError(404, "这个回合没有手臂活动区域数据")
    z = np.load(f)
    a = z["swL"] | z["obj_wood"] | z["obj_green"]
    b = z["swR"] | z["parkR"] | z["obj_sponge"] | z["obj_blue"]
    h, w = a.shape
    st = 8
    pts = lambda m: np.argwhere(m[::st, ::st]) * st
    A, B = pts(a), pts(b)
    # 自动线：在 xt, xb ∈ [120, 560] 里找让两边离线最远的那条
    ys = np.arange(0, h, st)
    amax = np.array([a[y].nonzero()[0].max() if a[y].any() else -1e9 for y in ys], float)
    bmin = np.array([b[y].nonzero()[0].min() if b[y].any() else 1e9 for y in ys], float)
    xs = np.arange(120, 561, 4, dtype=float)
    xl = xs[:, None, None] + (xs[None, :, None] - xs[:, None, None]) * ys[None, None, :] / (h - 1)
    m = np.minimum((xl - amax).min(-1), (bmin - xl).min(-1))
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return {"W": w, "H": h, "A": A.tolist(), "B": B.tolist(), "auto": [float(xs[i]), float(xs[j])], "auto_margin": float(m[i, j])}


# ----------------------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "LeRobotPreview/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("PREVIEW_VERBOSE"):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._handle("GET")

    def do_HEAD(self):  # noqa: N802
        self._handle("HEAD")

    def do_POST(self):  # noqa: N802
        self._handle("POST")

    def _handle(self, method):
        try:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = u.path
            if path in ("/", "/index.html"):
                return self.send_file(HERE / "index.html", "text/html; charset=utf-8", method)
            if path == "/api/datasets":
                return self.send_json(api_datasets())
            if path == "/api/dataset":
                return self.send_json(api_dataset(resolve_dataset(q.get("ds", ""))))
            if path == "/api/episode":
                return self.send_json(api_episode(resolve_dataset(q.get("ds", "")), int(q.get("ep", "0"))))
            if path == "/api/check":
                return self.send_json(api_check(resolve_dataset(q.get("ds", ""))))
            if path == "/api/marks" and method == "POST":
                p = resolve_dataset(q.get("ds", ""))
                n = int(self.headers.get("Content-Length") or 0)
                if n > 2_000_000:
                    raise ApiError(413, "太大了")
                body = json.loads(self.rfile.read(n) or b"{}")
                return self.send_json(save_marks(p, body.get("marks", {})))
            if path == "/api/seams" and method == "POST":
                p = resolve_dataset(q.get("ds", ""))
                n = int(self.headers.get("Content-Length") or 0)
                if n > 2_000_000:
                    raise ApiError(413, "太大了")
                body = json.loads(self.rfile.read(n) or b"{}")
                return self.send_json(save_seams(p, body.get("seams", {})))
            if path == "/api/seamhint":
                return self.send_json(api_seamhint(resolve_dataset(q.get("ds", "")), int(q.get("ep", "0"))))
            if path == "/file":
                p = resolve_dataset(q.get("ds", ""))
                rel = unquote(q.get("p", ""))
                f = (p / rel).resolve()
                if p not in f.parents or not f.is_file() or rel.split("/")[0] not in ("videos", "meta", "data"):
                    raise ApiError(404, "文件不存在")
                return self.send_file(f, mimetypes.guess_type(f.name)[0] or "application/octet-stream", method)
            raise ApiError(404, "没有这个地址")
        except ApiError as e:
            self.send_json({"error": e.msg}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:  # noqa: BLE001
                pass

    def send_file(self, f: Path, ctype: str, method: str):
        size = f.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = HTTPStatus.OK
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if f.suffix == ".html":
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if method == "HEAD":
            return
        with open(f, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split():
            if ":" not in ip:
                ips.add(ip)
    except Exception:  # noqa: BLE001
        pass
    return sorted(ips)


def main():
    ap = argparse.ArgumentParser(description="LeRobot v3 数据集预览与上传前检查（只读）")
    ap.add_argument("paths", nargs="*", help="数据集目录，或者包含多个数据集的上级目录。默认 $HF_LEROBOT_HOME（~/.cache/huggingface/lerobot）")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0（局域网可访问）；只想本机看用 127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--seam-hints", default=os.environ.get("PREVIEW_SEAM_HINTS"),
                    help="标分界线时显示手臂活动区域：aug01 analyze.py 的输出目录（里面的 source.txt 写着对应的数据集）")
    args = ap.parse_args()
    global SEAM_HINTS
    if args.seam_hints and Path(args.seam_hints).expanduser().is_dir():
        SEAM_HINTS = Path(args.seam_hints).expanduser().resolve()
    default = os.environ.get("PREVIEW_DEFAULT_ROOTS") or os.environ.get("HF_LEROBOT_HOME") or str(Path.home() / ".cache" / "huggingface" / "lerobot")
    paths = args.paths or [x for x in default.split(":") if x]
    for x in paths:
        p = Path(x).expanduser().resolve()
        if not p.exists():
            if args.paths:
                print(f"[提示] 目录不存在，跳过: {p}", file=sys.stderr)
            continue
        ROOTS.append(p)
    if not ROOTS:
        sys.exit("没有可用的目录")
    ds = find_datasets()
    print(f"扫描目录: {', '.join(map(str, ROOTS))}")
    print(f"找到 {len(ds)} 个数据集: " + ", ".join(d.name for d in ds))
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print("在浏览器里打开:")
    if args.host in ("0.0.0.0", ""):
        print(f"  本机:   http://127.0.0.1:{args.port}/")
        for ip in lan_ips():
            print(f"  局域网: http://{ip}:{args.port}/")
    else:
        print(f"  http://{args.host}:{args.port}/")
    print("按 Ctrl+C 停止。", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
