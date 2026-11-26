"""Build a small YOLO pothole dataset from HRP4K (Zenodo 21941931, CC BY 4.0) without downloading the 12.8 GB zip.

Runs of consecutive frames are fetched with one HTTP range request each (Zenodo rate-limits many small requests), 4K frames are shrunk to 1280 px wide on the fly,
and YOLO labels (normalised, single class 0 = pothole) are copied unchanged.

    python scripts/prepare_hrp4k.py --train 2000 --valid 400 --test 400 --out data/raw/hrp4k_yolo
"""
from __future__ import annotations

import argparse
import io
import random
import struct
import time
import zlib
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

URL = "https://zenodo.org/records/21941931/files/HRP4K.zip?download=1"


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str, size: int | None = None, final: str | None = None):
        self.pos = 0
        if size is None:
            r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
            size, final = int(r.headers["Content-Length"]), r.url
        self.size, self.final = size, final

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.final, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(10):
            try:
                data = urllib.request.urlopen(req, timeout=180).read()
                break
            except Exception as e:  # Zenodo answers 429 when hit too hard: back off and retry
                if attempt == 9:
                    raise
                wait = 30 if getattr(e, "code", None) == 429 else 3
                time.sleep(wait * (1 + attempt) * (0.5 + random.random()))
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def get_range(proto: HttpFile, lo: int, hi: int) -> bytes:
    """One HTTP range request [lo, hi) with back-off (Zenodo rate-limits bursts of small requests)."""
    f = HttpFile(URL, proto.size, proto.final)
    f.seek(lo)
    return f.read(hi - lo)


def member_bytes(buf: bytes, base: int, info: zipfile.ZipInfo) -> bytes:
    rel = info.header_offset - base
    n, m = struct.unpack("<HH", buf[rel + 26: rel + 30])     # local header: name + extra field lengths
    data = buf[rel + 30 + n + m: rel + 30 + n + m + info.compress_size]
    return zlib.decompress(data, -15) if info.compress_type == zipfile.ZIP_DEFLATED else data


def save_chunk(proto, chunk, labels: dict, split_dst: str, out: Path, width: int) -> dict:
    todo = [i for i in chunk if not (out / "images" / split_dst / Path(i.filename).name).exists()]
    if not todo:
        return {"skip": len(chunk)}
    lo = todo[0].header_offset
    hi = todo[-1].header_offset + 30 + 1024 + todo[-1].compress_size
    buf = get_range(proto, lo, min(hi, proto.size))
    res = {"ok": 0, "bad": 0}
    for info in todo:
        stem = Path(info.filename).stem
        img = cv2.imdecode(np.frombuffer(member_bytes(buf, lo, info), np.uint8), cv2.IMREAD_COLOR)
        if img is None or stem not in labels:
            res["bad"] += 1
            continue
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        (out / "labels" / split_dst / f"{stem}.txt").write_text(labels[stem], encoding="utf-8")
        cv2.imwrite(str(out / "images" / split_dst / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        res["ok"] += 1
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/hrp4k_yolo")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--valid", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=50, help="consecutive frames per HTTP request")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out).resolve()

    proto = HttpFile(URL)
    infos = zipfile.ZipFile(io.BufferedReader(HttpFile(URL, proto.size, proto.final), 1 << 20)).infolist()
    rng = random.Random(a.seed)
    jobs = []
    for split_src, split_dst, k in (("train", "train", a.train), ("valid", "val", a.valid), ("test", "test", a.test)):
        (out / "images" / split_dst).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_dst).mkdir(parents=True, exist_ok=True)
        lab = sorted((i for i in infos if i.filename.startswith(f"HRP4K/{split_src}/labels/") and not i.is_dir()),
                     key=lambda i: i.header_offset)
        buf = get_range(proto, lab[0].header_offset, lab[-1].header_offset + 30 + 1024 + lab[-1].compress_size)
        labels = {}
        for info in lab:   # keep only well-formed YOLO lines (class cx cy w h)
            lines = [ln for ln in member_bytes(buf, lab[0].header_offset, info).decode("utf-8", "replace").splitlines()
                     if len(ln.split()) == 5]
            labels[Path(info.filename).stem] = "\n".join(lines) + ("\n" if lines else "")
        imgs = sorted((i for i in infos if i.filename.startswith(f"HRP4K/{split_src}/images/") and not i.is_dir()),
                      key=lambda i: i.header_offset)
        chunks = [imgs[j: j + a.chunk] for j in range(0, len(imgs), a.chunk)]
        # random whole chunks: few requests, while still spread over many recording sessions
        for c in sorted(rng.sample(chunks, min(len(chunks), -(-k // a.chunk))), key=lambda c: c[0].header_offset):
            jobs.append((c, labels, split_dst))

    tot = {"ok": 0, "skip": 0, "bad": 0}
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(save_chunk, proto, c, lab, sd, out, a.width) for c, lab, sd in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                for kk, v in f.result().items():
                    tot[kk] += v
            except Exception as e:  # keep going; a rerun resumes missing chunks
                print("error:", e, flush=True)
            print(f"chunk {i}/{len(futs)} {tot}", flush=True)

    (out / "hrp4k_yolo.yaml").write_text(
        f"# HRP4K subset (Zenodo 21941931, CC BY 4.0). Built by scripts/prepare_hrp4k.py\n"
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pothole\n",
        encoding="utf-8")
    print("wrote", out / "hrp4k_yolo.yaml")


if __name__ == "__main__":
    main()
