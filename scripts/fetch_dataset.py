#!/usr/bin/env python3
"""
Download and unpack the dog telemetry corpus.

    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --force       # re-download even if present

The Kaggle mirror of this dataset is publicly downloadable, so this needs no
Kaggle CLI, no ~/.kaggle/kaggle.json and no username — which matters because the
newer `KGAT_…` API tokens do not slot into the classic CLI's credential file
without also knowing the account name.

If KAGGLE_API_TOKEN is set in .env it is sent as a bearer token anyway, so this
keeps working if the mirror is ever made private.

Resumes a partial download with a Range request, verifies the archive before
unpacking, and reports what landed.

Source: Vehkaoja et al., "Description of Movement Sensor Dataset for Dog
Behavior Classification", Data in Brief, 2022, University of Helsinki.
Mirrored on Kaggle as benjamingray44/inertial-data-for-dog-behaviour-classification.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from _common import DATA_DIR, die, header, info, load_env, ok, warn  # noqa: E402

SLUG = "benjamingray44/inertial-data-for-dog-behaviour-classification"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
EXPECTED = ["DogMoveData.csv", "DogInfo.csv"]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def download(dest: Path, force: bool) -> Path:
    load_env()
    headers = {}
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        info("sending KAGGLE_API_TOKEN as a bearer token")

    # Size via a streaming GET that is closed immediately, NOT via HEAD:
    # Kaggle's download endpoint answers HEAD with 404 and GET with 200, so a
    # HEAD probe reports the mirror as missing when it is perfectly fine.
    probe = requests.get(URL, headers=headers, stream=True,
                         allow_redirects=True, timeout=60)
    total = int(probe.headers.get("Content-Length", 0))
    status = probe.status_code
    probe.close()
    if status != 200:
        die(f"GET {URL} returned {status}. The mirror may have gone private; "
            f"set KAGGLE_API_TOKEN in .env, or download by hand.")
    info(f"remote archive: {human(total)}")

    if dest.exists() and not force:
        have = dest.stat().st_size
        if have == total:
            ok(f"{dest.name} already complete ({human(have)}) — skipping download")
            return dest
        if have < total:
            info(f"resuming from {human(have)} ({100 * have / total:.1f}%)")
            headers["Range"] = f"bytes={have}-"

    mode = "ab" if "Range" in headers else "wb"
    got = dest.stat().st_size if (mode == "ab" and dest.exists()) else 0
    t0 = time.time()
    last = 0.0

    with requests.get(URL, headers=headers, stream=True, timeout=120) as r:
        if r.status_code not in (200, 206):
            die(f"GET returned {r.status_code}: {r.text[:200]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 0.5:
                    last = now
                    el = now - t0
                    rate = (got / el) if el else 0
                    pct = 100 * got / total if total else 0
                    eta = (total - got) / rate if rate else 0
                    print(f"    {human(got)} / {human(total)}  {pct:5.1f}%  "
                          f"{human(rate)}/s  eta {eta / 60:4.1f} min",
                          end="\r", flush=True)
    print(" " * 90, end="\r")
    ok(f"downloaded {human(dest.stat().st_size)} in {(time.time() - t0) / 60:.1f} min")
    return dest


def unpack(archive: Path, out: Path) -> None:
    header("Verifying the archive")
    if not zipfile.is_zipfile(archive):
        die(f"{archive} is not a valid zip. Delete it and re-run with --force.")
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            die(f"archive is corrupt at {bad}. Delete it and re-run with --force.")
        names = z.namelist()
        ok(f"archive is valid, {len(names)} entries")
        for n in names:
            i = z.getinfo(n)
            print(f"    {n:<40} {human(i.file_size)}")

        header("Unpacking")
        out.mkdir(parents=True, exist_ok=True)
        z.extractall(out)
    ok(f"extracted to {out}")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    if missing:
        warn(f"expected files not found after extraction: {missing}")
        warn(f"present: {[p.name for p in out.iterdir()]}")
    else:
        for f in EXPECTED:
            p = out / f
            print(f"    {f:<24} {human(p.stat().st_size)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the dog telemetry corpus.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the .zip after extracting (default: keep)")
    args = ap.parse_args()

    out = Path(args.data_dir)
    archive = out / "inertial-data-for-dog-behaviour-classification.zip"

    header("Fetching the corpus")
    print(f"  source : {SLUG}")
    print(f"  dest   : {out}")
    download(archive, args.force)
    unpack(archive, out)

    header("Next")
    print("""  python scripts/profile_dataset.py     # GATE A — do not skip it
  python scripts/load_raw.py            # 10.6M rows -> Snowflake
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
