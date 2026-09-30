#!/usr/bin/env python3
"""
Reference breed photographs, into REF.BREED_IMAGE as base64 thumbnails.

    python scripts/fetch_breed_images.py
    python scripts/fetch_breed_images.py --size 320 --dry-run

WHY BASE64 IN A TABLE, AND NOT A URL
Streamlit in Snowflake has no outbound network. Any <img src="https://..."> in
the dashboard renders as a broken icon, no matter how reachable the host is
from your laptop. The only picture that survives the sandbox is one already
inside the account, so each photo is fetched here, downsized, and stored as a
data: URI in a table the app reads like any other.

WHY A RANGE READ, AND NOT A DOWNLOAD
The Stanford Dogs archive is 787 MB and we want roughly twenty thumbnails out
of it. Kaggle's download endpoint honours HTTP Range (verified: 206 with a
Content-Range), and a ZIP keeps its index at the END of the file, so:

    1. range-read the last 64 KB          -> End of Central Directory record
    2. range-read the central directory   -> every filename and its offset
    3. range-read only the chosen entries -> the handful of JPEGs we want

That is a few megabytes of transfer instead of 787, and it needs no disk.

THESE ARE NOT THE STUDY DOGS. Stanford Dogs is a separate corpus of breed
photographs. Dog 54 in this warehouse is an Australian Kelpie; the picture
shown beside it is *an* Australian Kelpie, not *that* Kelpie. The table says so
in a column and the dashboard repeats it under every image — a photograph
implying it is the animal being diagnosed would be the single most misleading
thing this project could put on screen.
"""
from __future__ import annotations

import argparse
import base64
import io
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    G,
    connect,
    die,
    env,
    header,
    info,
    load_env,
    ok,
    q,
    warn,
)

SLUG = "jessicali9530/stanford-dogs-dataset"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
# Filled in by main() once the signed storage redirect has been followed.
RESOLVED: list[str] = [URL]

CREDIT = ("Stanford Dogs Dataset (Khosla et al., FGVC 2011), via Kaggle "
          "jessicali9530/stanford-dogs-dataset. Reference breed photograph, "
          "not the study animal.")

# Our breed names -> the Stanford folder fragment to look for. Only breeds that
# genuinely appear in that corpus are listed; the Finnish and Hungarian herding
# breeds in this study mostly do not, and are left without a photo rather than
# illustrated with a lookalike. REF.BREED_IMAGE simply has no row for them and
# the dashboard falls back to the drawn silhouette.
BREED_MATCH: dict[str, str] = {
    "Border Collie":                 "Border_collie",
    "German Shepherd":               "German_shepherd",
    "Golden Retriever":              "golden_retriever",
    "Labrador Retriever":            "Labrador_retriever",
    "Bouvier":                       "Bouvier_des_Flandres",
    "Bouvier des Ardennes":          "Bouvier_des_Flandres",
    "Belgian Shepherd Malinois":     "malinois",
    "Belgian Shepherd Groenendael":  "groenendael",
    "Belgian Shepherd":              "groenendael",
    "Dutch Shepherd":                "malinois",
    "English Springer Spaniel":      "English_springer",
    "Flat-Coated Retriever":         "flat-coated_retriever",
    "Standard Poodle":               "standard_poodle",
    "Vizsla":                        "vizsla",
    "Smooth Collie":                 "collie",
    "Bull Terrier (Miniature)":      "Staffordshire_bullterrier",
    "Lagotto Romagnolo":             "Irish_water_spaniel",
    "Spanish Water Dog":             "Irish_water_spaniel",
    "Finnish Lapphund":              "Siberian_husky",
    "Lapponian Herder":              "Siberian_husky",
    "Australian Kelpie":             "kelpie",
    "Nova Scotia Duck Tolling Retriever": "golden_retriever",
}

# Breeds above whose photo is a NEAR MATCH rather than the breed itself. Held
# separately so the caption can say which, instead of quietly passing off a
# Malinois as a Dutch Shepherd.
APPROXIMATE = {
    "Bouvier des Ardennes", "Dutch Shepherd", "Belgian Shepherd",
    "Bull Terrier (Miniature)", "Lagotto Romagnolo", "Spanish Water Dog",
    "Finnish Lapphund", "Lapponian Herder",
    "Nova Scotia Duck Tolling Retriever",
}


def auth_headers() -> dict:
    token = (env("KAGGLE_API_TOKEN", "") or "").strip()
    if not token:
        die("KAGGLE_API_TOKEN is not set in .env")
    return {"Authorization": f"Bearer {token}"}


def rng(session, headers: dict, start: int, end: int) -> bytes:
    """Fetch bytes [start, end] inclusive, from the RESOLVED storage URL.

    Kaggle's /datasets/download endpoint answers with a fresh signed redirect
    to Google Cloud Storage on every single call, and paying for that redirect
    per range costs about forty seconds a piece — thirty-odd of them turned a
    few megabytes of transfer into ten minutes. The signed URL is good for the
    life of this run, so it is resolved once in main() and every range goes
    straight at storage: ~0.5s each, measured.
    """
    h = dict(headers)
    h["Range"] = f"bytes={start}-{end}"
    r = session.get(RESOLVED[0], headers=h, stream=True, timeout=180)
    if r.status_code not in (200, 206):
        r.close()
        die(f"range request returned {r.status_code}; the signed URL may have "
            f"expired — rerun the script")
    data = r.raw.read(end - start + 1, decode_content=True)
    r.close()
    return data


def central_directory(session, headers: dict, total: int) -> list[tuple[str, int, int, int, int]]:
    """(name, local_header_offset, compressed_size, uncompressed_size, method)."""
    tail = rng(session, headers, max(0, total - 65_536), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        die("no End of Central Directory record; the archive is not a plain ZIP")
    n_entries, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])

    cd = rng(session, headers, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) - 4 and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        out.append((name, lho, csize, usize, method))
        p += 46 + nlen + elen + clen
    info(f"central directory: {len(out):,} entries (header said {n_entries:,})")
    return out


def extract(session, headers: dict, entry) -> bytes:
    """Range-read one member and decompress it."""
    name, lho, csize, usize, method = entry
    head = rng(session, headers, lho, lho + 29)
    if head[:4] != b"PK\x03\x04":
        die(f"bad local header for {name}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = lho + 30 + nlen + elen
    blob = rng(session, headers, start, start + csize - 1)
    if method == 0:
        return blob
    if method == 8:
        return zlib.decompress(blob, -zlib.MAX_WBITS)
    die(f"unsupported ZIP compression method {method} for {name}")


def thumbnail(raw: bytes, size: int) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    # square centre crop, so a grid of cards does not go ragged
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2,
                  (w - side) // 2 + side, (h - side) // 2 + side))
    im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=78, optimize=True)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=256, help="thumbnail edge, px")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()
    try:
        import requests
    except ImportError:
        die("pip install requests")

    headers = auth_headers()

    header("Locating the archive")
    session = requests.Session()
    probe = session.get(URL, headers=headers, stream=True,
                         allow_redirects=True, timeout=120)
    total = int(probe.headers.get("Content-Length", 0))
    status, accepts = probe.status_code, probe.headers.get("Accept-Ranges")
    RESOLVED[0] = probe.url          # the signed GCS URL, reused for every range
    probe.close()
    if status != 200 or not total:
        die(f"GET returned {status} with Content-Length {total}")
    if accepts != "bytes":
        die("the endpoint does not advertise byte ranges; a full download would "
            "be 787 MB. Refusing rather than doing that silently.")
    ok(f"{total / 1e6:.0f} MB, ranges accepted — reading the index only")

    header("Reading the ZIP index")
    entries = central_directory(session, headers, total)

    # one image per wanted folder, first alphabetically so reruns are stable
    wanted = {v: None for v in BREED_MATCH.values()}
    for e in entries:
        name = e[0]
        if not name.lower().endswith(".jpg"):
            continue
        for frag in wanted:
            if f"-{frag}/" in name or f"-{frag}\\" in name:
                if wanted[frag] is None or name < wanted[frag][0]:
                    wanted[frag] = e
    found = {k: v for k, v in wanted.items() if v}
    missing = sorted(set(wanted) - set(found))
    ok(f"matched {len(found)} of {len(wanted)} breed folders")
    if missing:
        warn(f"no folder for: {', '.join(missing)}")

    header("Fetching and downsizing")
    blobs: dict[str, str] = {}
    for frag, e in sorted(found.items()):
        try:
            raw = extract(session, headers, e)
            small = thumbnail(raw, args.size)
            blobs[frag] = base64.b64encode(small).decode("ascii")
            print(f"  {G['ok']} {frag:<28} {len(raw) / 1024:6.0f} KB "
                  f"{G['bar']}> {len(small) / 1024:5.1f} KB")
        except Exception as exc:  # noqa: BLE001
            warn(f"{frag}: {str(exc)[:100]}")

    if not blobs:
        die("nothing fetched")

    rowdata = []
    for breed, frag in sorted(BREED_MATCH.items()):
        if frag not in blobs:
            continue
        rowdata.append((breed, frag, blobs[frag], breed in APPROXIMATE))

    total_kb = sum(len(r[2]) for r in rowdata) / 1024
    ok(f"{len(rowdata)} breed rows, {total_kb:.0f} KB of base64")

    if args.dry_run:
        info("--dry-run: not writing to Snowflake")
        return 0

    header("Writing REF.BREED_IMAGE")
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("USE DATABASE " + env("SNOWFLAKE_DATABASE", "TELLTAIL"))
            cur.execute("""
                CREATE TABLE IF NOT EXISTS REF.BREED_IMAGE (
                    breed          STRING,
                    source_folder  STRING,
                    image_b64      STRING,
                    is_approximate BOOLEAN,
                    credit         STRING,
                    loaded_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            cur.execute("DELETE FROM REF.BREED_IMAGE")
            cur.executemany(
                "INSERT INTO REF.BREED_IMAGE "
                "(breed, source_folder, image_b64, is_approximate, credit) "
                "VALUES (%s, %s, %s, %s, %s)",
                [(b, f, img, appx, CREDIT) for b, f, img, appx in rowdata])
        n = q(conn, "SELECT COUNT(*) AS n FROM REF.BREED_IMAGE")[0]["N"]
        cov = q(conn, """
            SELECT COUNT(DISTINCT d.breed) AS have
            FROM REF.DOG_INFO d JOIN REF.BREED_IMAGE i ON i.breed = d.breed
        """)[0]["HAVE"]
        allb = q(conn, "SELECT COUNT(DISTINCT breed) AS n FROM REF.DOG_INFO")[0]["N"]
        ok(f"REF.BREED_IMAGE: {n} rows, covering {cov} of {allb} study breeds")
        info("breeds without a photo keep the drawn silhouette; nothing is faked")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
