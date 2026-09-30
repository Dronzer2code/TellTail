#!/usr/bin/env python3
"""
Rebuild warehouse/tt_font.py — Satoshi, embedded as base64 woff2.

    python scripts/fetch_satoshi.py

Streamlit in Snowflake has no outbound network, so a webfont referenced by URL
never loads and never says it did not. The only way to get a typeface onto that
page is to ship its bytes, which is what this writes.

Run it when the weights the stylesheet uses change, or to re-pull from upstream.
It needs network; the app it produces does not.

Satoshi is by Indian Type Foundry under the ITF Free Font License, which permits
embedding in a web page. If that ever changes, this script is the one place that
has to be revisited.
"""
from __future__ import annotations

import base64
import re
import sys
import textwrap
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import die, header, info, ok  # noqa: E402

# 900 is deliberately absent: nothing in the stylesheet asks for it and it is
# 31 KB. 600 and 800 ARE used and are also absent — a browser resolves them to
# the nearest embedded face, which is what those weights should look like.
WEIGHTS = (400, 500, 700)
CSS_URL = "https://api.fontshare.com/v2/css?f[]=satoshi@" + ",".join(
    str(w) for w in WEIGHTS)
OUT = Path(__file__).resolve().parent.parent / "warehouse" / "tt_font.py"


def fetch(url: str, what: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001 - any network failure is the same story
        die(f"could not fetch {what}: {e}")
        raise


def main() -> int:
    header("Fetching Satoshi from Fontshare")
    css = fetch(CSS_URL, "the @font-face stylesheet").decode("utf-8")

    # One @font-face block per weight/style. Italics are skipped — the app has
    # no italic rule, and pulling them would double the payload for nothing.
    faces: dict[int, str] = {}
    for block in re.findall(r"@font-face\s*\{(.*?)\}", css, re.S):
        style = re.search(r"font-style:\s*(\w+)", block)
        if style and style.group(1) != "normal":
            continue
        weight = re.search(r"font-weight:\s*(\d+)", block)
        url = re.search(r"url\('(//[^']*\.woff2)'\)", block)
        if weight and url:
            faces[int(weight.group(1))] = "https:" + url.group(1)

    missing = [w for w in WEIGHTS if w not in faces]
    if missing:
        die(f"upstream did not serve weight(s) {missing}. Got {sorted(faces)}. "
            f"Fontshare changed its response shape — fix the parse above rather "
            f"than silently shipping a font with holes in it.")

    chunks, total = [], 0
    for w in WEIGHTS:
        raw = fetch(faces[w], f"Satoshi {w}")
        if raw[:4] != b"wOF2":
            die(f"weight {w} is not a woff2 file (magic {raw[:4]!r}). Refusing "
                f"to embed it — the browser would fail silently.")
        b64 = base64.b64encode(raw).decode("ascii")
        total += len(b64)
        info(f"  {w}: {len(raw):,} B raw, {len(b64):,} B base64")
        body = "\n".join(f'    "{line}"' for line in textwrap.wrap(b64, 78))
        chunks.append(f"_W{w} = (\n{body}\n)\n\n")

    OUT.write_text(HEADER + "".join(chunks) + FOOTER, encoding="utf-8")
    ok(f"wrote {OUT.relative_to(OUT.parent.parent.parent)} "
       f"({OUT.stat().st_size:,} B on disk, {total:,} B of base64)")
    info("stage it with: python scripts/deploy_streamlit.py")
    return 0


HEADER = '''"""
Satoshi, as bytes, because Streamlit in Snowflake cannot fetch a font.

WHY THIS FILE EXISTS. Naming a font in a CSS font-family does nothing unless the
browser can obtain it, and the SiS sandbox has no outbound network — the same
constraint that put the breed photographs in a table as base64 and BokehJS
inline in its own document (hazards 5 and 6 in streamlit_app.py). An
`@import url(fontshare.com/...)` in the stylesheet is a silent no-op there: no
error, no missing-font warning, just Segoe UI and a designer wondering why.

So the woff2 for each weight is embedded below as a `data:` URI and served from
the document itself. Three weights, not the four Fontshare publishes — 900 was
dropped because nothing in the app asks for it and it is 31 KB of a file that is
already large. The stylesheet does use 600 and 800 and neither is embedded: a
browser with no exact match picks the nearest available face, 500 or 700, which
is the correct result rather than a degraded one.

IT IS A SEPARATE MODULE, AND STAGED AS ONE. 102 KB of base64 in the middle of
streamlit_app.py would make the app unreadable for the sake of a typeface.
scripts/deploy_streamlit.py PUTs this file alongside the app and environment.yml;
streamlit_app.py imports it inside a try/except and falls back to the Geist /
Inter / system stack if it is missing, so a stage that somehow lacks this file
renders a slightly plainer dashboard rather than no dashboard at all.

Satoshi is by Indian Type Foundry, released under the ITF Free Font License,
which permits embedding in a web page. Regenerate with:

    python scripts/fetch_satoshi.py

DO NOT HAND-EDIT THE STRINGS BELOW.
"""
from __future__ import annotations

'''

FOOTER = '''
# NO BRACES ARE ESCAPED HERE AND NONE NEED TO BE. streamlit_app.py interpolates
# FACE_CSS into an f-string stylesheet, and f-string interpolation does not
# recurse into the value it substitutes — the CSS below is inserted verbatim.
FACE_CSS = "".join(
    "@font-face{font-family:'Satoshi';font-style:normal;font-display:swap;"
    "font-weight:" + str(_w) + ";"
    "src:url(data:font/woff2;base64," + _b + ") format('woff2');}"
    for _w, _b in ((400, _W400), (500, _W500), (700, _W700))
)

# The stack every rule in the app points at. Satoshi first, then the two the
# app used before it, then the system faces — so a missing embed degrades in
# one step rather than dropping to Times.
STACK = 'Satoshi, Geist, Inter, -apple-system, "Segoe UI", sans-serif'
'''

if __name__ == "__main__":
    raise SystemExit(main())
