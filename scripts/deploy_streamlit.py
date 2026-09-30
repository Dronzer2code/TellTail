#!/usr/bin/env python3
"""
Deploy the dashboard into the warehouse.

    python scripts/deploy_streamlit.py
    python scripts/deploy_streamlit.py --name TELLTAIL_DEV

Ships streamlit_app.py AND environment.yml to MARTS.APP_STAGE, then creates the
Streamlit object pointing at them. It refuses to deploy without environment.yml,
because a SiS app missing it fails at import time with a ModuleNotFoundError for
plotly that reproduces nowhere else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    WAREHOUSE_DIR,
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

STAGE = "MARTS.APP_STAGE"


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy the Streamlit app to Snowflake.")
    ap.add_argument("--name", default="TELLTAIL")
    ap.add_argument("--schema", default="MARTS")
    args = ap.parse_args()

    load_env()
    app = WAREHOUSE_DIR / "streamlit_app.py"
    envyml = WAREHOUSE_DIR / "environment.yml"
    font = WAREHOUSE_DIR / "tt_font.py"

    if not app.exists():
        die(f"{app} not found")
    if not envyml.exists():
        die(f"{envyml} not found. Streamlit in Snowflake needs it next to the app "
            f"or plotly will ModuleNotFoundError in SiS only.")

    # WARN RATHER THAN DIE. tt_font.py carries the embedded Satoshi woff2, and
    # the app imports it inside a try/except precisely so a stage without it
    # renders in the fallback stack instead of failing. That makes a missing
    # font a cosmetic regression, not a broken deploy — but it is invisible in
    # Snowsight, so it gets said out loud here rather than discovered later.
    if not font.exists():
        warn(f"{font.name} not found — the app will deploy and render in the "
             f"Geist/Inter fallback rather than Satoshi. Rebuild it with: "
             f"python scripts/fetch_satoshi.py")

    # A `python=` pin in environment.yml deploys cleanly and then fails the app
    # at load with 'Packages not found: python==3.11' — a blank error page, no
    # traceback, and nothing wrong with the app itself. SiS resolves this file
    # against the `snowflake` Anaconda channel, which has no package named
    # `python`; the interpreter version belongs to the STREAMLIT object.
    for i, line in enumerate(envyml.read_text(encoding="utf-8").splitlines(), 1):
        bare = line.split("#", 1)[0].strip().lstrip("-").strip()
        if bare.replace(" ", "").startswith(("python=", "python>", "python<")):
            die(f"{envyml}:{i} pins the interpreter: {line.strip()!r}. "
                f"Remove it. SiS has no `python` package to resolve, so the "
                f"app will load to 'Packages not found' instead of rendering.")

    # Gate the deploy on the SiS compatibility suite.
    #
    # Both failures that reached a running app were things a parse could not
    # catch: a `python=` pin in environment.yml, and an f-string interpolating
    # {BG} when the constant is SURFACE. The second one parses perfectly and
    # dies at import with NameError. Deploying is the expensive way to find
    # that out, so check here rather than in Snowsight.
    import subprocess
    repo = Path(__file__).resolve().parent.parent
    probe = subprocess.run(
        [sys.executable, str(repo / "tests" / "sis_compat.py"), str(app)],
        capture_output=True, text=True)
    for line in (probe.stdout + probe.stderr).splitlines():
        print("  " + line)
    if probe.returncode != 0:
        die("tests/sis_compat.py rejected the app. Fix it before deploying — "
            "these failures do not surface until the app loads in Snowsight.")

    db = env("SNOWFLAKE_DATABASE", "TELLTAIL")
    wh = env("SNOWFLAKE_WAREHOUSE", "TELLTAIL_WH")
    conn = connect()

    try:
        header("Uploading to " + STAGE)
        with conn.cursor() as cur:
            staged = [f for f in (app, envyml, font) if f.exists()]
            for f in staged:
                cur.execute(
                    f"PUT 'file://{f.resolve().as_posix()}' @{STAGE} "
                    f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                )
                for r in cur.fetchall():
                    print(f"    {r[0]:<24} {r[6]}")
        ok(", ".join(f.name for f in staged) + " staged")

        header(f"Creating STREAMLIT {db}.{args.schema}.{args.name}")
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE OR REPLACE STREAMLIT {db}.{args.schema}.{args.name}
                    ROOT_LOCATION = '@{db}.{STAGE}'
                    MAIN_FILE     = 'streamlit_app.py'
                    QUERY_WAREHOUSE = {wh}
                    TITLE = 'TELLTAIL — canine syndrome detection'
                    COMMENT = 'Nine tabs. Reads only tables; every Cortex call is cached upstream.'
            """)
        ok("Streamlit object created")

        rows = q(conn, f"SHOW STREAMLITS LIKE '{args.name}' IN SCHEMA {db}.{args.schema}")
        if rows:
            r = rows[0]
            url_id = r.get("url_id") or r.get("URL_ID")
            header("Open it")
            print(f"  Snowsight -> Projects -> Streamlit -> {args.name}")
            if url_id:
                print(f"  url_id: {url_id}")

        header("Preflight: does the app have data to render?")
        checks = [
            ("MARTS.PACK_STATUS",         "tab 1 (The Pack)"),
            ("MARTS.EPOCH_STATES",        "tabs 2, 3 (Live Collar, Ethogram)"),
            ("MARTS.SYNDROME_MATCHES",    "tab 4 (Syndromes) — THE tab"),
            ("MARTS.SYNDROME_MATCH_ROWS", "tab 4 symbol highlighting (the hero image)"),
            ("MARTS.SYNDROME_SENSITIVITY","tab 4 sensitivity curve"),
            ("MARTS.DOG_DEVIATION",       "tab 5 (Baselines)"),
            ("AI.VET_NOTES",              "tab 6 (Vet Note)"),
            ("ML.DRIVER_INSIGHTS",        "tab 7 (Drivers)"),
            ("ML.CONFUSION_MATRIX",       "tab 7 confusion matrix"),
            ("REF.AAC_OUTCOMES",          "tab 8 (Shelter Reality) — run austin_sync.py"),
            ("ORACLE.PUBLISH_QUEUE",      "tab 9 (Pipeline) publish log"),
        ]
        empty = []
        for table, what in checks:
            try:
                n = int(list(q(conn, f"SELECT COUNT(*) AS n FROM {table}")[0].values())[0])
                mark = "ok " if n else "EMPTY"
                print(f"  [{mark}] {table:<28} {n:>9,}   {what}")
                if not n:
                    empty.append((table, what))
            except Exception as exc:  # noqa: BLE001
                print(f"  [MISS] {table:<28} {'—':>9}   {what}")
                empty.append((table, f"{what} ({str(exc).splitlines()[0][:60]})"))

        if empty:
            print()
            warn(f"{len(empty)} source(s) are empty; those tabs will render a placeholder")
            info("the usual order is: load_raw.py -> run_sql.py --all -> replay.py "
                 "-> austin_sync.py")
        else:
            print()
            ok("every tab has data")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
