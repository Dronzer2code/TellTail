#!/usr/bin/env python3
"""
Run the warehouse, in order, idempotently.

    python scripts/run_sql.py --all              # everything, 00 through 11
    python scripts/run_sql.py --only 07          # just the syndrome layer
    python scripts/run_sql.py --from 05          # 05 onward
    python scripts/run_sql.py --all --dry-run    # print statements, run nothing
    python scripts/run_sql.py --all --skip-bootstrap   # DDL only, no procedures

Two things happen here that a plain `snowsql -f` cannot do:

  * ROLE SWITCHING. 00_account_setup.sql needs ACCOUNTADMIN; everything after it
    runs as SNOWFLAKE_ROLE. Running the whole build as ACCOUNTADMIN would mean
    the objects are owned by a role the Streamlit app does not use.

  * BOOTSTRAP HOOKS. Several files create a procedure that must then be CALLed
    once to populate the object the next file depends on — the classifier has to
    train before the state layer can predict, and REF.LABEL_MAP has to be pushed
    from the Gate A profile before the classifier has a target. The tasks in
    11_tasks.sql handle steady state; these hooks handle the cold start.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    REF_DIR,
    connect,
    die,
    env,
    execute_script,
    header,
    info,
    iter_sql_files,
    load_column_map,
    load_env,
    ok,
    q,
    warn,
)

BUILD_ID = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# bootstrap hooks — the cold start the tasks cannot do for you
# ---------------------------------------------------------------------------


def hook_after_ref_seed(conn, args) -> None:
    """Push the Gate A profile into REF.LABEL_MAP, then assign the holdout.

    This is the join between the profiler's view of the file and the
    warehouse's view of it. Without it ML.V_TRAIN is empty and every downstream
    object is built on nothing.
    """
    cm = load_column_map(allow_example=args.allow_example_map)
    src = cm.get("_meta", {}).get("loaded_from", "?")
    ethogram = cm.get("ethogram_map") or {}
    label_report = cm.get("label_report") or {}

    rows: list[tuple[str, str, str | None, int]] = []
    if label_report:
        # generated map: we know per-column counts
        for column, entries in label_report.items():
            canon = _canonical_label_column(cm, column)
            if canon is None:
                continue
            for e in entries:
                if e["value"] in ("<null>", "nan"):
                    continue
                rows.append((e["value"], canon, e["state"], int(e["count"])))
    else:
        # example prior: no counts, primary column only
        for raw, state in ethogram.items():
            if raw.startswith("_"):
                continue
            rows.append((raw, "label_primary", state, 0))

    if not rows:
        die("no label vocabulary to push. Run scripts/profile_dataset.py first.")

    with conn.cursor() as cur:
        cur.execute("DELETE FROM REF.LABEL_MAP")
        cur.executemany(
            "INSERT INTO REF.LABEL_MAP (raw_label, source_column, state, n_rows) "
            "VALUES (%s, %s, %s, %s)",
            rows,
        )
    mapped = sum(1 for r in rows if r[2])
    ok(f"REF.LABEL_MAP: {len(rows)} labels from {src} ({mapped} mapped to states)")

    unmapped = [r[0] for r in rows if not r[2] and r[1] == "label_primary"]
    if unmapped:
        warn(f"unmapped primary labels (they will not train): {unmapped}")

    states = {r[2] for r in rows if r[2]}
    for critical in ("SHAKE", "SCRATCH"):
        if critical not in states:
            warn(f"{critical} is not a label in this dataset — the state ladder will "
                 f"derive it heuristically and flag state_source='HEURISTIC'")

    n_dogs = q(conn, "SELECT COUNT(*) AS n FROM REF.DOG_INFO")[0]["N"]
    if n_dogs:
        r = q(conn, "CALL REF.SP_ASSIGN_HOLDOUT(0.22)")
        ok(str(list(r[0].values())[0]))
    else:
        warn("REF.DOG_INFO is empty — run scripts/load_raw.py before 05. "
             "No holdout assigned, so the classifier would train on everything.")


def _canonical_label_column(cm: dict, actual: str) -> str | None:
    """Map an actual CSV header back to its canonical slot name."""
    for canon, real in (cm.get("columns") or {}).items():
        if real == actual and canon.startswith(("label_", "point_", "task")):
            return canon
    return None


def hook_after_staging(conn, args) -> None:
    _call(conn, "STAGING.SP_BUILD_BULK_FEATURES()",
          "building historical epoch features (10.6M samples -> ~106K epochs)",
          slow=True)


def hook_after_ml(conn, args) -> None:
    _call(conn, "ML.SP_TRAIN_STATE_MODEL()", "training the ethogram classifier", slow=True)
    _call(conn, "ML.SP_EVALUATE_HOLDOUT()", "evaluating on entirely held-out dogs")


def hook_after_syndromes(conn, args) -> None:
    _call(conn, "MARTS.SP_BUILD_SYNDROMES()", "scanning for syndromes (MATCH_RECOGNIZE)")
    _call(conn, "MARTS.SP_BUILD_MATCH_ROWS()", "tagging matched epochs with pattern symbols")
    _call(conn, "MARTS.SP_SYNDROME_SWEEP()", "sensitivity sweep: 6 patterns x 3 strictness levels",
          slow=True)


def hook_after_timeseries(conn, args) -> None:
    _call(conn, "ML.SP_SNAPSHOT_ACTIVITY()", "snapshotting the activity series")
    _call(conn, "ML.SP_SET_BOUNDARY()", "fixing the train/detect boundary")
    _call(conn, "ML.SP_RUN_FORECAST()", "ML.FORECAST", slow=True)
    _call(conn, "ML.SP_RUN_ANOMALY()", "ML.ANOMALY_DETECTION", slow=True)
    _call(conn, "ML.SP_RUN_TOP_INSIGHTS()", "TOP_INSIGHTS / contribution analysis")


def hook_after_ai(conn, args) -> None:
    if args.no_cortex:
        warn("--no-cortex: skipping the AI layer bootstrap. "
             "The task in 11_tasks.sql will still run it on schedule.")
        return
    cap = env("CORTEX_MAX_ROWS_PER_BATCH", "25")
    info(f"Cortex batch cap is {cap} rows. Trial accounts are capped at roughly "
         f"ten credits/day of AI Function usage.")
    _call(conn, "AI.SP_GENERATE_NOTES()", "AI_COMPLETE: SOAP handoff notes", slow=True)
    _call(conn, "AI.SP_GENERATE_TRIAGE()", "AI_CLASSIFY: triage severity", slow=True)
    _call(conn, "AI.SP_GENERATE_PACK_BRIEF()", "AI_AGG: pack-wide brief", slow=True)


def hook_after_oracle(conn, args) -> None:
    _call(conn, "ORACLE.SP_ENQUEUE_ATTESTATIONS()", "staging attestations for the bridge")


def _call(conn, proc: str, label: str, *, slow: bool = False) -> None:
    info(f"{label}{'  (this one takes a while)' if slow else ''}")
    t0 = time.perf_counter()
    try:
        rows = q(conn, f"CALL {proc}")
        msg = list(rows[0].values())[0] if rows else "(no return value)"
        ok(f"{msg}   [{time.perf_counter() - t0:.1f}s]")
    except Exception as exc:  # noqa: BLE001
        warn(f"{proc} failed: {str(exc).splitlines()[0][:200]}")
        warn("   continuing — the procedure records its own failure state")


POST_HOOKS: dict[str, Callable] = {
    "02_ref_seed.sql":         hook_after_ref_seed,
    "04_staging_dt.sql":       hook_after_staging,
    "05_ml_classification.sql": hook_after_ml,
    "07_syndromes.sql":        hook_after_syndromes,
    "08_ml_timeseries.sql":    hook_after_timeseries,
    "09_ai_layer.sql":         hook_after_ai,
    "10_oracle.sql":           hook_after_oracle,
}

ADMIN_FILES = {"00_account_setup.sql"}


# ---------------------------------------------------------------------------


def log_build(conn, script: str, t0: float, n_ok: int, n_ko: int, status: str, detail: str = ""):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO REF.BUILD_LOG (build_id, script_name, started_at, "
                "finished_at, statements_ok, statements_ko, status, detail) "
                "SELECT %s, %s, DATEADD('second', %s, CURRENT_TIMESTAMP()), "
                "CURRENT_TIMESTAMP(), %s, %s, %s, %s",
                (BUILD_ID, script, -int(time.time() - t0), n_ok, n_ko, status, detail[:1000]),
            )
    except Exception:  # noqa: BLE001 - REF.BUILD_LOG may not exist yet on file 00/01
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Run warehouse/*.sql in order.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="run every file, 00 through 11")
    g.add_argument("--only", metavar="NN", help="run one file by numeric prefix")
    g.add_argument("--from", dest="from_", metavar="NN", help="run from this prefix onward")
    ap.add_argument("--dry-run", action="store_true", help="print statements, execute nothing")
    ap.add_argument("--skip-bootstrap", action="store_true", help="DDL only; call no procedures")
    ap.add_argument("--no-cortex", action="store_true", help="skip the AI layer bootstrap")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--allow-example-map", action="store_true",
                    help="permit the unverified ref/column_map.example.json (Gate A not run)")
    args = ap.parse_args()

    load_env()

    files = list(iter_sql_files(args.only))
    if args.from_:
        files = [f for f in iter_sql_files() if f.name[:2] >= args.from_]
    if not files:
        die(f"no warehouse/*.sql matched {args.only or args.from_ or 'anything'}")

    header(f"TELLTAIL build {BUILD_ID}")
    print(f"  account   : {env('SNOWFLAKE_ACCOUNT', '(unset)')}")
    print(f"  database  : {env('SNOWFLAKE_DATABASE', 'TELLTAIL')}")
    print(f"  warehouse : {env('SNOWFLAKE_WAREHOUSE', 'TELLTAIL_WH')}")
    print(f"  role      : {env('SNOWFLAKE_ROLE', 'SYSADMIN')}")
    print(f"  files     : {', '.join(f.name for f in files)}")
    if args.dry_run:
        warn("dry run: nothing will be executed")

    conn = None if args.dry_run else connect()
    admin_conn = None
    total_ok = total_ko = 0
    t_start = time.perf_counter()

    try:
        for f in files:
            t0 = time.time()
            use_admin = f.name in ADMIN_FILES

            if args.dry_run:
                execute_script(None, f, dry_run=True)  # type: ignore[arg-type]
                continue

            target = conn
            if use_admin:
                admin_role = env("SNOWFLAKE_ADMIN_ROLE", "ACCOUNTADMIN")
                info(f"{f.name} needs {admin_role}; opening a second session for it")
                try:
                    admin_conn = admin_conn or connect(role=admin_role)
                    target = admin_conn
                except Exception as exc:  # noqa: BLE001
                    warn(f"cannot assume {admin_role}: {str(exc).splitlines()[0]}")
                    warn("skipping 00_account_setup.sql — run it by hand in Snowsight. "
                         "The TIMEZONE line in particular is not optional.")
                    continue

            try:
                n_ok, n_ko = execute_script(
                    target, f, stop_on_error=not args.continue_on_error
                )
                total_ok += n_ok
                total_ko += n_ko
                log_build(conn, f.name, t0, n_ok, n_ko, "OK" if not n_ko else "PARTIAL")
            except Exception as exc:  # noqa: BLE001
                log_build(conn, f.name, t0, 0, 1, "FAILED", str(exc))
                die(f"{f.name} failed. Fix it and re-run:  "
                    f"python scripts/run_sql.py --from {f.name[:2]}")

            hook = POST_HOOKS.get(f.name)
            if hook and not args.skip_bootstrap:
                header(f"bootstrap · {f.name}")
                hook(conn, args)

        header("Build complete")
        if args.dry_run:
            ok("dry run finished")
            return 0

        ok(f"{total_ok} statements in {time.perf_counter() - t_start:.1f}s"
           + (f", {total_ko} failed" if total_ko else ""))

        _summary(conn)
        return 0 if not total_ko else 1
    finally:
        for c in (conn, admin_conn):
            if c:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass


def _summary(conn) -> None:
    """Print what actually got built. If a number here is zero, the next stage
    is standing on nothing and you want to see that now."""
    probes = [
        ("raw rows (bulk)",     "SELECT COUNT(*) FROM RAW.COLLAR_TELEMETRY_BULK"),
        ("raw rows (live)",     "SELECT COUNT(*) FROM RAW.COLLAR_TELEMETRY"),
        ("dogs",                "SELECT COUNT(*) FROM REF.DOG_INFO"),
        ("holdout dogs",        "SELECT COUNT(*) FROM REF.HOLDOUT_DOGS"),
        ("labels mapped",       "SELECT COUNT(*) FROM REF.LABEL_MAP WHERE state IS NOT NULL"),
        ("epochs (bulk)",       "SELECT COUNT(*) FROM STAGING.EPOCH_FEATURES_BULK"),
        ("epochs (live)",       "SELECT COUNT(*) FROM STAGING.EPOCH_FEATURES"),
        ("epochs classified",   "SELECT COUNT(*) FROM MARTS.EPOCH_STATES"),
        ("syndrome matches",    "SELECT COUNT(*) FROM MARTS.SYNDROME_MATCHES"),
        ("sensitivity matches", "SELECT COUNT(*) FROM MARTS.SYNDROME_SENSITIVITY"),
        ("vet notes",           "SELECT COUNT(*) FROM AI.VET_NOTES"),
        ("attestations queued", "SELECT COUNT(*) FROM ORACLE.PUBLISH_QUEUE"),
    ]
    header("What got built")
    for label, sql in probes:
        try:
            n = list(q(conn, sql)[0].values())[0]
            flag = "" if n else "   <- empty"
            print(f"  {label:<22} {n:>12,}{flag}")
        except Exception:  # noqa: BLE001
            print(f"  {label:<22} {'n/a':>12}")

    try:
        rows = q(conn, "SELECT holdout_accuracy, holdout_dogs, macro_f1, classifier "
                       "FROM ML.MODEL_SUMMARY")
        if rows:
            r = rows[0]
            print()
            print(f"  classifier            {r['CLASSIFIER']}")
            print(f"  holdout accuracy      {100 * (r['HOLDOUT_ACCURACY'] or 0):.2f}%"
                  f"  on {r['HOLDOUT_DOGS']} unseen dogs  (macro F1 {r['MACRO_F1']})")
            print("  ^ dog-disjoint. Lower than a row-split figure, and the honest one.")
    except Exception:  # noqa: BLE001
        pass

    try:
        rows = q(conn, "SELECT syndrome_code, COUNT(*) AS n, ROUND(AVG(confidence),3) AS c "
                       "FROM MARTS.SYNDROME_MATCHES GROUP BY 1 ORDER BY 1")
        if rows:
            print()
            print("  syndrome matches:")
            for r in rows:
                print(f"    {r['SYNDROME_CODE']}  {r['N']:>5} matches   avg confidence {r['C']}")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    raise SystemExit(main())
