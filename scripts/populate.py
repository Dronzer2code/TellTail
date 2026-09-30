#!/usr/bin/env python3
"""
Fill the warehouse with enough REAL telemetry that every panel has something to
say — before a demo, a screenshot pass, or a write-up.

    python scripts/populate.py                    # 45 dogs, ~4 min of live feed
    python scripts/populate.py --minutes 15 --speed 6
    python scripts/populate.py --census           # print the census, change nothing
    python scripts/populate.py --no-replay        # drive the DAG over what is here

NOTHING IN HERE IS GENERATED, AND THAT IS THE WHOLE CONSTRAINT.

Every row this script causes to exist comes from RAW.COLLAR_TELEMETRY_BULK —
the 10.6M-sample Vehkaoja et al. corpus, 45 dogs, dual IMU at 100 Hz — moved
into the live landing table by ingest/replay.py with a server-side
INSERT ... SELECT. It is the same sensor data, re-anchored to now so the DAG
sees it arrive. Nothing is synthesised, nothing is fabricated to make a chart
look busier, and no number is written into a mart by hand.

There IS a synthetic path in this repo — scripts/demo_spike.py — and this
script deliberately does not call it. That one injects labelled signal with
is_synthetic = TRUE, the flag propagates to every layer, and the dashboard
prints SYNTHETIC on any panel showing it. It exists so a detector can be shown
firing on camera; it has no business in a population run whose entire claim is
that the charts are real.

WHAT THIS ACTUALLY DOES, in order:

  1. Census. Counts every table a tab reads, so the run has a before.
  2. Replay. Pushes a real slice of the corpus across ALL dogs into the live
     table, which is what makes the feed fresh and the pack complete.
  3. Waits for the transform DAG, which is not a sleep — it polls until
     MARTS.EPOCH_STATES has caught up with the newest sample that landed.
  4. Triggers the task graph once, explicitly, and waits for it to finish, so
     the syndrome scan, the ML routines and the Cortex batches all run over the
     rows that just arrived rather than whenever their 2-minute schedule
     next fires.
  5. Census again, and prints the delta plus a per-tab readiness report.

Cost: the replay is warehouse compute on an INSERT ... SELECT, the DAG is
Dynamic Table refreshes that were going to happen anyway, and the only Cortex
spend is AI_COMPLETE / AI_CLASSIFY / AI_AGG over findings that are NEW — the
AI layer dedupes on the finding key, so re-running this does not re-bill for a
note it already has.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    REPO,
    connect,
    header,
    info,
    load_env,
    ok,
    q,
    warn,
)

BATCH_WALL_SECONDS = 8      # must match ingest/replay.py
ROOT_TASK = "MARTS.T_ROOT"

# The three tips of the task graph. T_ATTEST ends the AI chain (notes, triage,
# brief, enqueue), T_DRIVERS ends the ML chain (boundary, forecast, anomaly,
# insights) and T_OBSERVE ends the syndrome chain (scan, symbol tagging,
# observability snapshot). When all three have completed, everything the root
# fanned out into has run.
LEAF_TASKS = ("T_ATTEST", "T_DRIVERS", "T_OBSERVE")

# The census. One label per thing a tab reads, in the order the tabs appear, so
# the report reads like a walk through the dashboard rather than like a schema
# dump. COUNT(*) on each — cheap, and honest about being a row count rather
# than a quality claim.
CENSUS = [
    ("raw rows, live",        "SELECT COUNT(*) FROM RAW.COLLAR_TELEMETRY"),
    ("raw dogs, live",        "SELECT COUNT(DISTINCT dog_id) FROM RAW.COLLAR_TELEMETRY"),
    ("epochs, live",          "SELECT COUNT(*) FROM STAGING.EPOCH_FEATURES"),
    ("epochs, classified",    "SELECT COUNT(*) FROM MARTS.EPOCH_STATES"),
    ("dogs on the pack tab",  "SELECT COUNT(*) FROM MARTS.PACK_STATUS"),
    ("state bouts",           "SELECT COUNT(*) FROM MARTS.STATE_BOUTS"),
    ("state transitions",     "SELECT COUNT(*) FROM MARTS.STATE_TRANSITIONS"),
    ("deviation rows",        "SELECT COUNT(*) FROM MARTS.DOG_DEVIATION"),
    ("syndrome findings",     "SELECT COUNT(*) FROM MARTS.SYNDROME_MATCHES"),
    ("syndromes firing",      "SELECT COUNT(DISTINCT syndrome_code) FROM MARTS.SYNDROME_MATCHES"),
    ("dogs with a finding",   "SELECT COUNT(DISTINCT dog_id) FROM MARTS.SYNDROME_MATCHES"),
    ("match rows tagged",     "SELECT COUNT(*) FROM MARTS.SYNDROME_MATCH_ROWS"),
    ("sensitivity rows",      "SELECT COUNT(*) FROM MARTS.SYNDROME_SENSITIVITY"),
    ("vet notes",             "SELECT COUNT(*) FROM AI.VET_NOTES"),
    ("triage rows",           "SELECT COUNT(*) FROM AI.TRIAGE"),
    ("activity history",      "SELECT COUNT(*) FROM ML.ACTIVITY_HISTORY"),
    ("forecast points",       "SELECT COUNT(*) FROM ML.ACTIVITY_FORECAST"),
    ("anomaly points",        "SELECT COUNT(*) FROM ML.ACTIVITY_ANOMALIES"),
    ("driver insights",       "SELECT COUNT(*) FROM ML.DRIVER_INSIGHTS"),
    ("holdout predictions",   "SELECT COUNT(*) FROM ML.HOLDOUT_PREDICTIONS"),
    ("shelter outcomes",      "SELECT COUNT(*) FROM REF.AAC_OUTCOMES"),
    ("attestations queued",   "SELECT COUNT(*) FROM ORACLE.PUBLISH_QUEUE"),
    ("attestations on chain",
     "SELECT COUNT(*) FROM ORACLE.PUBLISH_QUEUE WHERE status = 'CONFIRMED'"),
    ("task runs recorded",    "SELECT COUNT(*) FROM MARTS.TASK_HISTORY_SNAPSHOT"),
]


def census(conn) -> dict[str, int]:
    """Every count in one round trip.

    Twenty-four separate SELECTs is twenty-four warehouse round trips for
    numbers that are only ever read together; a UNION ALL is one.
    """
    sql = "\nUNION ALL\n".join(
        f"SELECT {i} AS ord, '{label}' AS label, ({q_}) AS n"
        for i, (label, q_) in enumerate(CENSUS))
    out: dict[str, int] = {}
    try:
        for r in q(conn, f"SELECT label, n FROM ({sql}) ORDER BY ord"):
            out[str(r["LABEL"])] = int(r["N"] or 0)
    except Exception as exc:  # noqa: BLE001
        warn(f"census incomplete: {str(exc).splitlines()[0][:100]}")
    return out


def freshness(conn) -> dict:
    """How far behind the pipeline's own clock is — the number the Live Collar
    tab is really showing. Measured against MAX(sample_ts), never against the
    wall clock: the replayer moves dog time faster than real time on purpose,
    so CURRENT_TIMESTAMP() is not the baseline anything downstream uses."""
    r = q(conn, """
        SELECT
            (SELECT MAX(sample_ts) FROM RAW.COLLAR_TELEMETRY)          AS raw_max,
            (SELECT MAX(epoch_ts)  FROM MARTS.EPOCH_STATES)            AS epoch_max,
            (SELECT COUNT(*) FROM RAW.COLLAR_TELEMETRY
              WHERE sample_ts > DATEADD('minute', -1,
                    (SELECT MAX(sample_ts) FROM RAW.COLLAR_TELEMETRY))) AS last_minute
    """)
    return r[0] if r else {}


def print_census(now: dict[str, int], before: dict[str, int] | None = None) -> None:
    header("Census")
    width = max(len(k) for k in now) if now else 20
    for label, n in now.items():
        if before is None:
            print(f"  {label:<{width}}  {n:>12,}")
            continue
        delta = n - before.get(label, 0)
        mark = f"  +{delta:,}" if delta > 0 else (f"  {delta:,}" if delta else "")
        print(f"  {label:<{width}}  {n:>12,}{mark}")


def replay(minutes: float, speed: float, dogs: int | None,
           reset: bool) -> int:
    """Run the replayer as its own process.

    It is the component that owns the live feed and it already knows how to do
    this correctly — two time bases, a fixed anchor, server-side inserts. Not
    importing and re-driving it from here: a second caller of an interactive
    loop is how the anchor gets resolved twice and the timeline sawtooths.
    """
    batches = max(1, int(round(minutes * 60 / BATCH_WALL_SECONDS)))
    cmd = [sys.executable, str(REPO / "ingest" / "replay.py"),
           "--speed", str(speed), "--max-batches", str(batches)]
    if dogs:
        cmd += ["--dogs", str(dogs)]
    if reset:
        cmd += ["--reset"]
    header("Replaying the corpus into the live table")
    info(" ".join(cmd[1:]))
    info(f"up to {batches} batches of {BATCH_WALL_SECONDS}s "
         f"(~{minutes:.0f} min wall clock, or less if it reaches the end)")
    print()
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def wait_for_dag(conn, *, timeout: int = 600, slack: int = 90) -> bool:
    """Poll until the classified epochs have caught up with the newest sample.

    NOT A SLEEP. Every Dynamic Table here is REFRESH_MODE = FULL on a one-minute
    target lag, so how long the chain takes is a function of how many rows just
    landed — a fixed wait is either a stall or a lie. `slack` is the tail of the
    newest epoch that is legitimately still forming.
    """
    header("Waiting for the transform DAG")
    started = time.time()
    last = None
    while time.time() - started < timeout:
        f = freshness(conn)
        raw_max, epoch_max = f.get("RAW_MAX"), f.get("EPOCH_MAX")
        if raw_max is None:
            warn("live table is empty — nothing to wait for")
            return False
        if epoch_max is not None:
            behind = (raw_max - epoch_max).total_seconds()
            if behind != last:
                info(f"epochs are {behind:,.0f}s behind the newest sample")
                last = behind
            if behind <= slack:
                ok(f"DAG caught up ({behind:,.0f}s behind, within {slack}s)")
                return True
        time.sleep(20)
    warn(f"DAG still behind after {timeout}s — continuing anyway; the Dynamic "
         f"Tables will finish on their own schedule")
    return False


def run_graph(conn, *, timeout: int = 900) -> bool:
    """Trigger the task DAG once and wait for its three leaves to finish.

    The graph already runs every two minutes, so this is not what makes it work
    — it is what makes the run DETERMINISTIC. Without it the script would exit
    somewhere between 'the syndrome scan has seen the new rows' and 'it has
    not', which is the difference between a screenshot of the pipeline and a
    screenshot of half of it.

    WAIT ON THE LEAVES, NOT ON AN IDLE ACCOUNT. Two things make the obvious
    condition — 'nothing is SCHEDULED or EXECUTING' — a wait that never ends.
    INFORMATION_SCHEMA.TASK_HISTORY returns every task the role can see,
    including Snowflake's own account housekeeping, which is permanently
    scheduled; and T_ROOT itself always has its next 2-minute tick sitting in
    SCHEDULED. So the condition is that each tip of the graph has COMPLETED a
    run that started after the trigger.
    """
    header("Running the task graph")
    t_trigger = q(conn, "SELECT CURRENT_TIMESTAMP() AS t")[0]["T"]
    try:
        with conn.cursor() as cur:
            cur.execute(f"EXECUTE TASK {ROOT_TASK}")
        ok(f"EXECUTE TASK {ROOT_TASK}")
    except Exception as exc:  # noqa: BLE001
        # A graph run already in flight is the common case on a live account,
        # and it is not an error: the schedule is doing the same work.
        warn(f"could not trigger the root task ({str(exc).splitlines()[0][:80]})")
        info("waiting on the scheduled run instead")

    started = time.time()
    waiting_on = set(LEAF_TASKS)
    while time.time() - started < timeout:
        done = q(conn, f"""
            SELECT name, MAX(completed_time) AS finished
            FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
                     SCHEDULED_TIME_RANGE_START => DATEADD('minute', -30,
                                                           CURRENT_TIMESTAMP())))
            WHERE schema_name = 'MARTS'
              AND state IN ('SUCCEEDED', 'FAILED')
              AND name IN ({', '.join(f"'{t}'" for t in LEAF_TASKS)})
            GROUP BY name
        """)
        finished = {str(r["NAME"]) for r in done
                    if r["FINISHED"] and r["FINISHED"] >= t_trigger}
        if waiting_on <= finished:
            ok("every leaf of the graph has completed a run since the trigger")
            return True
        still = sorted(waiting_on - finished)
        info(f"waiting on {', '.join(still)}")
        time.sleep(20)
    warn(f"graph still running after {timeout}s — leaving it to finish on its "
         f"own schedule")
    return False


def coverage(conn) -> None:
    """What the write-up is allowed to claim, per syndrome.

    Printed because two of the six patterns do not fire on this corpus, and the
    honest thing to do with that is say which two rather than let a reader
    assume six-for-six from a full-looking chart.
    """
    header("Syndrome coverage")
    rowset = q(conn, """
        SELECT c.syndrome_code, c.syndrome_name, c.body_system,
               COUNT(m.match_id)             AS findings,
               COUNT(DISTINCT m.dog_id)      AS dogs
        FROM REF.SYNDROME_CATALOGUE c
        LEFT JOIN MARTS.SYNDROME_MATCHES m ON m.syndrome_code = c.syndrome_code
        GROUP BY 1, 2, 3 ORDER BY 1
    """)
    for r in rowset:
        n = int(r["FINDINGS"] or 0)
        mark = "ok  " if n else "none"
        print(f"  [{mark}] {r['SYNDROME_CODE']:<3} {str(r['SYNDROME_NAME'])[:38]:<40}"
              f"{n:>5} findings   {int(r['DOGS'] or 0):>3} dogs")
    silent = [r["SYNDROME_CODE"] for r in rowset if not int(r["FINDINGS"] or 0)]
    if silent:
        print()
        info(f"{', '.join(silent)} did not fire on this corpus. That is a result, "
             f"not a gap to fill — the sequence is simply not in these sessions.")
    print()
    info("A REPLAY DOES NOT MANUFACTURE FINDINGS, and it should not. The bulk "
         "path has already scanned all six patterns over the complete corpus, "
         "so the finding count is a property of the data rather than of how "
         "long this ran. What the feed adds is fresh epochs, the full pack on "
         "the live path, and a pipeline with something to do — replaying far "
         "enough to re-detect the same events under new timestamps would just "
         "be counting the same dog-seconds twice.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Populate the warehouse from the real corpus and drive the DAG.")
    # SIZING, BECAUSE THE DEFAULTS LOOK TIMID AND ARE NOT.
    #
    # Rows per batch = speed x 8s x dogs x 100 Hz. At the defaults that is
    # 2 x 8 x 45 x 100 = 72,000 rows every eight seconds, ~2.2M over the run —
    # which puts all 45 dogs on the live feed and re-anchors it to now without
    # doubling a landing table that every Dynamic Table above it re-reads in
    # FULL on a one-minute lag. Turn --speed up and the live table grows
    # linearly; so does every refresh that reads it, and the Pipeline tab will
    # tell you about it within a minute.
    ap.add_argument("--minutes", type=float, default=4.0,
                    help="wall-clock minutes of replay (default 4)")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="dog-time seconds per wall second (default 2)")
    ap.add_argument("--dogs", type=int, default=45,
                    help="how many dogs to put on the live feed (default 45, the pack)")
    ap.add_argument("--reset", action="store_true",
                    help="truncate the live table first. Destructive: the epochs "
                         "derived from the current live rows go with it.")
    ap.add_argument("--no-replay", action="store_true",
                    help="skip the feed; just drive the DAG over what is already here")
    ap.add_argument("--skip-tasks", action="store_true",
                    help="do not trigger the task graph")
    ap.add_argument("--census", action="store_true",
                    help="print the census and exit without changing anything")
    args = ap.parse_args()

    load_env()
    conn = connect()
    try:
        before = census(conn)
        if args.census:
            print_census(before)
            f = freshness(conn)
            header("Freshness")
            print(f"  newest sample   {f.get('RAW_MAX')}")
            print(f"  newest epoch    {f.get('EPOCH_MAX')}")
            print(f"  last dog-minute {int(f.get('LAST_MINUTE') or 0):,} rows")
            coverage(conn)
            return 0

        print_census(before)

        if args.reset:
            warn("--reset truncates RAW.COLLAR_TELEMETRY. Every epoch derived "
                 "from the current live rows disappears with it, and the "
                 "findings built on those epochs go on the next scan.")

        if not args.no_replay:
            rc = replay(args.minutes, args.speed, args.dogs, args.reset)
            if rc != 0:
                warn(f"replay exited {rc} — continuing with what landed")
        else:
            info("--no-replay: using the rows already in the live table")

        wait_for_dag(conn)

        if not args.skip_tasks:
            run_graph(conn)
            # The syndrome scan, the notes and the enqueue all write on the way
            # through the graph; the marts they write into are plain tables, so
            # there is nothing further to wait for once the graph is idle.
        else:
            info("--skip-tasks: the 2-minute schedule will pick this up")

        after = census(conn)
        print_census(after, before)
        coverage(conn)

        f = freshness(conn)
        header("Freshness")
        print(f"  newest sample   {f.get('RAW_MAX')}")
        print(f"  newest epoch    {f.get('EPOCH_MAX')}")
        print(f"  last dog-minute {int(f.get('LAST_MINUTE') or 0):,} rows")

        lagging = q(conn, """
            SELECT schema_name, object_name, target_lag_sec, mean_lag_sec
            FROM MARTS.V_DAG_LAG
            WHERE target_lag_sec > 0 AND mean_lag_sec > target_lag_sec
            ORDER BY mean_lag_sec / target_lag_sec DESC
        """)
        if lagging:
            header("Dynamic Tables behind their declared lag")
            for r in lagging:
                print(f"  {r['SCHEMA_NAME']}.{r['OBJECT_NAME']:<24} "
                      f"mean {float(r['MEAN_LAG_SEC']):>9,.0f}s "
                      f"of {float(r['TARGET_LAG_SEC']):>6,.0f}s target")
            info("more live rows means a longer FULL refresh. Replay a smaller "
                 "slice, or leave it a few minutes to settle.")

        header("Next")
        print("  Live feed for screenshots, in a second terminal:")
        print("      python ingest/replay.py --speed 60 --dogs 12")
        print("  On-chain attestations for the queue this run enqueued:")
        print("      npm run bridge")
        print("  Redeploy the dashboard if you changed it:")
        print("      python scripts/deploy_streamlit.py")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
