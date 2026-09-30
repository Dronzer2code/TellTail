#!/usr/bin/env python3
"""
The replayer. Turns a Kaggle download into a live feed.

    python ingest/replay.py                    # wall clock, all dogs
    python ingest/replay.py --speed 60         # one minute of dog time per second
    python ingest/replay.py --dogs 12          # keep the pack view legible
    python ingest/replay.py --speed 60 --dogs 12 --loop

TWO TIME BASES, and confusing them is the bug that makes every epoch contain
6,000 samples instead of 100:

  dog time    t_sec, seconds from session start. This is what sample_ts encodes,
              always 1:1. An epoch is one second of DOG time and holds ~100
              samples no matter what --speed is set to.

  wall clock  how fast the rows are pushed. --speed 60 means each 8-second real
              sleep advances the cursor by 480 seconds of dog time.

So at --speed 60 the pipeline's notion of "now" (MAX(sample_ts)) runs ahead of
the wall clock, deliberately. Everything downstream measures staleness against
MAX(epoch_ts) rather than CURRENT_TIMESTAMP() for exactly this reason.

NO DATA ROUND-TRIP. The window is moved with a server-side INSERT ... SELECT
from RAW.COLLAR_TELEMETRY_BULK. Python decides which window and when; Snowflake
moves the rows. Pulling 10.6M rows through pandas to push them back would be a
transformation pipeline outside the warehouse, which this build does not do.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _common import (  # noqa: E402
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

BATCH_WALL_SECONDS = 8          # one micro-batch every 8 real seconds
_STOP = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _STOP
    if _STOP:
        raise KeyboardInterrupt
    _STOP = True
    print("\n  stopping after this batch (ctrl-c again to force)…", flush=True)


def pick_dogs(conn, n: int | None, explicit: list[int] | None) -> list[int]:
    if explicit:
        return explicit
    rows = q(conn, """
        SELECT dog_id, COUNT(*) AS n
        FROM RAW.COLLAR_TELEMETRY_BULK
        GROUP BY dog_id
        ORDER BY n DESC
    """)
    if not rows:
        die("RAW.COLLAR_TELEMETRY_BULK is empty. Run scripts/load_raw.py first.")
    dogs = [int(r["DOG_ID"]) for r in rows]
    if n:
        # Spread the subset across the roster rather than taking the top N by
        # row count, which would be the same handful of long sessions.
        step = max(1, len(dogs) // n)
        dogs = dogs[::step][:n]
    return dogs


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay the corpus into the live landing table.")
    ap.add_argument("--speed", type=float, default=None,
                    help="dog-time seconds per wall-clock second (default: $REPLAY_SPEED or 1)")
    ap.add_argument("--dogs", type=int, default=None,
                    help="restrict to N dogs. 45 cards is a wall; 12 is a dashboard.")
    ap.add_argument("--dog-ids", type=str, default=None, help="explicit comma-separated ids")
    ap.add_argument("--test-num", type=int, default=None, help="restrict to one session")
    ap.add_argument("--start-at", type=float, default=0.0, help="starting t_sec")
    ap.add_argument("--loop", action="store_true", help="restart from the beginning at the end")
    ap.add_argument("--reset", action="store_true", help="truncate the live table first")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    load_env()
    speed = args.speed if args.speed is not None else float(env("REPLAY_SPEED", "1"))
    if speed <= 0:
        die("--speed must be positive")

    signal.signal(signal.SIGINT, _handle_sigint)
    conn = connect()
    run_id = f"REPLAY_{uuid.uuid4().hex[:8]}"

    try:
        explicit = [int(x) for x in args.dog_ids.split(",")] if args.dog_ids else None
        dogs = pick_dogs(conn, args.dogs, explicit)
        dog_list = ", ".join(str(d) for d in dogs)

        bounds = q(conn, f"""
            SELECT MIN(t_sec) AS t0, MAX(t_sec) AS t1, COUNT(*) AS n
            FROM RAW.COLLAR_TELEMETRY_BULK
            WHERE dog_id IN ({dog_list})
              {f'AND test_num = {args.test_num}' if args.test_num else ''}
        """)[0]
        t0, t1 = float(bounds["T0"] or 0), float(bounds["T1"] or 0)
        total_rows = int(bounds["N"] or 0)
        if not total_rows:
            die("no rows matched the dog/session selection")

        if args.reset:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE RAW.COLLAR_TELEMETRY")
                cur.execute("DELETE FROM RAW.INGEST_LOG")
            warn("live table truncated")

        dog_seconds_per_batch = BATCH_WALL_SECONDS * speed

        header(f"Replay {run_id}")
        print(f"  dogs        : {len(dogs)}  ({dog_list[:70]}{'…' if len(dog_list) > 70 else ''})")
        print(f"  corpus      : {total_rows:,} samples, t_sec {t0:.2f} .. {t1:.2f}")
        print(f"  speed       : {speed}x  ->  {dog_seconds_per_batch:.0f}s of dog time "
              f"per {BATCH_WALL_SECONDS}s batch")
        print(f"  wall time   : ~{(t1 - t0) / speed / 60:.1f} min to replay "
              f"{(t1 - t0) / 3600:.1f}h of dog time")
        print(f"  anchor      : sample_ts starts at CURRENT_TIMESTAMP()")
        print()

        # Resolve the anchor ONCE. If each batch re-evaluated CURRENT_TIMESTAMP()
        # every batch would restart sample_ts from a new origin and the timeline
        # would sawtooth backwards.
        anchor = q(conn, "SELECT CURRENT_TIMESTAMP()::TIMESTAMP_NTZ AS a")[0]["A"]
        info(f"replay anchor: {anchor}")

        cursor_t = max(t0, args.start_at)
        batch_no = 0
        rows_total = 0
        started = time.time()

        while not _STOP:
            if args.max_batches and batch_no >= args.max_batches:
                info(f"reached --max-batches {args.max_batches}")
                break

            window_lo = cursor_t
            window_hi = cursor_t + dog_seconds_per_batch
            if window_lo >= t1:
                if args.loop:
                    info("end of corpus — looping")
                    cursor_t = t0
                    continue
                ok("end of corpus")
                break

            batch_no += 1
            batch_id = f"{run_id}_B{batch_no:05d}"
            t_batch = time.perf_counter()

            # Server-side. Python picks the window; Snowflake moves the rows.
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO RAW.COLLAR_TELEMETRY
                        (dog_id, test_num, t_sec, sample_ts,
                         neck_ax, neck_ay, neck_az, neck_gx, neck_gy, neck_gz,
                         back_ax, back_ay, back_az, back_gx, back_gy, back_gz,
                         label_primary, label_secondary, label_tertiary, point_event, task,
                         raw_payload, _batch_id, is_replay, is_synthetic)
                    SELECT
                        dog_id, test_num, t_sec,
                        -- dog time -> wall-clock replay epoch, 1:1. Millisecond
                        -- resolution so 100 Hz samples stay distinct.
                        DATEADD('millisecond',
                                CAST((t_sec - {t0}) * 1000 AS NUMBER),
                                '{anchor}'::TIMESTAMP_NTZ)                      AS sample_ts,
                        neck_ax, neck_ay, neck_az, neck_gx, neck_gy, neck_gz,
                        back_ax, back_ay, back_az, back_gx, back_gy, back_gz,
                        label_primary, label_secondary, label_tertiary, point_event, task,
                        OBJECT_CONSTRUCT(
                            'dog_id', dog_id, 'test_num', test_num, 't_sec', t_sec,
                            'label_primary', label_primary,
                            'label_secondary', label_secondary,
                            'label_tertiary', label_tertiary,
                            'point_event', point_event, 'task', task,
                            'shard', _shard
                        )                                                       AS raw_payload,
                        '{batch_id}', TRUE, FALSE
                    FROM RAW.COLLAR_TELEMETRY_BULK
                    WHERE dog_id IN ({dog_list})
                      AND t_sec >= {window_lo} AND t_sec < {window_hi}
                      {f'AND test_num = {args.test_num}' if args.test_num else ''}
                """)
                n_rows = cur.rowcount or 0

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO RAW.INGEST_LOG
                        (batch_id, run_id, n_rows, window_start, window_end, speed, dogs)
                    SELECT %s, %s, %s,
                           (SELECT MIN(sample_ts) FROM RAW.COLLAR_TELEMETRY WHERE _batch_id = %s),
                           (SELECT MAX(sample_ts) FROM RAW.COLLAR_TELEMETRY WHERE _batch_id = %s),
                           %s, PARSE_JSON(%s)
                """, (batch_id, run_id, n_rows, batch_id, batch_id, speed, str(dogs)))

            rows_total += n_rows
            cursor_t = window_hi
            elapsed = time.perf_counter() - t_batch
            pct = 100.0 * (cursor_t - t0) / max(t1 - t0, 1e-9)

            print(f"  {batch_no:>4}  t_sec {window_lo:>9.1f}→{window_hi:<9.1f}  "
                  f"{n_rows:>7,} rows  {elapsed:5.2f}s  "
                  f"[{pct:5.1f}%  {rows_total:,} total]", flush=True)

            sleep_for = max(0.0, BATCH_WALL_SECONDS - elapsed)
            if elapsed > BATCH_WALL_SECONDS:
                warn(f"    batch took {elapsed:.1f}s > {BATCH_WALL_SECONDS}s budget — "
                     f"the feed is falling behind. Lower --speed or --dogs.")
            slept = 0.0
            while slept < sleep_for and not _STOP:
                step = min(0.25, sleep_for - slept)
                time.sleep(step)
                slept += step

        header("Replay finished")
        ok(f"{batch_no} batches, {rows_total:,} rows, "
           f"{(time.time() - started) / 60:.1f} min wall clock")
        info("the Dynamic Table DAG has a one-minute target lag; "
             "give it a minute before expecting epochs downstream")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
