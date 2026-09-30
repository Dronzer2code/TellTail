#!/usr/bin/env python3
"""
Inject a LABELLED deterioration so the detector fires on camera without lying.

    python scripts/demo_spike.py --dog 7 --syndrome S1
    python scripts/demo_spike.py --dog 7 --anomaly
    python scripts/demo_spike.py --clean

How it stays honest, which is the entire point:

  * every injected row carries  is_synthetic = TRUE  in RAW.COLLAR_TELEMETRY.
  * the flag propagates: STAGING.EPOCH_FEATURES aggregates it with BOOLOR_AGG,
    MARTS.EPOCH_STATES carries it, ML.ACTIVITY_HISTORY carries it.
  * DETECTION sees it (ML.V_ACTIVITY_DETECT has no synthetic filter).
  * TRAINING never fits it (ML.V_ACTIVITY_TRAIN excludes it, and so does the
    self-baseline in MARTS.ACTIVITY_BASELINE — otherwise the detector would
    learn that the spike is normal and the demo would silently do nothing).
  * the dashboard prints "SYNTHETIC" on any panel showing it.
  * --clean removes every trace.

What is injected is not a state label — it is 100 Hz accelerometer and gyroscope
SIGNAL, synthesised with the physical characteristics of the target behaviour.
The real feature layer computes real features from it, the real classifier or
the real threshold ladder assigns the state, and the real MATCH_RECOGNIZE finds
the sequence. Nothing is written directly into MARTS. The whole DAG does the
work, which is what makes the demo a demonstration rather than a puppet show.
"""
from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from _common import connect, die, header, info, load_env, ok, q, warn  # noqa: E402

SR = 100  # Hz, matching the corpus

# ---------------------------------------------------------------------------
# Signal recipes.
#
# Each state is a set of oscillator parameters for the neck and back sensors.
# The discriminating property is the PHASE RELATIONSHIP between them:
#
#   locomotion       both sensors oscillate together        -> CORR near +1
#   neck-dominant    neck oscillates, back nearly static    -> CORR near 0
#
# which is precisely what STAGING.EPOCH_FEATURES.neck_back_corr measures, and
# why a synthetic scratch bout is classified as a scratch bout rather than
# being asserted as one.
# ---------------------------------------------------------------------------
# Parameters, and what each one is doing:
#   neck_amp/back_amp  oscillation amplitude in g. Because the oscillation adds
#                      to gravity in quadrature, mean|a| ≈ sqrt(1 + 0.65·amp²);
#                      crossing the WALK/TROT/GALLOP magnitude thresholds needs
#                      much larger amplitudes than intuition suggests.
#   coupled            do both sensors share phase? This is THE property. Coupled
#                      -> CORR near +1 -> whole-body locomotion. Decoupled -> CORR
#                      near 0 -> the neck moved and the body did not.
#   yaw                sustained gyro-z, i.e. turning consistently one way.
#   yaw_osc            oscillating gyro-z. Held at an integer frequency so it
#                      integrates to zero over a one-second epoch regardless of
#                      phase — otherwise yaw_consistency depends on where the
#                      epoch boundary happens to fall, and CIRCLE fires on
#                      everything.
#   pitch_offset       constant head-down tilt (SNIFF).
#   pitch_drift        linear tilt ACROSS the second, so pitch variance rises
#                      while magnitude stays low (SLOW_TRANSITION).
#
# Calibrated against the thresholds in REF.PARAMS by tests/test_demo_signal.py,
# which fails if any recipe stops producing the state it claims.
YAW_OSC_HZ = 2.0        # integer cycles per epoch => mean yaw contribution ~0
GYRO_DPS   = 180.0      # recipe yaw is dimensionless shape; the corpus is deg/s

RECIPES: dict[str, dict] = {
    "REST":            dict(neck_amp=0.010, back_amp=0.010, freq=0.30, coupled=True,
                            yaw=0.00, yaw_osc=0.01, pitch_offset=0.00, pitch_drift=0.00),
    "STAND":           dict(neck_amp=0.060, back_amp=0.055, freq=0.80, coupled=True,
                            yaw=0.00, yaw_osc=0.03, pitch_offset=0.00, pitch_drift=0.00),
    # SIT is model-only: the rules fallback cannot separate sitting from standing
    # without a per-dog posture reference. Kept so --syndrome sequences can use
    # it once ML.CLASSIFICATION is trained.
    "SIT":             dict(neck_amp=0.040, back_amp=0.035, freq=0.60, coupled=True,
                            yaw=0.00, yaw_osc=0.02, pitch_offset=0.00, pitch_drift=0.00),
    "WALK":            dict(neck_amp=0.620, back_amp=0.580, freq=2.20, coupled=True,
                            yaw=0.02, yaw_osc=0.06, pitch_offset=0.00, pitch_drift=0.00),
    "TROT":            dict(neck_amp=1.000, back_amp=0.940, freq=3.40, coupled=True,
                            yaw=0.02, yaw_osc=0.06, pitch_offset=0.00, pitch_drift=0.00),
    "GALLOP":          dict(neck_amp=1.750, back_amp=1.650, freq=4.20, coupled=True,
                            yaw=0.03, yaw_osc=0.08, pitch_offset=0.00, pitch_drift=0.00),
    # head down, body still: decoupled, modest neck activity, negative pitch
    "SNIFF":           dict(neck_amp=0.600, back_amp=0.045, freq=1.40, coupled=False,
                            yaw=0.04, yaw_osc=0.10, pitch_offset=-0.55, pitch_drift=0.00),
    # neck-dominant: the back barely moves, so the two channels decouple
    # SCRATCH and SHAKE are separated by variance alone, and the bands are
    # adjacent (SCRATCH 0.40–0.90, SHAKE above 0.90). Both amplitudes are set to
    # land mid-band rather than near an edge, so a threshold nudge in REF.PARAMS
    # does not silently turn a head shake into a scratch bout — which would
    # break S1, whose entire signal is the ALTERNATION of the two.
    "SCRATCH":         dict(neck_amp=2.200, back_amp=0.050, freq=7.00, coupled=False,
                            yaw=0.00, yaw_osc=0.30, pitch_offset=0.00, pitch_drift=0.00),
    "SHAKE":           dict(neck_amp=4.400, back_amp=0.090, freq=11.0, coupled=False,
                            yaw=0.00, yaw_osc=0.80, pitch_offset=0.00, pitch_drift=0.00),
    # derived-state geometry
    "PACE":            dict(neck_amp=0.620, back_amp=0.580, freq=2.20, coupled=True,
                            yaw=0.00, yaw_osc=0.90, pitch_offset=0.00, pitch_drift=0.00),
    "CIRCLE":          dict(neck_amp=0.120, back_amp=0.100, freq=1.20, coupled=True,
                            yaw=0.85, yaw_osc=0.05, pitch_offset=0.00, pitch_drift=0.00),
    # still, but not lying: pitch variance above the REST ceiling and below the
    # SLOW_TRANSITION floor, so the rules classifier says STAND and the context
    # rung can promote it to PAUSE between two locomotion epochs
    "PAUSE":           dict(neck_amp=0.085, back_amp=0.080, freq=0.50, coupled=True,
                            yaw=0.00, yaw_osc=0.02, pitch_offset=0.00, pitch_drift=0.00),
    "SLOW_TRANSITION": dict(neck_amp=0.090, back_amp=0.080, freq=0.45, coupled=True,
                            yaw=0.05, yaw_osc=0.06, pitch_offset=0.00, pitch_drift=1.10),
}

# Sequences that satisfy each syndrome's tuned pattern, with slack so a single
# smoothed epoch cannot break the quantifier floor.
SIGNATURES: dict[str, list[tuple[str, int]]] = {
    "S1": [("REST", 6), ("SHAKE", 1), ("SCRATCH", 5), ("SHAKE", 1), ("SCRATCH", 4),
           ("REST", 4)],
    "S2": [("WALK", 5), ("PAUSE", 1), ("WALK", 3), ("PAUSE", 1), ("WALK", 2),
           ("PAUSE", 1), ("WALK", 3)],
    "S3": [("TROT", 5), ("REST", 8), ("TROT", 2), ("REST", 12), ("STAND", 3)],
    "S4": [("REST", 14), ("SLOW_TRANSITION", 1), ("STAND", 1), ("REST", 14)],
    "S5": [("STAND", 1), ("PACE", 6), ("STAND", 1), ("PACE", 6), ("STAND", 2)],
    "S6": [("SNIFF", 7), ("CIRCLE", 3), ("SNIFF", 7), ("STAND", 2)],
}


def synth_second(state: str, t_offset: float, rng: np.random.Generator) -> np.ndarray:
    """One second of 12-channel IMU signal for a state. Shape (100, 12).

    Channel order: neck ax ay az gx gy gz, back ax ay az gx gy gz.
    """
    r = RECIPES[state]
    t = np.arange(SR) / SR
    phase = 2 * math.pi * r["freq"] * (t + t_offset)

    # A coupled pair shares phase (whole-body translation); a decoupled pair does
    # not, and the back channel carries only its own small independent motion.
    back_phase = phase if r["coupled"] else 2 * math.pi * 0.7 * (t + t_offset * 1.7)

    n_amp, b_amp = r["neck_amp"], r["back_amp"]

    # Sensor noise scales with the motion rather than sitting at a fixed floor.
    # A fixed floor swamps a resting dog: at REST the whole signal is smaller
    # than the noise, pitch variance climbs above its ceiling, and a sleeping
    # dog classifies as a slow postural transition.
    def noise(amp: float) -> np.ndarray:
        return rng.normal(0.0, max(0.004, 0.03 * amp), SR)

    # Gravity sits on z. pitch_drift tilts the neck sensor across the second,
    # which is what makes SLOW_TRANSITION show high pitch variance at low
    # magnitude — a dog levering itself up rather than springing.
    tilt = r["pitch_drift"] * (t - 0.5) + r["pitch_offset"]

    neck_ax = n_amp * np.sin(phase) + tilt + noise(n_amp)
    neck_ay = n_amp * 0.55 * np.cos(phase * 1.3) + noise(n_amp)
    neck_az = 1.0 + n_amp * 0.40 * np.sin(phase * 0.9) + noise(n_amp)

    back_ax = b_amp * np.sin(back_phase) + noise(b_amp)
    back_ay = b_amp * 0.55 * np.cos(back_phase * 1.3) + noise(b_amp)
    back_az = 1.0 + b_amp * 0.40 * np.sin(back_phase * 0.9) + noise(b_amp)

    # Integer cycles per epoch, so the oscillating part integrates to zero over
    # the second no matter where the epoch boundary falls. Without that,
    # yaw_consistency = |mean| / mean|·| is an artefact of phase and CIRCLE
    # fires on scratching.
    yaw = r["yaw"] + r["yaw_osc"] * np.sin(2 * math.pi * YAW_OSC_HZ * (t + t_offset))
    # GYRO_DPS puts the synthetic gyro in the corpus's units. The recipes above
    # describe SHAPE — direction, steadiness, oscillation — on a dimensionless
    # 0-1 scale; the corpus reports angular rate in DEGREES PER SECOND, where a
    # resting dog reads 2.6 and a head shake reads 186. Without the conversion
    # the synthetic feed sat ~180x below the real one, so every geometry
    # threshold that is correct for real data rejected the demo signal, and
    # CIRCLE and PACE could not be synthesised at all.
    neck_gx = GYRO_DPS * r["yaw_osc"] * np.sin(phase) + noise(0.05)
    neck_gy = GYRO_DPS * r["yaw_osc"] * np.cos(phase) + noise(0.05)
    neck_gz = GYRO_DPS * yaw + noise(0.05)
    back_gx = GYRO_DPS * r["yaw_osc"] * 0.5 * np.sin(back_phase) + noise(0.05)
    back_gy = GYRO_DPS * r["yaw_osc"] * 0.5 * np.cos(back_phase) + noise(0.05)
    back_gz = GYRO_DPS * yaw + noise(0.05)   # yaw is a whole-body property

    return np.column_stack([
        neck_ax, neck_ay, neck_az, neck_gx, neck_gy, neck_gz,
        back_ax, back_ay, back_az, back_gx, back_gy, back_gz,
    ])


def build_rows(dog_id: int, test_num: int, sequence: list[tuple[str, int]],
               start_t_sec: float, anchor, batch_id: str, seed: int):
    rng = np.random.default_rng(seed)
    rows: list[tuple] = []
    t_sec = start_t_sec
    plan: list[tuple[str, int]] = []

    for state, seconds in sequence:
        for _ in range(seconds):
            block = synth_second(state, t_sec, rng)
            for i in range(SR):
                rows.append((
                    dog_id, test_num, round(t_sec + i / SR, 4),
                    *(float(x) for x in block[i]),
                    None, None, None, None, "SYNTHETIC_DEMO",
                    batch_id,
                ))
            plan.append((state, 1))
            t_sec += 1.0
    return rows, t_sec


def insert_rows(conn, rows: list[tuple], anchor_t0: float, anchor_ts) -> int:
    sql = f"""
        INSERT INTO RAW.COLLAR_TELEMETRY
            (dog_id, test_num, t_sec, sample_ts,
             neck_ax, neck_ay, neck_az, neck_gx, neck_gy, neck_gz,
             back_ax, back_ay, back_az, back_gx, back_gy, back_gz,
             label_primary, label_secondary, label_tertiary, point_event, task,
             _batch_id, is_replay, is_synthetic)
        SELECT
            column1, column2, column3,
            DATEADD('millisecond', CAST((column3 - {anchor_t0}) * 1000 AS NUMBER),
                    '{anchor_ts}'::TIMESTAMP_NTZ),
            column4, column5, column6, column7, column8, column9,
            column10, column11, column12, column13, column14, column15,
            column16, column17, column18, column19, column20,
            column21, TRUE, TRUE
        FROM VALUES {{VALUES}}
    """
    # Chunked so a single statement stays a sane size.
    total = 0
    CH = 2000
    with conn.cursor() as cur:
        for i in range(0, len(rows), CH):
            chunk = rows[i : i + CH]
            placeholders = ",".join(
                "(" + ",".join(["%s"] * len(chunk[0])) + ")" for _ in chunk
            )
            flat = [v for r in chunk for v in r]
            cur.execute(sql.replace("{VALUES}", placeholders), flat)
            total += cur.rowcount or 0
            print(f"    inserted {total:,} / {len(rows):,} samples", end="\r", flush=True)
    print(" " * 60, end="\r")
    return total


def clean(conn) -> None:
    header("Removing every synthetic row")
    n = q(conn, "SELECT COUNT(*) AS n FROM RAW.COLLAR_TELEMETRY WHERE is_synthetic")[0]["N"]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM RAW.COLLAR_TELEMETRY WHERE is_synthetic")
        cur.execute("DELETE FROM ML.ACTIVITY_HISTORY WHERE is_synthetic")
        cur.execute("DELETE FROM RAW.INGEST_LOG WHERE is_synthetic")
    ok(f"removed {int(n):,} synthetic samples")
    info("the Dynamic Tables will drop the derived epochs on their next refresh "
         "(one minute). MARTS.SYNDROME_MATCHES clears on the next task run.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inject a labelled synthetic deterioration.")
    ap.add_argument("--dog", type=int, help="dog_id to inject into")
    ap.add_argument("--syndrome", choices=sorted(SIGNATURES), default="S1")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the signature N times")
    ap.add_argument("--anomaly", action="store_true",
                    help="also inject a sustained activity surge for ANOMALY_DETECTION")
    ap.add_argument("--anomaly-seconds", type=int, default=180)
    ap.add_argument("--clean", action="store_true", help="remove everything synthetic and exit")
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()

    load_env()
    conn = connect()
    try:
        if args.clean:
            clean(conn)
            return 0
        if not args.dog:
            die("--dog is required (or pass --clean)")

        # Land the injection immediately after this dog's existing live data, so
        # it reads as a continuation of the feed rather than an interleave.
        cur_state = q(conn, """
            SELECT MAX(t_sec) AS t_max, MAX(sample_ts) AS ts_max,
                   MIN(t_sec) AS t_min, MIN(sample_ts) AS ts_min,
                   COUNT(*) AS n, MAX(test_num) AS test_num
            FROM RAW.COLLAR_TELEMETRY WHERE dog_id = %s
        """, (args.dog,))[0]

        if not cur_state["N"]:
            die(f"dog {args.dog} has no live rows yet. Start the replayer first:\n"
                f"    python ingest/replay.py --speed 60 --dog-ids {args.dog}")

        start_t = float(cur_state["T_MAX"]) + 1.0
        anchor_t0 = float(cur_state["T_MIN"])
        anchor_ts = cur_state["TS_MIN"]
        test_num = int(cur_state["TEST_NUM"] or 1)
        batch_id = f"SPIKE_{uuid.uuid4().hex[:8]}"

        header(f"Injecting {args.syndrome} into dog {args.dog}")
        sequence = SIGNATURES[args.syndrome] * args.repeat
        pattern = q(conn, "SELECT syndrome_name, pattern_text FROM REF.SYNDROME_CATALOGUE "
                          "WHERE syndrome_code = %s", (args.syndrome,))
        if pattern:
            print(f"  target    : {pattern[0]['SYNDROME_NAME']}")
            print(f"  pattern   : {pattern[0]['PATTERN_TEXT']}")
        print(f"  sequence  : {' '.join(f'{s}x{n}' for s, n in sequence)}")
        print(f"  seconds   : {sum(n for _, n in sequence)}")
        print(f"  batch     : {batch_id}   is_synthetic = TRUE on every row")
        print()

        rows, end_t = build_rows(args.dog, test_num, sequence, start_t,
                                 anchor_ts, batch_id, args.seed)
        n = insert_rows(conn, rows, anchor_t0, anchor_ts)
        ok(f"{n:,} synthetic samples ({sum(x for _, x in sequence)} seconds of dog time)")

        if args.anomaly:
            header("Injecting an activity surge for ANOMALY_DETECTION")
            surge = [("GALLOP", args.anomaly_seconds)]
            rows2, end_t = build_rows(args.dog, test_num, surge, end_t + 1.0,
                                      anchor_ts, batch_id, args.seed + 1)
            n2 = insert_rows(conn, rows2, anchor_t0, anchor_ts)
            ok(f"{n2:,} samples of sustained galloping ({args.anomaly_seconds}s)")

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO RAW.INGEST_LOG
                    (batch_id, run_id, n_rows, window_start, window_end, speed, dogs, is_synthetic)
                SELECT %s, 'DEMO_SPIKE', %s,
                       (SELECT MIN(sample_ts) FROM RAW.COLLAR_TELEMETRY WHERE _batch_id = %s),
                       (SELECT MAX(sample_ts) FROM RAW.COLLAR_TELEMETRY WHERE _batch_id = %s),
                       1, PARSE_JSON(%s), TRUE
            """, (batch_id, len(rows), batch_id, batch_id, f"[{args.dog}]"))

        header("What happens next, on its own")
        print("""  1. STAGING.EPOCH_FEATURES refreshes within its 1-minute target lag and
     computes real features from the synthetic signal. Watch neck_back_corr
     collapse on the SCRATCH and SHAKE seconds — that is the feature working.
  2. MARTS.EPOCH_STATES classifies each epoch. Nothing was labelled by hand.
  3. MARTS.T_SYNDROMES runs within 2 minutes and MATCH_RECOGNIZE finds the
     sequence, if the states came out the way the physics says they should.
  4. ML.T_ML flags the surge, because detection sees synthetic rows and
     training does not.

  Verify:
     SELECT epoch_ts, state, state_source, ROUND(neck_back_corr,3), is_synthetic
     FROM MARTS.EPOCH_STATES WHERE dog_id = %d ORDER BY epoch_ts DESC LIMIT 40;

     SELECT * FROM MARTS.SYNDROME_MATCHES WHERE dog_id = %d ORDER BY onset_ts DESC;

  Undo:
     python scripts/demo_spike.py --clean
""" % (args.dog, args.dog))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
