#!/usr/bin/env python3
"""
GATE A — profile the dataset before a single line of DDL is written.

Reads Data_description.txt, then streams DogMoveData.csv in chunks and reports
what is ACTUALLY in the file: exact column names, the distinct values of every
label-ish column with counts, per-dog row counts, and the semantics of the time
column. It then proposes an ethogram vocabulary from the real labels and writes
ref/column_map.json, which every downstream script loads.

It answers one question loudly, because two of the six syndromes depend on it:
    are SCRATCH and SHAKE present as first-class labels, or must they be derived?

    python scripts/profile_dataset.py
    python scripts/profile_dataset.py --sample 2000000     # faster first pass
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from _common import (  # noqa: E402
    DATA_DIR,
    REF_DIR,
    die,
    header,
    info,
    ok,
    warn,
)

# ---------------------------------------------------------------------------
# column resolution
#
# Each canonical name owns an ordered list of regex patterns, matched against a
# normalised form of the header (lowercased, non-alphanumerics stripped).
# Earlier patterns score higher. This is a resolver, not an assumption: whatever
# it picks is printed for review and every unmatched column is reported.
# ---------------------------------------------------------------------------

RESOLVERS: dict[str, list[str]] = {
    "dog_id":          [r"^dogid$", r"^dog$", r"^subjectid$", r"^subject$", r"^animalid$"],
    "test_num":        [r"^testnum$", r"^test$", r"^testnumber$", r"^session$", r"^trial$"],
    "t_sec":           [r"^tsec$", r"^timesec$", r"^time$", r"^t$", r"^timestamp$", r"^elapsed.*"],

    "neck_ax":         [r"^aneckx$", r"^neckax$", r"^accneckx$", r"^neckaccx$"],
    "neck_ay":         [r"^anecky$", r"^neckay$", r"^accnecky$", r"^neckaccy$"],
    "neck_az":         [r"^aneckz$", r"^neckaz$", r"^accneckz$", r"^neckaccz$"],
    "neck_gx":         [r"^gneckx$", r"^neckgx$", r"^gyroneckx$", r"^neckgyrox$"],
    "neck_gy":         [r"^gnecky$", r"^neckgy$", r"^gyronecky$", r"^neckgyroy$"],
    "neck_gz":         [r"^gneckz$", r"^neckgz$", r"^gyroneckz$", r"^neckgyroz$"],

    "back_ax":         [r"^abackx$", r"^backax$", r"^accbackx$", r"^backaccx$"],
    "back_ay":         [r"^abacky$", r"^backay$", r"^accbacky$", r"^backaccy$"],
    "back_az":         [r"^abackz$", r"^backaz$", r"^accbackz$", r"^backaccz$"],
    "back_gx":         [r"^gbackx$", r"^backgx$", r"^gyrobackx$", r"^backgyrox$"],
    "back_gy":         [r"^gbacky$", r"^backgy$", r"^gyrobacky$", r"^backgyroy$"],
    "back_gz":         [r"^gbackz$", r"^backgz$", r"^gyrobackz$", r"^backgyroz$"],

    "label_primary":   [r"^behavior1$", r"^behaviour1$", r"^behavior$", r"^behaviour$", r"^label$"],
    "label_secondary": [r"^behavior2$", r"^behaviour2$", r"^label2$"],
    "label_tertiary":  [r"^behavior3$", r"^behaviour3$", r"^label3$"],
    "point_event":     [r"^pointevent$", r"^event$", r"^events$"],
    "task":            [r"^task$", r"^protocol$", r"^activity$"],
}

INFO_RESOLVERS: dict[str, list[str]] = {
    "dog_id":    [r"^dogid$", r"^dog$", r"^subjectid$"],
    "breed":     [r"^breed$", r"^dogbreed$"],
    "sex":       [r"^sex$", r"^gender$"],
    "age_years": [r"^ageyears$", r"^agey$", r"^age$"],
    # This dataset stores age in MONTHS ("Age months"). Resolved as its own
    # canonical column so the unit is explicit and the conversion happens once,
    # in the loader, instead of a 76-year-old Belgian Shepherd reaching the
    # cohort bands.
    "age_months": [r"^agemonths?$", r"^agemo$", r"^agemonth$"],
    "weight_kg": [r"^weight$", r"^weightkg$", r"^masskg$"],
    "height_cm": [r"^height$", r"^heightcm$", r"^witherheight$"],
    "neutered":  [r"^neuteringstatus$", r"^neutered$", r"^neuterstatus$"],
}

# Gender is coded numerically in DogInfo.csv: 1 = female, 2 = male
# (Data_description.txt). Mapped on load so REF.DOG_INFO.sex reads as a label
# and the Drivers tab does not report a dimension called "2".
SEX_CODES = {1: "female", 2: "male", "1": "female", "2": "male"}

# Canonical states TELLTAIL's syndrome catalogue is written against.
ETHOGRAM_STATES = [
    "REST", "SIT", "STAND", "WALK", "TROT", "GALLOP",
    "SNIFF", "PLAY", "SHAKE", "SCRATCH", "PACE", "CIRCLE",
]

# Labels that are NOT postures and must never become an ethogram state.
# Checked before LABEL_HINTS, so a substring cannot sneak them in.
#
#   Panting          a respiratory behaviour, not a posture. In this dataset it
#                    is the PRIMARY annotation for 836K rows while the actual
#                    posture sits in Behavior_2 (48% Sitting, 48% Standing).
#                    Mapping it to a posture would be wrong about half the time;
#                    ML.V_LABELLED_EPOCHS falls back to the secondary column
#                    instead and recovers the true posture.
#   Synchronization  sensor calibration markers, not animal behaviour at all.
#   Bark             a vocalisation point-event with no postural meaning.
LABEL_EXCLUDE = [
    r"^panting$",
    r"synchroni",
    r"^bark$",
    r"^<undefined>$",
]

# Raw-label -> state guesses, applied only to labels actually observed. Anything
# observed and unmapped is reported for a human decision, never dropped quietly.
LABEL_HINTS: list[tuple[str, str]] = [
    (r"gallop",                       "GALLOP"),
    (r"trot",                         "TROT"),
    (r"\bpac(e|ing)\b",               "PACE"),
    (r"walk",                         "WALK"),
    (r"stand",                        "STAND"),
    (r"\bsit",                        "SIT"),
    (r"ly(ing)?.*(chest|down|side)",  "REST"),
    (r"lie[ -]?down",             "REST"),
    (r"^lying$|^lie",                 "REST"),
    (r"rest|sleep",                   "REST"),
    (r"sniff|treat.?search|search",   "SNIFF"),
    (r"eat|drink|chew",               "SNIFF"),
    (r"play|bow|jump|tug",            "PLAY"),
    (r"shak|shudder",                 "SHAKE"),
    (r"scratch|itch",                 "SCRATCH"),
    (r"circl|spin|turn",              "CIRCLE"),
    (r"carry",                        "WALK"),
]

# The two states the syndrome catalogue cannot live without.
SYNDROME_CRITICAL = ["SHAKE", "SCRATCH"]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_columns(
    headers: list[str], resolvers: dict[str, list[str]]
) -> tuple[dict[str, str | None], dict[str, str], list[str]]:
    """Map canonical -> actual header. Returns (mapping, confidence, unmatched)."""
    normed = {h: norm(h) for h in headers}
    mapping: dict[str, str | None] = {}
    confidence: dict[str, str] = {}
    claimed: set[str] = set()

    for canon, patterns in resolvers.items():
        hit: str | None = None
        rank = -1
        for r_i, pat in enumerate(patterns):
            for h in headers:
                if h in claimed:
                    continue
                if re.match(pat, normed[h]):
                    hit, rank = h, r_i
                    break
            if hit:
                break
        mapping[canon] = hit
        if hit:
            claimed.add(hit)
            confidence[canon] = "exact" if rank == 0 else f"alias:{patterns[rank]}"
        else:
            confidence[canon] = "MISSING"

    unmatched = [h for h in headers if h not in claimed]
    return mapping, confidence, unmatched


def propose_state(label: str) -> str | None:
    low = str(label).strip().lower()
    if not low or low in {"nan", "none", "null", "<undefined>", "undefined", "-"}:
        return None
    for pat in LABEL_EXCLUDE:
        if re.search(pat, low):
            return None
    for pat, state in LABEL_HINTS:
        if re.search(pat, low):
            return state
    return None


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------


def profile(csv_path: Path, chunksize: int, sample_rows: int | None) -> dict[str, Any]:
    header("Reading header")
    head = pd.read_csv(csv_path, nrows=5)
    headers = list(head.columns)
    print(f"  {len(headers)} columns:")
    for i, h in enumerate(headers):
        print(f"    {i:>2}. {h!r}   dtype≈{head[h].dtype}")

    mapping, confidence, unmatched = resolve_columns(headers, RESOLVERS)

    header("Column resolution")
    for canon in RESOLVERS:
        actual, conf = mapping[canon], confidence[canon]
        mark = "✓" if actual else "✗"
        print(f"  {mark} {canon:<16} -> {str(actual):<20} [{conf}]")
    if unmatched:
        warn(f"columns in the file with no canonical slot: {unmatched}")
        info("       -> they still land in RAW.COLLAR_TELEMETRY.raw_payload (VARIANT)")

    for req in ("dog_id", "t_sec", "neck_ax", "back_ax"):
        if not mapping.get(req):
            die(f"could not resolve required column '{req}' from {headers}. "
                f"Add a pattern to RESOLVERS in this file and re-run.")

    # ---- which columns are label-ish? object dtype, low cardinality ----------
    label_candidates = [
        h for h in headers
        if head[h].dtype == object or norm(h).startswith(("behavior", "behaviour", "label", "point", "task"))
    ]
    info(f"label-candidate columns: {label_candidates or '(none)'}")

    # ---- stream ------------------------------------------------------------
    header(f"Streaming {csv_path.name} in {chunksize:,}-row chunks")
    value_counts: dict[str, Counter] = {c: Counter() for c in label_candidates}
    dog_rows: Counter = Counter()
    dog_tests: dict[Any, set] = {}
    t_min: dict[Any, float] = {}
    t_max: dict[Any, float] = {}
    t_deltas: Counter = Counter()
    total = 0
    null_counts: Counter = Counter()

    dog_col = mapping["dog_id"]
    t_col = mapping["t_sec"]
    test_col = mapping.get("test_num")

    reader = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
    prev_key: tuple | None = None
    prev_t: float | None = None

    for chunk in reader:
        total += len(chunk)

        for c in label_candidates:
            value_counts[c].update(chunk[c].fillna("<null>").astype(str).value_counts().to_dict())

        for c in chunk.columns:
            null_counts[c] += int(chunk[c].isna().sum())

        grp = chunk.groupby(dog_col, dropna=False)
        for dog, g in grp:
            dog_rows[dog] += len(g)
            if test_col:
                dog_tests.setdefault(dog, set()).update(g[test_col].dropna().unique().tolist())
            tmin, tmax = float(g[t_col].min()), float(g[t_col].max())
            t_min[dog] = min(t_min.get(dog, tmin), tmin)
            t_max[dog] = max(t_max.get(dog, tmax), tmax)

        # sample the inter-sample delta to confirm the stated 100 Hz
        key_cols = [dog_col] + ([test_col] if test_col else [])
        sub = chunk[key_cols + [t_col]].head(20000)
        for row in sub.itertuples(index=False):
            key = tuple(row[:-1])
            t = float(row[-1])
            if prev_key == key and prev_t is not None:
                d = round(t - prev_t, 6)
                if 0 < d < 1.0:
                    t_deltas[d] += 1
            prev_key, prev_t = key, t

        print(f"    {total:>12,} rows", end="\r", flush=True)
        if sample_rows and total >= sample_rows:
            warn(f"stopping early at --sample {sample_rows:,} rows (partial profile)")
            break

    print(f"    {total:>12,} rows  done      ")

    # ---- report ------------------------------------------------------------
    header("Row counts")
    print(f"  total rows        : {total:,}")
    print(f"  distinct dogs     : {len(dog_rows)}")
    if test_col:
        n_repeat = sum(1 for d in dog_tests.values() if len(d) > 1)
        print(f"  dogs with >1 test : {n_repeat}")

    header("Per-dog row counts")
    for dog, n in sorted(dog_rows.items(), key=lambda kv: -kv[1]):
        span = t_max.get(dog, 0) - t_min.get(dog, 0)
        tests = sorted(dog_tests.get(dog, [])) if test_col else []
        print(f"  dog {str(dog):>4}: {n:>10,} rows   t∈[{t_min.get(dog,0):.2f}, "
              f"{t_max.get(dog,0):.2f}]  span={span:,.1f}s  tests={tests}")

    header("Time column semantics")
    if t_deltas:
        common = t_deltas.most_common(5)
        print(f"  modal Δt between consecutive samples (per dog/test):")
        for d, c in common:
            print(f"    Δ={d:<10} n={c:,}   => {1/d if d else 0:,.1f} Hz")
        hz = round(1 / common[0][0]) if common[0][0] else 0
        print(f"  inferred sample rate: ~{hz} Hz")
    else:
        warn("could not infer Δt — time column may not be monotonic within a dog/test")
        hz = 0
    print(f"  semantics: '{t_col}' is seconds from session start, NOT a wall clock.")
    print(f"             replay.py projects it onto a wall-clock epoch as sample_ts.")

    header("Label vocabulary (this is the Gate A decision)")
    observed_states: set[str] = set()
    label_report: dict[str, Any] = {}
    for c in label_candidates:
        vc = value_counts[c]
        real = {k: v for k, v in vc.items() if k not in ("<null>", "nan")}
        print(f"\n  {c}  —  {len(real)} distinct values, "
              f"{vc.get('<null>', 0) + vc.get('nan', 0):,} null")
        entries = []
        for val, cnt in sorted(vc.items(), key=lambda kv: -kv[1]):
            state = propose_state(val)
            if state:
                observed_states.add(state)
            pct = 100.0 * cnt / max(total, 1)
            flag = f"-> {state}" if state else ("(null)" if val in ("<null>", "nan") else "-> UNMAPPED")
            print(f"      {val!r:<28} {cnt:>12,}  {pct:5.2f}%  {flag}")
            entries.append({"value": val, "count": cnt, "pct": round(pct, 4), "state": state})
        label_report[c] = entries

    header("Syndrome feasibility")
    missing = [s for s in SYNDROME_CRITICAL if s not in observed_states]
    if not missing:
        ok("SHAKE and SCRATCH are present as labels. "
           "S1 (otitis) can run on model-derived states end to end.")
    else:
        warn(f"NOT present as labels: {', '.join(missing)}")
        print("""
      This changes the syndrome catalogue design, exactly as the spec warned.
      The fallback is already built and is NOT a quiet relabel:

        MARTS.EPOCH_STATES.state_source = 'HEURISTIC' for these states, derived
        from the neck/back correlation feature (neck-dominant, high-frequency,
        low-correlation epochs are shake/scratch candidates), with a row in
        HONESTY.md and the flag surfaced in the dashboard.

      Affected syndromes: S1 (otitis) directly; S5/S6 depend on PACE/CIRCLE,
      which are derived from gyro yaw regardless of labelling.
        """.rstrip())

    proposed_map = {}
    for c, entries in label_report.items():
        for e in entries:
            if e["value"] not in ("<null>", "nan"):
                proposed_map[e["value"]] = e["state"]

    return {
        "source_file": csv_path.name,
        "columns": {k: v for k, v in mapping.items() if v},
        "missing_columns": [k for k, v in mapping.items() if not v],
        "unmatched_headers": unmatched,
        "column_confidence": confidence,
        "ethogram_map": proposed_map,
        "observed_states": sorted(observed_states),
        "label_report": label_report,
        "heuristic_states_required": missing,
        "observed": {
            "row_count": total,
            "column_count": len(headers),
            "distinct_dogs": len(dog_rows),
            "sample_rate_hz": hz,
            "per_dog_rows": {str(k): v for k, v in sorted(dog_rows.items())},
            "null_counts": {k: v for k, v in null_counts.items() if v},
        },
        "partial": bool(sample_rows and total >= sample_rows),
    }


def profile_info(csv_path: Path) -> dict[str, Any]:
    header(f"Reading {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")
    mapping, confidence, unmatched = resolve_columns(list(df.columns), INFO_RESOLVERS)
    for canon in INFO_RESOLVERS:
        print(f"  {'✓' if mapping[canon] else '✗'} {canon:<12} -> {mapping[canon]}")
    if unmatched:
        info(f"  extra columns: {unmatched}")

    breed_col = mapping.get("breed")
    if breed_col:
        print(f"\n  {df[breed_col].nunique()} distinct breeds")
        for b, c in df[breed_col].value_counts().items():
            print(f"      {b!r:<32} {c}")

    for num in ("age_years", "age_months", "weight_kg", "height_cm"):
        col = mapping.get(num)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            print(f"  {num:<12} mean={s.mean():.2f} min={s.min():.2f} max={s.max():.2f}")

    if mapping.get("age_months") and not mapping.get("age_years"):
        m = df[mapping["age_months"]]
        print(f"  -> age is in MONTHS; the loader divides by 12 "
              f"({m.mean() / 12:.1f}y mean, {m.min() / 12:.1f}–{m.max() / 12:.1f}y range)")

    sex_col = mapping.get("sex")
    if sex_col:
        vals = df[sex_col].value_counts().to_dict()
        print(f"  sex codes    {vals}  -> mapped via SEX_CODES "
              f"{{1: 'female', 2: 'male'}}")

    return {
        "info_columns": {k: v for k, v in mapping.items() if v},
        "info_unmatched": unmatched,
        "info_row_count": len(df),
        "info_breeds": int(df[breed_col].nunique()) if breed_col else None,
        "info_age_unit": "months" if mapping.get("age_months") else "years",
        "info_sex_codes": SEX_CODES,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GATE A: profile the dataset. Writes ref/column_map.json.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--sample", type=int, default=None,
                    help="stop after N rows for a fast first pass (marks the map partial)")
    ap.add_argument("--out", default=str(REF_DIR / "column_map.json"))
    args = ap.parse_args()

    ddir = Path(args.data_dir)
    move = ddir / "DogMoveData.csv"
    info_csv = ddir / "DogInfo.csv"
    desc = ddir / "Data_description.txt"

    if not move.exists():
        die(f"""{move} not found. Download the dataset first:

    kaggle datasets download -d benjamingray44/inertial-data-for-dog-behaviour-classification
    unzip inertial-data-for-dog-behaviour-classification.zip -d ./data/

  (needs ~/.kaggle/kaggle.json — Account -> Settings -> Create New Token)""")

    header("Data_description.txt")
    if desc.exists():
        print(desc.read_text(encoding="utf-8", errors="replace"))
    else:
        warn(f"{desc} not found — read the dataset page before trusting anything below")

    result = profile(move, args.chunksize, args.sample)
    if info_csv.exists():
        result.update(profile_info(info_csv))
    else:
        warn(f"{info_csv} not found — REF.DOG_INFO will be empty and cohort baselines will be null")

    result["_meta"] = {
        "status": "GENERATED — verified against the file",
        "generated_by": "scripts/profile_dataset.py",
        "authoritative": True,
    }
    result["expected"] = {
        "row_count": result["observed"]["row_count"],
        "column_count": result["observed"]["column_count"],
        "distinct_dogs": result["observed"]["distinct_dogs"],
        "sample_rate_hz": result["observed"]["sample_rate_hz"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    header("Gate A output")
    ok(f"wrote {out}")
    print("""
  Review these before running any DDL:
    1. the column resolution table above — every ✓ is a resolver guess, confirm it
    2. the ethogram_map: every UNMAPPED label needs a decision
    3. the syndrome feasibility verdict (SHAKE / SCRATCH)

  Then:
    python scripts/run_sql.py --all
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
