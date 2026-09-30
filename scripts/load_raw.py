#!/usr/bin/env python3
"""
Load DogMoveData.csv into Snowflake. Once, properly, and never touch it again.

    python scripts/load_raw.py
    python scripts/load_raw.py --limit-rows 2000000    # smaller first pass
    python scripts/load_raw.py --skip-parquet          # shards already built

A 10.6 million row CSV is a fine Snowflake workload and a poor pandas workload,
so pandas only ever sees one chunk at a time:

    CSV --chunked read--> Parquet shards --PUT--> internal stage --COPY INTO--> table

Column names come from ref/column_map.json — the Gate A profile — and the CSV
header is verified against it before a single shard is written. If the file does
not have the columns the profile says it has, this aborts. That is the whole
point: a rename that goes unnoticed here becomes an hour of silent nulls in the
feature layer.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from _common import (  # noqa: E402
    CANONICAL_KEY_COLS,
    CANONICAL_LABEL_COLS,
    CANONICAL_SENSOR_COLS,
    DATA_DIR,
    connect,
    die,
    env,
    header,
    info,
    load_column_map,
    load_env,
    ok,
    q,
    warn,
)

BULK_TABLE = "RAW.COLLAR_TELEMETRY_BULK"
STAGE = "@TELLTAIL.RAW.DOG_STAGE"


def build_parquet_shards(csv_path: Path, out_dir: Path, colmap: dict,
                         rows_per_shard: int, limit_rows: int | None) -> list[Path]:
    columns = colmap["columns"]
    rename = {actual: canon for canon, actual in columns.items()}

    header("Verifying the CSV header against the Gate A profile")
    head = pd.read_csv(csv_path, nrows=1)
    actual_headers = set(head.columns)
    missing = {canon: real for canon, real in columns.items() if real not in actual_headers}
    if missing:
        die(
            "the CSV does not have the columns ref/column_map.json says it has.\n"
            f"    missing: {missing}\n"
            f"    present: {sorted(actual_headers)}\n"
            "  Re-run Gate A:  python scripts/profile_dataset.py"
        )
    ok(f"all {len(columns)} mapped columns present")

    extra = sorted(actual_headers - set(columns.values()))
    if extra:
        info(f"unmapped columns (kept in the shards, land in raw_payload): {extra}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header(f"Sharding to Parquet ({rows_per_shard:,} rows per shard)")
    shards: list[Path] = []
    total = 0
    t0 = time.perf_counter()

    reader = pd.read_csv(csv_path, chunksize=rows_per_shard, low_memory=False)
    for i, chunk in enumerate(reader):
        chunk = chunk.rename(columns=rename)
        # keep only the canonical columns the table declares, in table order
        want = CANONICAL_KEY_COLS + CANONICAL_SENSOR_COLS + CANONICAL_LABEL_COLS
        have = [c for c in want if c in chunk.columns]
        out = chunk[have].copy()
        out["_shard"] = f"shard_{i:04d}"

        # Snowflake maps Parquet columns case-insensitively by name; upper-casing
        # here removes any doubt about MATCH_BY_COLUMN_NAME.
        out.columns = [c.upper() for c in out.columns]

        p = out_dir / f"shard_{i:04d}.parquet"
        out.to_parquet(p, index=False, compression="snappy")
        shards.append(p)
        total += len(out)
        print(f"    {p.name}  {len(out):>9,} rows   ({total:,} total)", end="\r", flush=True)

        if limit_rows and total >= limit_rows:
            warn(f"stopping at --limit-rows {limit_rows:,}")
            break

    dt = time.perf_counter() - t0
    print(f"    {len(shards)} shards, {total:,} rows, {dt:.1f}s" + " " * 20)
    return shards


def load_dog_info(conn, csv_path: Path, colmap: dict) -> int:
    if not csv_path.exists():
        warn(f"{csv_path} not found — REF.DOG_INFO stays empty, cohort baselines "
             f"will be null and the holdout cannot be breed-stratified")
        return 0

    info_cols = colmap.get("info_columns") or {}
    if not info_cols:
        warn("ref/column_map.json has no info_columns; skipping REF.DOG_INFO")
        return 0

    df = pd.read_csv(csv_path)
    rename = {actual: canon for canon, actual in info_cols.items()}
    df = df.rename(columns=rename)

    # Age is stored in MONTHS in this dataset ("Age months"). Convert once, here,
    # rather than letting a 76-month Belgian Shepherd reach REF.V_DOG_COHORT and
    # land in the 'senior' band at six years of overstatement.
    if "age_months" in df.columns and "age_years" not in df.columns:
        df["age_years"] = pd.to_numeric(df["age_months"], errors="coerce") / 12.0
        info("age converted from months to years")

    # Gender is coded 1 = female, 2 = male (Data_description.txt). Mapped so the
    # Drivers tab reports a dimension called 'female', not a dimension called 2.
    if "sex" in df.columns:
        from profile_dataset import SEX_CODES
        df["sex"] = df["sex"].map(lambda v: SEX_CODES.get(v, SEX_CODES.get(str(v), None)))

    want = ["dog_id", "breed", "sex", "age_years", "weight_kg", "height_cm"]
    for c in want:
        if c not in df.columns:
            df[c] = None
    df = df[want]

    rows = [tuple(None if pd.isna(v) else v for v in r) for r in df.itertuples(index=False)]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM REF.DOG_INFO")
        cur.executemany(
            "INSERT INTO REF.DOG_INFO (dog_id, breed, sex, age_years, weight_kg, height_cm) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
    ok(f"REF.DOG_INFO: {len(rows)} dogs, {df['breed'].nunique()} breeds")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-load the telemetry corpus.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--rows-per-shard", type=int, default=500_000)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--skip-parquet", action="store_true")
    ap.add_argument("--allow-example-map", action="store_true",
                    help="use the unverified prior instead of the Gate A profile")
    args = ap.parse_args()

    load_env()
    ddir = Path(args.data_dir)
    csv_path = ddir / "DogMoveData.csv"
    info_path = ddir / "DogInfo.csv"
    shard_dir = ddir / "parquet"

    if not csv_path.exists():
        die(f"""{csv_path} not found.

    kaggle datasets download -d benjamingray44/inertial-data-for-dog-behaviour-classification
    unzip inertial-data-for-dog-behaviour-classification.zip -d ./data/""")

    colmap = load_column_map(allow_example=args.allow_example_map)
    if colmap.get("_meta", {}).get("status", "").startswith("PRIOR"):
        warn("loading against an UNVERIFIED column map. Gate A has not been run.")

    if not args.skip_parquet:
        build_parquet_shards(csv_path, shard_dir, colmap,
                             args.rows_per_shard, args.limit_rows)
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        die(f"no Parquet shards in {shard_dir}")

    conn = connect()
    try:
        db = env("SNOWFLAKE_DATABASE", "TELLTAIL")

        header("PUT shards to the internal stage")
        t0 = time.perf_counter()
        # PUT wants forward slashes and a file:// URL even on Windows.
        pattern = shard_dir.resolve().as_posix()
        with conn.cursor() as cur:
            cur.execute(f"REMOVE {STAGE}")
            cur.execute(
                f"PUT 'file://{pattern}/*.parquet' {STAGE} "
                f"AUTO_COMPRESS=TRUE PARALLEL=8 OVERWRITE=TRUE"
            )
            for r in cur.fetchall():
                print(f"    {r[0]:<28} {r[6]}")
        ok(f"staged {len(shards)} shards in {time.perf_counter() - t0:.1f}s")

        header("COPY INTO " + BULK_TABLE)
        t0 = time.perf_counter()
        rows = q(conn, f"""
            COPY INTO {db}.{BULK_TABLE}
                (dog_id, test_num, t_sec,
                 neck_ax, neck_ay, neck_az, neck_gx, neck_gy, neck_gz,
                 back_ax, back_ay, back_az, back_gx, back_gy, back_gz,
                 label_primary, label_secondary, label_tertiary, point_event, task,
                 _shard)
            FROM (
                SELECT
                    $1:DOG_ID::NUMBER,          $1:TEST_NUM::NUMBER,  $1:T_SEC::FLOAT,
                    $1:NECK_AX::FLOAT, $1:NECK_AY::FLOAT, $1:NECK_AZ::FLOAT,
                    $1:NECK_GX::FLOAT, $1:NECK_GY::FLOAT, $1:NECK_GZ::FLOAT,
                    $1:BACK_AX::FLOAT, $1:BACK_AY::FLOAT, $1:BACK_AZ::FLOAT,
                    $1:BACK_GX::FLOAT, $1:BACK_GY::FLOAT, $1:BACK_GZ::FLOAT,
                    $1:LABEL_PRIMARY::STRING,   $1:LABEL_SECONDARY::STRING,
                    $1:LABEL_TERTIARY::STRING,  $1:POINT_EVENT::STRING, $1:TASK::STRING,
                    $1:_SHARD::STRING
                FROM {STAGE}
            )
            FILE_FORMAT = (TYPE = PARQUET)
            ON_ERROR = ABORT_STATEMENT
            PURGE = FALSE
        """)
        loaded = sum(int(r.get("rows_loaded", 0) or 0) for r in rows) if rows else 0
        ok(f"COPY INTO finished in {time.perf_counter() - t0:.1f}s "
           f"({loaded:,} rows across {len(rows)} files)")

        load_dog_info(conn, info_path, colmap)

        # ---- verification: the profile said N, the warehouse must agree -----
        header("Verifying the load against the Gate A profile")
        prof = q(conn, f"SELECT * FROM {db}.RAW.V_BULK_PROFILE")[0]
        expect = colmap.get("expected") or {}

        print(f"  rows in warehouse : {int(prof['ROW_COUNT']):,}")
        print(f"  distinct dogs     : {int(prof['DISTINCT_DOGS'])}")
        print(f"  t_sec range       : {prof['T_MIN']} .. {prof['T_MAX']}")
        print(f"  distinct labels   : {int(prof['DISTINCT_LABELS'])}")
        print(f"  null labels       : {int(prof['NULL_LABELS']):,}")
        print(f"  null sensor rows  : {int(prof['NULL_SENSOR_ROWS']):,}")

        problems: list[str] = []
        if not args.limit_rows and expect.get("row_count"):
            if int(prof["ROW_COUNT"]) != int(expect["row_count"]):
                problems.append(
                    f"row count {int(prof['ROW_COUNT']):,} != profiled "
                    f"{int(expect['row_count']):,}"
                )
        if expect.get("distinct_dogs"):
            if int(prof["DISTINCT_DOGS"]) != int(expect["distinct_dogs"]):
                problems.append(
                    f"dog count {int(prof['DISTINCT_DOGS'])} != profiled "
                    f"{int(expect['distinct_dogs'])}"
                )
        if int(prof["NULL_SENSOR_ROWS"]) > 0:
            problems.append(f"{int(prof['NULL_SENSOR_ROWS']):,} rows have a null "
                            f"neck_ax or back_ax — a column mapping is probably wrong")

        if problems:
            for p in problems:
                warn(p)
            die("load verification failed. Do not build on top of this.")
        ok("load matches the profile")

        header("Label coverage")
        for r in q(conn, f"SELECT * FROM {db}.RAW.V_LABEL_COVERAGE ORDER BY n_rows DESC"):
            flag = "  <- UNMAPPED" if r["IS_UNMAPPED"] else f"  -> {r['MAPPED_STATE']}"
            print(f"  {str(r['RAW_LABEL']):<28} {int(r['N_ROWS']):>12,}  "
                  f"{r['PCT']:>6}%{flag}")

        print()
        ok("bulk load complete. Next:  python scripts/run_sql.py --all")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
