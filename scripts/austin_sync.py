#!/usr/bin/env python3
"""
Pull Austin Animal Center intakes and outcomes into REF, filtered to dogs.

    python scripts/austin_sync.py
    python scripts/austin_sync.py --limit 100000 --since 2015-01-01

Why a host-side script and not a Snowflake external function: Streamlit in
Snowflake has NO public internet egress, and external access integrations are
not supported on trial accounts. So the host writes into REF and the app only
ever reads a table. Same shape as the football sync last time.

Why this data at all: it answers a different objection than "is the telemetry
real". It answers "does any of this matter". Behaviour is a named outcome reason
in the Austin records, sitting alongside aggression and medical, and the
behavioural categories this pipeline detects on a collar at home are the same
categories that get recorded at intake after the relationship has already broken
down.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from _common import (  # noqa: E402
    connect, env, header, info, load_env, ok, q, split_statements, warn,
)

INTAKES_URL = "https://data.austintexas.gov/resource/wter-evkm.json"
OUTCOMES_URL = "https://data.austintexas.gov/resource/9t4d-g238.json"
PAGE = 5000

INTAKE_COLS = [
    "animal_id", "name", "datetime", "intake_type", "intake_condition",
    "animal_type", "sex_upon_intake", "age_upon_intake", "breed", "color",
    "found_location",
]
OUTCOME_COLS = [
    "animal_id", "name", "datetime", "date_of_birth", "outcome_type",
    "outcome_subtype", "animal_type", "sex_upon_outcome", "age_upon_outcome",
    "breed", "color",
]


def fetch(url: str, limit: int, since: str | None, token: str | None) -> list[dict]:
    headers = {"X-App-Token": token} if token else {}
    where = "animal_type='Dog'"
    if since:
        where += f" AND datetime >= '{since}T00:00:00'"

    out: list[dict] = []
    offset = 0
    while len(out) < limit:
        page = min(PAGE, limit - len(out))
        params = {"$limit": page, "$offset": offset, "$where": where,
                  "$order": "datetime DESC"}
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=60)
                if r.status_code == 429:
                    wait = 2 ** attempt
                    warn(f"rate limited; sleeping {wait}s "
                         f"(set SOCRATA_APP_TOKEN to raise the limit)")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                batch = r.json()
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    raise
                warn(f"{type(exc).__name__}; retrying in {2 ** attempt}s")
                time.sleep(2 ** attempt)
        else:
            break

        if not batch:
            break
        out.extend(batch)
        offset += len(batch)
        print(f"    {len(out):,} records", end="\r", flush=True)
        if len(batch) < page:
            break
    print(" " * 40, end="\r")
    return out


def push(conn, table: str, cols: list[str], records: list[dict]) -> int:
    """Replace the table's contents with these records.

    raw_payload is VARIANT and the connector cannot bind a VARIANT directly, so
    the JSON is bound as text and PARSE_JSON'd in the statement. Chunked into
    multi-row VALUES rather than executemany over an INSERT...SELECT, because
    the latter degrades to one round trip per row and 50,000 of those is a
    coffee break.
    """
    if not records:
        return 0

    rows = [tuple([rec.get(c) for c in cols] + [json.dumps(rec)]) for rec in records]
    collist = ", ".join(cols) + ", raw_payload"
    # column1..columnN from VALUES; the last one becomes PARSE_JSON'd.
    select_expr = ", ".join(f"column{i + 1}" for i in range(len(cols)))
    select_expr += f", PARSE_JSON(column{len(cols) + 1})"
    row_ph = "(" + ", ".join(["%s"] * (len(cols) + 1)) + ")"

    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table}")
        CH = 500
        n = 0
        for i in range(0, len(rows), CH):
            chunk = rows[i : i + CH]
            values = ", ".join([row_ph] * len(chunk))
            flat = [v for r in chunk for v in r]
            cur.execute(
                f"INSERT INTO {table} ({collist}) "
                f"SELECT {select_expr} FROM VALUES {values}",
                flat,
            )
            n += len(chunk)
            print(f"    pushed {n:,} / {len(rows):,}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Socrata -> REF.AAC_*")
    ap.add_argument("--limit", type=int, default=50_000)
    ap.add_argument("--since", default=None, help="ISO date, e.g. 2015-01-01")
    ap.add_argument("--skip-intakes", action="store_true")
    ap.add_argument("--skip-outcomes", action="store_true")
    args = ap.parse_args()

    load_env()
    token = env("SOCRATA_APP_TOKEN") or None
    if not token:
        info("no SOCRATA_APP_TOKEN; using the anonymous rate limit, which is fine "
             "for a one-shot sync")

    conn = connect()
    try:
        if not args.skip_intakes:
            header("Austin Animal Center — intakes (dogs)")
            recs = fetch(INTAKES_URL, args.limit, args.since, token)
            n = push(conn, "REF.AAC_INTAKES", INTAKE_COLS, recs)
            ok(f"REF.AAC_INTAKES: {n:,} records")

        if not args.skip_outcomes:
            header("Austin Animal Center — outcomes (dogs)")
            recs = fetch(OUTCOMES_URL, args.limit, args.since, token)
            n = push(conn, "REF.AAC_OUTCOMES", OUTCOME_COLS, recs)
            ok(f"REF.AAC_OUTCOMES: {n:,} records")

        header("Building the shelter-reality views")
        # One statement per execute(): the connector rejects a multi-statement
        # string unless num_statements is set, and we already own a splitter that
        # understands $$ blocks and embedded semicolons.
        #
        # V_SHELTER_PUNCHLINE joins MARTS.SYNDROME_MATCHES, which only exists
        # once warehouse/07 has run. This script is legitimately runnable before
        # that — the shelter tables are independent of the telemetry — so a
        # forward dependency is a warning, not a failure. Re-run after the
        # pipeline build and it resolves.
        built, deferred = 0, []
        with conn.cursor() as cur:
            for stmt in split_statements(SHELTER_VIEWS):
                name = "?"
                m = re.search(r"CREATE OR REPLACE VIEW\s+(\S+)", stmt, re.I)
                if m:
                    name = m.group(1)
                try:
                    cur.execute(stmt)
                    built += 1
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).splitlines()[-1][:120]
                    if "does not exist" in msg:
                        deferred.append((name, msg))
                    else:
                        raise
        ok(f"REF.V_AAC_* built ({built} views)")
        for name, msg in deferred:
            warn(f"deferred {name}: {msg}")
            info("       re-run scripts/austin_sync.py after run_sql.py --all")

        header("Sanity check")
        for sql, label in [
            ("SELECT COUNT(*) AS n FROM REF.AAC_INTAKES", "intake records"),
            ("SELECT COUNT(*) AS n FROM REF.AAC_OUTCOMES", "outcome records"),
            # SUM(n), not COUNT(*): the view is a GROUPed summary, so COUNT(*)
            # reports how many (outcome_type, subtype) pairs exist — it read
            # "2" against 50,000 records and looked like a broken join.
            ("SELECT COALESCE(SUM(n), 0) AS n FROM REF.V_AAC_BEHAVIOUR_OUTCOMES",
             "behaviour-linked outcomes"),
            ("""SELECT COUNT(*) AS n FROM REF.AAC_INTAKES i
                JOIN REF.AAC_BEHAVIOUR_MAP m
                  ON m.field = 'intake_condition' AND m.value = i.intake_condition
                WHERE m.maps_to_syndrome IS NOT NULL""",
             "syndrome-linked intakes"),
        ]:
            try:
                print(f"  {label:<28} {int(q(conn, sql)[0]['N']):>10,}")
            except Exception as exc:  # noqa: BLE001
                warn(f"{label}: {exc}")

        rows = q(conn, """
            SELECT outcome_type, COUNT(*) AS n,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM REF.AAC_OUTCOMES GROUP BY 1 ORDER BY 2 DESC LIMIT 8
        """)
        if rows:
            print()
            print("  outcome distribution (Austin is a no-kill shelter; over 90% of")
            print("  animals are adopted, transferred or returned, which is what makes")
            print("  the behavioural tail pointed rather than routine):")
            for r in rows:
                print(f"    {str(r['OUTCOME_TYPE']):<24} {int(r['N']):>8,}  {r['PCT']:>6}%")
        return 0
    finally:
        conn.close()


# The four things the Shelter Reality tab shows. Built here rather than in
# warehouse/*.sql because they depend on tables this script populates, and a
# build that runs before the sync would create views over empty tables.
SHELTER_VIEWS = """
CREATE OR REPLACE VIEW REF.V_AAC_INTAKE_TREND AS
SELECT
    DATE_TRUNC('month', datetime)              AS month,
    intake_type,
    intake_condition,
    COALESCE(m.is_behaviour, FALSE)            AS is_behaviour_linked,
    COUNT(*)                                   AS n
FROM REF.AAC_INTAKES i
LEFT JOIN REF.AAC_BEHAVIOUR_MAP m
       ON m.field = 'intake_condition' AND m.value = i.intake_condition
WHERE datetime IS NOT NULL
GROUP BY 1, 2, 3, 4;

CREATE OR REPLACE VIEW REF.V_AAC_BEHAVIOUR_OUTCOMES AS
SELECT
    o.outcome_type,
    o.outcome_subtype,
    m.maps_to_syndrome,
    COUNT(*)                                   AS n
FROM REF.AAC_OUTCOMES o
JOIN REF.AAC_BEHAVIOUR_MAP m
     ON m.field = 'outcome_subtype'
    AND m.value = o.outcome_subtype
    AND m.is_behaviour
GROUP BY 1, 2, 3;

-- Median length of stay: intake to outcome, per animal, by breed group and
-- intake condition. The cost of a behavioural label, visible in days.
CREATE OR REPLACE VIEW REF.V_AAC_LENGTH_OF_STAY AS
WITH paired AS (
    SELECT
        i.animal_id,
        i.breed,
        i.intake_condition,
        i.intake_type,
        i.datetime                                             AS intake_ts,
        MIN(o.datetime)                                        AS outcome_ts
    FROM REF.AAC_INTAKES i
    JOIN REF.AAC_OUTCOMES o
          ON o.animal_id = i.animal_id
         AND o.datetime >= i.datetime
    GROUP BY 1, 2, 3, 4, 5
)
SELECT
    CASE
        WHEN LOWER(breed) LIKE '%pit bull%'   THEN 'pit bull type'
        WHEN LOWER(breed) LIKE '%chihuahua%'  THEN 'chihuahua type'
        WHEN LOWER(breed) LIKE '%labrador%'   THEN 'labrador type'
        WHEN LOWER(breed) LIKE '%german shepherd%' THEN 'shepherd type'
        WHEN LOWER(breed) LIKE '%terrier%'    THEN 'terrier type'
        WHEN LOWER(breed) LIKE '%mix%'        THEN 'other mix'
        ELSE 'other'
    END                                                        AS breed_group,
    intake_condition,
    COUNT(*)                                                   AS n,
    ROUND(MEDIAN(DATEDIFF('day', intake_ts, outcome_ts)), 1)   AS median_los_days,
    ROUND(AVG(DATEDIFF('day', intake_ts, outcome_ts)), 1)      AS mean_los_days
FROM paired
WHERE outcome_ts IS NOT NULL
  AND DATEDIFF('day', intake_ts, outcome_ts) BETWEEN 0 AND 730
GROUP BY 1, 2;

-- THE PUNCHLINE. The six syndrome categories this pipeline detects on a collar,
-- at home, set beside the behaviour-linked outcome counts the shelter records
-- after the relationship has already broken down.
CREATE OR REPLACE VIEW REF.V_SHELTER_PUNCHLINE AS
SELECT
    c.syndrome_code,
    c.syndrome_name,
    c.body_system,
    COALESCE(tt.matches, 0)                    AS telltail_detections,
    COALESCE(tt.dogs, 0)                       AS telltail_dogs,
    COALESCE(sh.shelter_records, 0)            AS shelter_behaviour_records,
    m.note                                     AS mapping_note
FROM REF.SYNDROME_CATALOGUE c
LEFT JOIN (
    SELECT syndrome_code, COUNT(*) AS matches, COUNT(DISTINCT dog_id) AS dogs
    FROM MARTS.SYNDROME_MATCHES GROUP BY 1
) tt ON tt.syndrome_code = c.syndrome_code
LEFT JOIN (
    -- Both ends of the shelter record, not just the outcome.
    --
    -- Counting only V_AAC_BEHAVIOUR_OUTCOMES left four of six syndromes
    -- showing zero shelter records, because that view covers outcome_subtype
    -- only and the sole behavioural subtypes Austin records are 'Behavior' and
    -- 'Aggressive' — both mapped to S5. The intake side of REF.AAC_BEHAVIOUR_MAP
    -- was already carrying Injured -> S2, Sick -> S6 and Aged -> S4 and was
    -- simply never read, so the comparison silently understated itself by
    -- thousands of animals.
    SELECT maps_to_syndrome, SUM(n) AS shelter_records
    FROM (
        SELECT m.maps_to_syndrome, COUNT(*) AS n
        FROM REF.AAC_INTAKES i
        JOIN REF.AAC_BEHAVIOUR_MAP m
             ON m.field = 'intake_condition' AND m.value = i.intake_condition
        WHERE m.maps_to_syndrome IS NOT NULL
        GROUP BY 1
        UNION ALL
        SELECT maps_to_syndrome, n
        FROM REF.V_AAC_BEHAVIOUR_OUTCOMES
        WHERE maps_to_syndrome IS NOT NULL
    )
    GROUP BY 1
) sh ON sh.maps_to_syndrome = c.syndrome_code
LEFT JOIN (
    SELECT maps_to_syndrome, ANY_VALUE(note) AS note
    FROM REF.AAC_BEHAVIOUR_MAP WHERE maps_to_syndrome IS NOT NULL GROUP BY 1
) m ON m.maps_to_syndrome = c.syndrome_code;
"""


if __name__ == "__main__":
    raise SystemExit(main())
