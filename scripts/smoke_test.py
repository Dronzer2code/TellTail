#!/usr/bin/env python3
"""
The ninety-second smoke test. Run this at hour zero, before writing anything.

Checks, in order of how badly each one ruins the weekend:
  1. connection, region, account, timezone
  2. MATCH_RECOGNIZE on a five-row inline table          <- the whole submission
  3. SNOWFLAKE.CORTEX.COMPLETE                            <- the AI layer
  4. AI_CLASSIFY / AI_AGG availability
  5. ML function availability (CLASSIFICATION, FORECAST, ANOMALY, TOP_INSIGHTS)
  6. ASOF JOIN
  7. Dynamic Table creation privilege
  8. remaining trial credit

If MATCH_RECOGNIZE or Cortex fails on your account or region you need to know
at hour zero, not hour thirty.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import connect, env, fail, header, info, load_env, ok, q, warn  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, critical: bool = False):
    """Decorator-ish context: run a probe, record pass/fail, never abort early."""

    def run(fn):
        try:
            detail = fn() or ""
            ok(f"{name}  {detail}")
            RESULTS.append((name, True, str(detail)))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).split("\n")[0][:200]
            (fail if critical else warn)(f"{name}  —  {msg}")
            RESULTS.append((name, False, msg))
            if "--verbose" in sys.argv:
                traceback.print_exc()
        return fn

    return run


MATCH_RECOGNIZE_SMOKE = """
WITH t AS (
    SELECT * FROM VALUES
        (1,1,'REST'),(1,2,'SHAKE'),(1,3,'SCRATCH'),(1,4,'SCRATCH'),
        (1,5,'SCRATCH'),(1,6,'SHAKE'),(1,7,'SCRATCH'),(1,8,'SCRATCH')
    AS v(dog_id, i, state)
)
SELECT * FROM t
MATCH_RECOGNIZE (
    PARTITION BY dog_id ORDER BY i
    MEASURES MATCH_NUMBER() AS m,
             FIRST(onset.i)  AS start_i,
             LAST(itch.i)    AS end_i,
             COUNT(itch.*)   AS scratch_epochs
    ONE ROW PER MATCH
    AFTER MATCH SKIP PAST LAST ROW
    PATTERN ( onset shake itch{3,} shake itch{2,} )
    DEFINE onset AS state = 'REST',
           shake AS state = 'SHAKE',
           itch  AS state = 'SCRATCH'
)
"""


def main() -> int:
    load_env()
    header("TELLTAIL smoke test")

    try:
        conn = connect()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(f"cannot connect: {exc}")
        return 2

    @check("connection / region / timezone", critical=True)
    def _conn():
        r = q(conn, "SELECT CURRENT_REGION() rg, CURRENT_ACCOUNT() ac, "
                    "CURRENT_VERSION() v, CURRENT_TIMESTAMP() ts")[0]
        tz = q(conn, "SHOW PARAMETERS LIKE 'TIMEZONE' IN ACCOUNT")
        tzval = tz[0].get("value") if tz else "?"
        if tzval != "Etc/UTC":
            warn(f"    account TIMEZONE is {tzval!r}, not 'Etc/UTC'. "
                 f"Epoch boundaries will drift. Run: ALTER ACCOUNT SET TIMEZONE='Etc/UTC';")
        return f"{r['AC']} / {r['RG']} / sf {r['V']} / tz={tzval}"

    @check("MATCH_RECOGNIZE  (THE submission)", critical=True)
    def _mr():
        rows = q(conn, MATCH_RECOGNIZE_SMOKE)
        assert len(rows) == 1, f"expected exactly 1 match, got {len(rows)}"
        r = rows[0]
        assert int(r["M"]) == 1, r
        assert int(r["START_I"]) == 1, r
        assert int(r["END_I"]) == 8, r
        assert int(r["SCRATCH_EPOCHS"]) == 5, r
        return f"1 match, epochs 1→8, 5 scratch epochs — pattern semantics confirmed"

    @check("ASOF JOIN")
    def _asof():
        rows = q(conn, """
            WITH a AS (SELECT * FROM VALUES (1, 10::FLOAT) AS v(k, t)),
                 b AS (SELECT * FROM VALUES (1, 5::FLOAT), (1, 9::FLOAT) AS v(k, t))
            SELECT a.t AS at, b.t AS bt
            FROM a ASOF JOIN b MATCH_CONDITION (a.t >= b.t) ON a.k = b.k
        """)
        assert rows and float(rows[0]["BT"]) == 9.0, rows
        return "matched nearest preceding row"

    model = env("CORTEX_MODEL", "claude-3-5-sonnet")

    @check(f"SNOWFLAKE.CORTEX.COMPLETE ({model})", critical=True)
    def _cortex():
        r = q(conn, "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, 'Reply with exactly: OK') AS s", (model,))
        return repr((r[0]["S"] or "").strip()[:40])

    @check("AI_CLASSIFY")
    def _classify():
        r = q(conn, """
            SELECT AI_CLASSIFY('The dog is limping badly and will not put weight on it.',
                   ['routine monitoring','schedule appointment','urgent veterinary attention']) AS c
        """)
        return str(r[0]["C"])[:80]

    @check("AI_AGG")
    def _aiagg():
        r = q(conn, """
            WITH t AS (SELECT * FROM VALUES ('dog A is fine'),('dog B is limping') AS v(s))
            SELECT AI_AGG(s, 'Name the dog needing attention. Five words max.') AS b FROM t
        """)
        return str(r[0]["B"])[:80]

    @check("ML.CLASSIFICATION available")
    def _mlclass():
        q(conn, "SHOW SNOWFLAKE.ML.CLASSIFICATION")
        return "class registered in account"

    @check("ML.FORECAST available")
    def _mlfc():
        q(conn, "SHOW SNOWFLAKE.ML.FORECAST")
        return "class registered in account"

    @check("ML.ANOMALY_DETECTION available")
    def _mlad():
        q(conn, "SHOW SNOWFLAKE.ML.ANOMALY_DETECTION")
        return "class registered in account"

    @check("SNOWFLAKE.ML.TOP_INSIGHTS available")
    def _ti():
        # The Top Insights call signature has moved between preview and GA, so
        # this probe reports which shape (if either) this account accepts.
        # ML.SP_RUN_TOP_INSIGHTS falls back to a transparent SQL contribution
        # decomposition regardless, so a failure here costs presentation, not
        # substance.
        try:
            q(conn, """
                SELECT 1 FROM TABLE(SNOWFLAKE.ML.TOP_INSIGHTS(
                    SELECT {'d': 'a'} AS dimensions, 1.0 AS metric, TRUE AS label
                    FROM VALUES (1) AS v(x)
                )) LIMIT 1
            """)
            return "native OBJECT-dimensions signature accepted"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{str(exc).splitlines()[0][:120]} — the SQL contribution "
                f"fallback will be used, and ML.DRIVER_INSIGHTS.method will say so"
            ) from exc

    @check("CREATE DYNAMIC TABLE privilege")
    def _dt():
        # A dynamic table must have at least one BASE TABLE; `SELECT 1` is not a
        # valid definition, so the probe creates a real one to select from.
        db = env("SNOWFLAKE_DATABASE", "TELLTAIL")
        wh = env("SNOWFLAKE_WAREHOUSE", "TELLTAIL_WH")
        q(conn, f"CREATE SCHEMA IF NOT EXISTS {db}._SMOKE")
        try:
            q(conn, f"CREATE OR REPLACE TABLE {db}._SMOKE.BASE (x NUMBER)")
            q(conn, f"INSERT INTO {db}._SMOKE.BASE VALUES (1)")
            q(conn, f"""CREATE OR REPLACE DYNAMIC TABLE {db}._SMOKE.DT_PROBE
                        TARGET_LAG = '1 minute' WAREHOUSE = {wh}
                        AS SELECT x FROM {db}._SMOKE.BASE""")
            n = q(conn, f"SELECT COUNT(*) AS n FROM {db}._SMOKE.DT_PROBE")[0]["N"]
            return f"created, refreshed, {n} row(s) visible"
        finally:
            q(conn, f"DROP SCHEMA IF EXISTS {db}._SMOKE CASCADE")

    @check("trial credit remaining")
    def _credits():
        rows = q(conn, """
            SELECT ROUND(SUM(credits_used), 3) AS used
            FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
            WHERE start_time > DATEADD('day', -30, CURRENT_TIMESTAMP())
        """)
        used = rows[0]["USED"] or 0
        return f"{used} credits used in 30d (trial grant is 400)"

    # ---- summary -----------------------------------------------------------
    header("Summary")
    critical_failed = False
    for name, passed, detail in RESULTS:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        if not passed and name.startswith(("MATCH_RECOGNIZE", "connection", "SNOWFLAKE.CORTEX")):
            critical_failed = True

    print()
    if critical_failed:
        fail("A critical probe failed. Fix it now — hour zero, not hour thirty.")
        print("""
  MATCH_RECOGNIZE failing  -> wrong account edition. Nothing else matters; fix first.
  Cortex COMPLETE failing  -> wrong region. Run:
        ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
     or recreate the trial in AWS US West (Oregon).
""")
        return 1

    n_warn = sum(1 for _, p, _ in RESULTS if not p)
    if n_warn:
        warn(f"{n_warn} non-critical probe(s) failed — see the honest fallbacks in HONESTY.md")
        info("ML.CLASSIFICATION unavailable is survivable: warehouse/05 ships a "
             "transparent SQL rules ethogram behind PARAMS.use_rules_classifier.")
    else:
        ok("everything green. Build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
