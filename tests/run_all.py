#!/usr/bin/env python3
"""
Every offline check, in one command. No Snowflake account, no credits, no network.

    python tests/run_all.py

  1. syndrome patterns      compiles the real PATTERN/DEFINE out of the SQL and
                            exercises it against fixtures
  2. demo signal            synthesises IMU signal, runs the real feature math
                            and state ladder, and matches the real patterns
  3. SiS compatibility      the Streamlit app must parse under Python 3.11
  4. chart layer            every chart helper builds a figure on a current
                            plotly AND on the old one SiS may solve to, plus
                            the Altair 4/5 selection shim
  5. SQL parse              every warehouse/*.sql splits into statements cleanly
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(label: str, args: list[str]) -> bool:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    r = subprocess.run(args, cwd=REPO)
    return r.returncode == 0


def check_sql() -> bool:
    """Every SQL file must render its template variables and split into
    statements. Catches an unbalanced $$ block or an unknown ${VAR} before
    run_sql.py hits it against a live warehouse mid-build."""
    print(f"\n{'=' * 74}\nSQL parse\n{'=' * 74}")
    sys.path.insert(0, str(REPO / "scripts"))
    import os

    os.environ.setdefault("CORTEX_MODEL", "claude-3-5-sonnet")
    os.environ.setdefault("TELLTAIL_HASH_SALT", "test-salt")
    os.environ.setdefault("CORTEX_MAX_ROWS_PER_BATCH", "25")
    os.environ.setdefault("SNOWFLAKE_WAREHOUSE", "TELLTAIL_WH")
    os.environ.setdefault("SNOWFLAKE_DATABASE", "TELLTAIL")

    from _common import first_line, render_sql, split_statements  # noqa: E402

    ok = True
    for f in sorted((REPO / "warehouse").glob("[0-9][0-9]_*.sql")):
        try:
            stmts = split_statements(render_sql(f.read_text(encoding="utf-8"),
                                                source=f.name))
        except SystemExit as exc:
            print(f"  FAIL {f.name}: {exc}")
            ok = False
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {f.name}: {type(exc).__name__}: {exc}")
            ok = False
            continue

        problems: list[str] = []

        if any("${" in s for s in stmts):
            problems.append("unresolved ${TEMPLATE} variable")

        # A statement that begins with CALL, END or BEGIN is a FRAGMENT of a
        # multi-statement `AS BEGIN ... END;` body that the splitter tore apart
        # on an inner semicolon. Snowflake accepts such bodies; no client that
        # splits on semicolons can. The failure is silent and expensive: the
        # task gets created with only its first statement and quietly does a
        # fraction of its job. Keep procedure bodies inside $$ ... $$ and give
        # each task exactly one CALL.
        for i, s in enumerate(stmts):
            head = first_line(s, 40).upper()
            if head.startswith(("CALL ", "END", "BEGIN")):
                problems.append(
                    f"statement {i} is a fragment of a BEGIN...END body: "
                    f"{first_line(s, 60)!r}"
                )

        if problems:
            print(f"  FAIL {f.name}")
            for p in problems:
                print(f"       {p}")
            ok = False
        else:
            print(f"  PASS {f.name:<28} {len(stmts):>3} statements")
    return ok


def main() -> int:
    results = [
        ("syndrome patterns", run("1 · syndrome patterns",
                                  [PY, "tests/test_syndrome_patterns.py"])),
        ("demo signal", run("2 · demo signal",
                            [PY, "tests/test_demo_signal.py"])),
        ("SiS compatibility", run("3 · Streamlit in Snowflake compatibility",
                                  [PY, "tests/sis_compat.py",
                                   "warehouse/streamlit_app.py"])),
        ("chart layer", run("4 · chart layer, native and degraded",
                            [PY, "tests/test_chart_layer.py"])),
        ("SQL parse", check_sql()),
    ]

    print(f"\n{'=' * 74}")
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [l for l, ok in results if not ok]
    print("=" * 74)
    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}\n")
        return 1
    print("\nAll offline checks pass.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
