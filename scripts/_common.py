"""
TELLTAIL shared plumbing: env, Snowflake connection, SQL script execution.

Every other script imports from here so there is exactly one place that knows
how to talk to the warehouse and exactly one SQL statement splitter.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = REPO / "warehouse"
REF_DIR = REPO / "ref"
DATA_DIR = REPO / "data"

# ---------------------------------------------------------------------------
# console
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

# Windows consoles default to cp1252, which cannot encode box-drawing or check
# glyphs — and Python raises UnicodeEncodeError rather than degrading, so a
# decorative character aborts the build. Try UTF-8 first; if the stream will not
# take it, fall back to an ASCII glyph set instead of crashing.
# line_buffering matters when stdout is redirected to a file or a pipe: Python
# switches to block buffering there, so a long-running profile or load shows
# nothing at all until it finishes or fills 8 KB. Progress you cannot see is
# indistinguishable from a hang.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[union-attr]
except Exception:  # noqa: BLE001 - older Python, or a stream that is not a TextIO
    pass


def _encodable(s: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        s.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE_OK = _encodable("·✓✗─")
G = {
    "dot":  "·" if _UNICODE_OK else "*",
    "ok":   "✓" if _UNICODE_OK else "+",
    "bad":  "✗" if _UNICODE_OK else "x",
    "bar":  "─" if _UNICODE_OK else "-",
    "ell":  "…" if _UNICODE_OK else "...",
}


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def info(msg: str) -> None:
    print(f"{_c('36', '  ' + G['dot'])} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{_c('32', '  ' + G['ok'])} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{_c('33', '  !')} {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"{_c('31', '  ' + G['bad'])} {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[valid-type]
    fail(msg)
    sys.exit(code)


def header(msg: str) -> None:
    bar = G["bar"]
    print()
    print(_c("1", f"{bar * 2} {msg} " + bar * max(0, 68 - len(msg))), flush=True)


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


def load_env() -> dict[str, str]:
    """Load .env if present. Never raises on a missing file: some scripts
    (the offline pattern validator, the profiler) need no credentials."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:  # pragma: no cover - dotenv is in requirements
        pass
    return dict(os.environ)


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.environ.get(name, default if default is not None else "")
    if required and not val:
        die(
            f"{name} is not set. Copy .env.example to .env and fill it in.\n"
            f"    cp .env.example .env"
        )
    return val


# ---------------------------------------------------------------------------
# column map (Gate A output) — the single source of truth for CSV column names
# ---------------------------------------------------------------------------

CANONICAL_SENSOR_COLS = [
    "neck_ax", "neck_ay", "neck_az", "neck_gx", "neck_gy", "neck_gz",
    "back_ax", "back_ay", "back_az", "back_gx", "back_gy", "back_gz",
]
CANONICAL_KEY_COLS = ["dog_id", "test_num", "t_sec"]
CANONICAL_LABEL_COLS = ["label_primary", "label_secondary", "label_tertiary", "point_event", "task"]


def load_column_map(*, allow_example: bool = False) -> dict[str, Any]:
    """Load the generated column map. Refuses the unverified prior unless the
    caller explicitly opts in, because a wrong column name here is an hour of
    silent nulls downstream."""
    generated = REF_DIR / "column_map.json"
    if generated.exists():
        cm = json.loads(generated.read_text(encoding="utf-8"))
        cm.setdefault("_meta", {})["loaded_from"] = str(generated)
        return cm
    if not allow_example:
        die(
            "ref/column_map.json not found. Gate A has not been run.\n"
            "    python scripts/profile_dataset.py\n"
            "  That reads the real CSV header and the real label vocabulary and\n"
            "  writes the map every other script depends on. Do not skip it."
        )
    example = REF_DIR / "column_map.example.json"
    warn("using ref/column_map.example.json — an UNVERIFIED prior, not a profile")
    cm = json.loads(example.read_text(encoding="utf-8"))
    cm.setdefault("_meta", {})["loaded_from"] = str(example)
    return cm


# ---------------------------------------------------------------------------
# Snowflake
# ---------------------------------------------------------------------------


def connect(role: str | None = None, *, autocommit: bool = True):
    """Open a Snowflake connection from .env."""
    import platform

    import snowflake.connector

    # connector >= 4.x calls platform.libc_ver() during login, which opens
    # sys.executable. Under Microsoft Store Python that path is an app execution
    # alias, open() raises EINVAL, and every login dies before reaching Snowflake.
    # libc is meaningless on Windows anyway, so answer for it.
    if sys.platform == "win32":
        platform.libc_ver = lambda *a, **k: ("", "")  # type: ignore[assignment]

    load_env()
    conn = snowflake.connector.connect(
        account=env("SNOWFLAKE_ACCOUNT", required=True),
        user=env("SNOWFLAKE_USER", required=True),
        password=env("SNOWFLAKE_PASSWORD", required=True),
        role=role or env("SNOWFLAKE_ROLE", "SYSADMIN"),
        warehouse=env("SNOWFLAKE_WAREHOUSE", "TELLTAIL_WH"),
        database=env("SNOWFLAKE_DATABASE", "TELLTAIL"),
        autocommit=autocommit,
        client_session_keep_alive=True,
        session_parameters={"TIMEZONE": "Etc/UTC"},
    )
    return conn


def q(conn, sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    """Run a query, return rows as dicts with plain Python scalars.

    Decimal -> float conversion happens HERE, once. Snowpark/connector return
    numerics as object-dtype Decimals; Plotly then treats them as categories and
    every line chart renders as an identical straight diagonal.
    """
    from decimal import Decimal

    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        rec: dict[str, Any] = {}
        for name, val in zip(cols, r):
            rec[name] = float(val) if isinstance(val, Decimal) else val
        out.append(rec)
    return out


def scalar(conn, sql: str, params: Iterable[Any] | None = None) -> Any:
    rows = q(conn, sql, params)
    if not rows:
        return None
    return next(iter(rows[0].values()))


# ---------------------------------------------------------------------------
# SQL script splitting
# ---------------------------------------------------------------------------

# Only these may appear as ${VAR} in a .sql file. An unknown one is an error,
# not a silent empty string.
SQL_TEMPLATE_VARS = {
    "CORTEX_MODEL": lambda: env("CORTEX_MODEL", "claude-3-5-sonnet"),
    "TELLTAIL_HASH_SALT": lambda: env("TELLTAIL_HASH_SALT", "telltail-dev-salt"),
    "CORTEX_MAX_ROWS_PER_BATCH": lambda: env("CORTEX_MAX_ROWS_PER_BATCH", "25"),
    "SNOWFLAKE_WAREHOUSE": lambda: env("SNOWFLAKE_WAREHOUSE", "TELLTAIL_WH"),
    "SNOWFLAKE_DATABASE": lambda: env("SNOWFLAKE_DATABASE", "TELLTAIL"),
    "SNOWFLAKE_ROLE": lambda: env("SNOWFLAKE_ROLE", "SYSADMIN"),
}

_TEMPLATE_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def render_sql(text: str, *, source: str = "<sql>") -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in SQL_TEMPLATE_VARS:
            die(f"{source}: unknown template variable ${{{name}}}. "
                f"Known: {', '.join(sorted(SQL_TEMPLATE_VARS))}")
        return SQL_TEMPLATE_VARS[name]()

    return _TEMPLATE_RE.sub(sub, text)


def split_statements(sql: str) -> list[str]:
    """Split a script into statements on top-level semicolons.

    Handles, because Snowflake scripts contain all of them:
      -- line comments          /* block comments, nested */
      'single quoted' with ''   "quoted identifiers"
      $$ dollar quoted bodies $$   (stored procedures, UDFs)

    A naive text.split(';') mangles every stored procedure in warehouse/07 and
    warehouse/09, which is why this exists.
    """
    out: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    depth_block_comment = 0

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # inside a block comment: only look for the terminator (Snowflake nests)
        if depth_block_comment:
            if ch == "/" and nxt == "*":
                depth_block_comment += 1
                buf.append("/*")
                i += 2
                continue
            if ch == "*" and nxt == "/":
                depth_block_comment -= 1
                buf.append("*/")
                i += 2
                continue
            buf.append(ch)
            i += 1
            continue

        # comment starts
        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            j = n if j == -1 else j
            buf.append(sql[i:j])
            i = j
            continue
        if ch == "/" and nxt == "*":
            depth_block_comment = 1
            buf.append("/*")
            i += 2
            continue

        # dollar-quoted body
        if ch == "$" and nxt == "$":
            j = sql.find("$$", i + 2)
            if j == -1:
                die("unterminated $$ ... $$ block in SQL script")
            buf.append(sql[i : j + 2])
            i = j + 2
            continue

        # single-quoted literal ('' is an escaped quote)
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "\\":
                    j += 2
                    continue
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            buf.append(sql[i : j + 1])
            i = j + 1
            continue

        # quoted identifier
        if ch == '"':
            j = sql.find('"', i + 1)
            j = n - 1 if j == -1 else j
            buf.append(sql[i : j + 1])
            i = j + 1
            continue

        # statement terminator
        if ch == ";":
            stmt = "".join(buf).strip()
            if _is_executable(stmt):
                out.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if _is_executable(tail):
        out.append(tail)
    return out


def _is_executable(stmt: str) -> bool:
    """True if the statement has anything besides comments and whitespace."""
    stripped = re.sub(r"/\*.*?\*/", " ", stmt, flags=re.S)
    stripped = re.sub(r"--[^\n]*", " ", stripped)
    return bool(stripped.strip())


def first_line(stmt: str, width: int = 78) -> str:
    """A readable one-line label for progress output."""
    body = re.sub(r"--[^\n]*", " ", stmt)
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    body = " ".join(body.split())
    return body[:width] + (G["ell"] if len(body) > width else "")


def execute_script(
    conn,
    path: Path,
    *,
    stop_on_error: bool = True,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Execute one .sql file statement by statement. Returns (ok, failed)."""
    raw = path.read_text(encoding="utf-8")
    text = render_sql(raw, source=path.name)
    stmts = split_statements(text)

    header(f"{path.name}  ({len(stmts)} statements)")
    n_ok = n_fail = 0

    for idx, stmt in enumerate(stmts, 1):
        label = first_line(stmt)
        if dry_run:
            print(f"  {idx:>3}. [dry] {label}")
            n_ok += 1
            continue

        t0 = time.perf_counter()
        try:
            with conn.cursor() as cur:
                cur.execute(stmt)
            dt = time.perf_counter() - t0
            print(f"  {_c('32', G['ok'])} {idx:>3}. {label}  {_c('90', f'{dt:6.2f}s')}")
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - we want the raw driver message
            dt = time.perf_counter() - t0
            print(f"  {_c('31', G['bad'])} {idx:>3}. {label}  {_c('90', f'{dt:6.2f}s')}")
            fail(f"      {type(exc).__name__}: {exc}")
            n_fail += 1
            if stop_on_error:
                raise
    return n_ok, n_fail


def iter_sql_files(only: str | None = None) -> Iterator[Path]:
    """warehouse/NN_*.sql in numeric order. `only` matches a prefix like '07'."""
    files = sorted(WAREHOUSE_DIR.glob("[0-9][0-9]_*.sql"))
    for f in files:
        if only and not f.name.startswith(only):
            continue
        yield f
