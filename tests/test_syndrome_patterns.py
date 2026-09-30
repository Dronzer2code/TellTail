#!/usr/bin/env python3
"""
Gate E, run on a laptop.

These tests read the ACTUAL pattern text out of warehouse/07_syndromes.sql and
warehouse/02_ref_seed.sql — not copies of it — and exercise it against fixtures
through tests/match_recognize_sim.py. They answer, before a single credit is
spent:

  1. does every pattern compile, and does every pattern variable have a DEFINE?
  2. does each syndrome fire on the sequence it was designed for?
  3. does each syndrome stay silent on the near-miss that should NOT fire?
  4. is the strictness ladder real — does 'strict' actually reject what 'tuned'
     accepts, on a sequence engineered to sit between them?
  5. do the six hand-written views and REF.SYNDROME_CATALOGUE agree, so the SQL
     printed on the Syndromes tab is the SQL that ran?
  6. do the patterns stay quiet on a long stream of unrelated behaviour?

Run:
    python tests/test_syndrome_patterns.py        # standalone, no pytest needed
    pytest tests/ -q                              # also works
"""
from __future__ import annotations

import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_recognize_sim import (  # noqa: E402
    PatternError,
    compile_pattern,
    match_all,
    match_partition,
    parse_define,
)

REPO = Path(__file__).resolve().parent.parent
SQL_SYNDROMES = REPO / "warehouse" / "07_syndromes.sql"
SQL_REF_SEED = REPO / "warehouse" / "02_ref_seed.sql"

T0 = datetime(2026, 8, 16, 14, 0, 0)


# ---------------------------------------------------------------------------
# extract the real SQL, so the tests cannot drift from the warehouse
# ---------------------------------------------------------------------------

def extract_views() -> dict[str, dict[str, str]]:
    """PATTERN and DEFINE out of the six hand-written views in 07_syndromes.sql."""
    text = SQL_SYNDROMES.read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(
        r"CREATE OR REPLACE VIEW MARTS\.V_SYNDROME_(S\d)\b(.*?)(?=CREATE OR REPLACE|\Z)",
        text,
        re.S,
    ):
        code, body = m.group(1), m.group(2)
        pat = re.search(r"PATTERN\s*\(([^)]*)\)", body)
        dfn = re.search(r"^\s*DEFINE\s*\n(.*?)^\s*\)\s*$", body, re.S | re.M)
        if not (pat and dfn):
            continue
        out[code] = {
            "pattern_text": " ".join(pat.group(1).split()),
            "define_text": " ".join(dfn.group(1).split()),
        }
    return out


def strip_line_comments(sql: str) -> str:
    """Remove `-- ...` comments, respecting single-quoted string literals.

    The catalogue extractor below matches a VALUES tuple as one run of literals
    separated by whitespace only, which cannot span a comment line. Commenting a pattern
    where the pattern lives — the only place the comment is any use — therefore
    made the row invisible to this suite, and three syndromes silently vanished
    from every assertion here rather than failing one. Quote-aware because the
    clinical prose in the catalogue contains apostrophes and dashes.
    """
    out, i, n, in_str = [], 0, len(sql), False
    while i < n:
        c = sql[i]
        if in_str:
            out.append(c)
            if c == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    out.append(sql[i + 1]); i += 2; continue
                in_str = False
            i += 1
        elif c == "'":
            in_str = True; out.append(c); i += 1
        elif c == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find(chr(10), i)
            if j == -1:
                break
            i = j                      # keep the newline itself
        else:
            out.append(c); i += 1
    return "".join(out)


def extract_catalogue() -> dict[str, dict[str, str]]:
    """pattern_text / define_text out of the REF.SYNDROME_CATALOGUE insert."""
    text = strip_line_comments(SQL_REF_SEED.read_text(encoding="utf-8"))
    block = re.search(
        r"INSERT INTO REF\.SYNDROME_CATALOGUE(.*?)AS v\(column1", text, re.S
    )
    assert block, "could not locate the REF.SYNDROME_CATALOGUE insert"
    out: dict[str, dict[str, str]] = {}
    # each tuple starts with  ( \n 'Sn', 'name', 'system', '[json]', 'pattern', 'define',
    for m in re.finditer(
        r"\(\s*'(S\d)',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*"
        r"'(\[[^\]]*\])',\s*'((?:[^']|'')*)',\s*'((?:[^']|'')*)',\s*(\d+),\s*(\d+)",
        block.group(1),
        re.S,
    ):
        out[m.group(1)] = {
            "syndrome_name": m.group(2).replace("''", "'"),
            "symbols_json": m.group(4),
            "pattern_text": " ".join(m.group(5).split()),
            "define_text": " ".join(m.group(6).split()),
            "min_epochs": m.group(7),
            "default_severity": m.group(8),
        }
    return out


def extract_variants() -> dict[tuple[str, str], dict[str, str]]:
    """(code, variant) -> pattern text, out of REF.SYNDROME_VARIANTS."""
    text = strip_line_comments(SQL_REF_SEED.read_text(encoding="utf-8"))
    block = re.search(
        r"INSERT INTO REF\.SYNDROME_VARIANTS(.*?)AS v\(syndrome_code", text, re.S
    )
    assert block, "could not locate the REF.SYNDROME_VARIANTS insert"
    out: dict[tuple[str, str], dict[str, str]] = {}
    for m in re.finditer(
        r"\('(S\d)','(loose|tuned|strict)',\s*'((?:[^']|'')*)',\s*(\d+)",
        block.group(1),
    ):
        out[(m.group(1), m.group(2))] = {
            "pattern_text": " ".join(m.group(3).split()),
            "min_epochs": m.group(4),
        }
    return out


VIEWS = extract_views()
CATALOGUE = extract_catalogue()
VARIANTS = extract_variants()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def extract_activity_class() -> dict[str, str]:
    """state -> activity_class, out of the REF.ETHOGRAM seed."""
    text = strip_line_comments(SQL_REF_SEED.read_text(encoding="utf-8"))
    block = re.search(r"INSERT INTO REF\.ETHOGRAM(.*?)AS v\(state,", text, re.S)
    assert block, "could not locate the REF.ETHOGRAM insert"
    out = {}
    for m in re.finditer(
        r"\('([A-Z_]+)',\s*'[^']*',\s*'[^']*',\s*'[^']*',\s*"
        r"(?:TRUE|FALSE),\s*(?:TRUE|FALSE),\s*'([A-Z_]+)'",
        block.group(1),
    ):
        out[m.group(1)] = m.group(2)
    assert out, "no ethogram rows parsed"
    return out


ACTIVITY_CLASS = extract_activity_class()


def seq(*runs: tuple[str, int], start: datetime = T0, dog_id: int = 7, test_num: int = 1):
    """seq(('REST',1), ('SHAKE',1), ('SCRATCH',3)) -> one row per second."""
    rows = []
    ts = start
    for state, n in runs:
        for _ in range(n):
            rows.append({
                "dog_id": dog_id, "test_num": test_num, "epoch_ts": ts,
                "state": state,
                # S3 matches on activity_class rather than on state; every
                # fixture row therefore carries both, mirroring
                # MARTS.V_SYNDROME_INPUT. Derived from the same table the
                # warehouse derives it from, so a regrouping there fails here
                # instead of quietly diverging.
                "activity_class": ACTIVITY_CLASS.get(state, "OTHER"),
            })
            ts += timedelta(seconds=1)
    return rows


R = lambda n=1: ("REST", n)          # noqa: E731
SH = lambda n=1: ("SHAKE", n)        # noqa: E731
SC = lambda n=1: ("SCRATCH", n)      # noqa: E731
W = lambda n=1: ("WALK", n)          # noqa: E731
P = lambda n=1: ("PAUSE", n)         # noqa: E731
TR = lambda n=1: ("TROT", n)         # noqa: E731
ST = lambda n=1: ("STAND", n)        # noqa: E731
SL = lambda n=1: ("SLOW_TRANSITION", n)  # noqa: E731
PA = lambda n=1: ("PACE", n)         # noqa: E731
SN = lambda n=1: ("SNIFF", n)        # noqa: E731
CI = lambda n=1: ("CIRCLE", n)       # noqa: E731

# For each syndrome: a sequence that MUST match, and near-misses that must not.
FIXTURES: dict[str, dict] = {
    "S1": {
        # the spec's own smoke test, verbatim
        "positive": seq(R(1), SH(1), SC(3), SH(1), SC(2)),
        "expect_epochs": 8,
        "negatives": {
            "first bout one epoch short": seq(R(1), SH(1), SC(2), SH(1), SC(2)),
            "second bout one epoch short": seq(R(1), SH(1), SC(3), SH(1), SC(1)),
            "no head shake at all": seq(R(1), SC(12)),
            # onset is any resting POSTURE (REST/SIT/STAND); locomotion is not
            # one, so a bout that begins mid-walk still must not match.
            "does not emerge from a resting posture":
                seq(W(1), SH(1), SC(3), SH(1), SC(2)),
            "shake and scratch never alternate": seq(R(1), SH(2), SC(9)),
        },
        # engineered to satisfy tuned {3,}/{2,} but not strict {5,}/{4,}
        "between_tuned_and_strict": seq(R(1), SH(1), SC(4), SH(1), SC(3)),
    },
    "S2": {
        "positive": seq(W(4), P(1), W(2), P(1), W(2), P(1)),
        "expect_epochs": 11,
        "negatives": {
            "opening stride run too short": seq(W(2), P(1), W(2), P(1), W(2), P(1)),
            "only two interruptions": seq(W(4), P(1), W(2), P(1)),
            "uninterrupted gait": seq(W(20)),
            "middle stride run too long": seq(W(4), P(1), W(6), P(1), W(2), P(1)),
        },
        "between_tuned_and_strict": seq(W(4), P(1), W(3), P(1), W(3), P(1)),
    },
    "S3": {
        "positive": seq(TR(3), R(6), TR(2), R(9)),
        "expect_epochs": 20,
        "negatives": {
            "first recovery too short": seq(TR(3), R(3), TR(2), R(9)),
            "second recovery too short": seq(TR(3), R(6), TR(2), R(5)),
            "second burst not shortened": seq(TR(3), R(6), TR(4), R(9)),
            "steady trotting, no collapse": seq(TR(30)),
        },
        "between_tuned_and_strict": seq(TR(3), R(6), TR(2), R(9)),
    },
    "S4": {
        "positive": seq(R(12), SL(1), ST(1), R(12)),
        "expect_epochs": 26,
        "negatives": {
            "leading rest too short": seq(R(6), SL(1), ST(1), R(12)),
            "trailing rest too short": seq(R(12), SL(1), ST(1), R(6)),
            "sprang up, no slow transition": seq(R(12), ST(1), R(12)),
            "stayed up after rising": seq(R(12), SL(1), ST(20)),
        },
        "between_tuned_and_strict": seq(R(12), SL(1), ST(1), R(12)),
    },
    "S5": {
        "positive": seq(ST(1), PA(5), ST(1), PA(5)),
        "expect_epochs": 12,
        "negatives": {
            "first pacing run too short": seq(ST(1), PA(3), ST(1), PA(5)),
            "second pacing run too short": seq(ST(1), PA(5), ST(1), PA(2)),
            "pacing without the door check": seq(ST(1), PA(20)),
            "only one cycle": seq(ST(1), PA(6)),
        },
        "between_tuned_and_strict": seq(ST(1), PA(5), ST(1), PA(5)),
    },
    "S6": {
        "positive": seq(SN(6), CI(3), SN(6)),
        "expect_epochs": 15,
        "negatives": {
            "first casting run too short": seq(SN(3), CI(3), SN(6)),
            "second casting run too short": seq(SN(6), CI(3), SN(3)),
            "single turn, resolved": seq(SN(6), CI(1), SN(6)),
            "normal sniffing, no circling": seq(SN(30)),
        },
        "between_tuned_and_strict": seq(SN(6), CI(2), SN(6)),
    },
}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_all_six_syndromes_present():
    assert set(VIEWS) == {"S1", "S2", "S3", "S4", "S5", "S6"}, sorted(VIEWS)
    assert set(CATALOGUE) == {"S1", "S2", "S3", "S4", "S5", "S6"}, sorted(CATALOGUE)


def test_views_and_catalogue_agree():
    """The pattern the Syndromes tab prints must be the pattern that ran.

    The six views hold the literal SQL; REF.SYNDROME_CATALOGUE drives the
    sensitivity sweep, the ALL ROWS PER MATCH build and the on-screen code
    block. If they drift, the dashboard shows SQL that did not produce the
    result beside it, which is the one dishonesty this project cannot afford.
    """
    for code, view in sorted(VIEWS.items()):
        cat = CATALOGUE[code]
        assert view["pattern_text"] == cat["pattern_text"], (
            f"{code} PATTERN drift:\n  view      {view['pattern_text']}\n"
            f"  catalogue {cat['pattern_text']}"
        )
        v_def = parse_define(view["define_text"])
        c_def = parse_define(cat["define_text"])
        assert v_def == c_def, f"{code} DEFINE drift:\n  view {v_def}\n  cat  {c_def}"


def test_tuned_variant_matches_catalogue():
    for code in sorted(CATALOGUE):
        assert VARIANTS[(code, "tuned")]["pattern_text"] == CATALOGUE[code]["pattern_text"], (
            f"{code}: the 'tuned' variant must be the catalogue's clinical default"
        )


def test_every_pattern_compiles():
    """Catches a typo in a metadata pattern here rather than inside an
    EXECUTE IMMEDIATE in the warehouse, where the error arrives without a
    line number."""
    for code, cat in sorted(CATALOGUE.items()):
        for variant in ("loose", "tuned", "strict"):
            pt = VARIANTS[(code, variant)]["pattern_text"]
            try:
                cp = compile_pattern(pt, cat["define_text"])
            except PatternError as exc:
                raise AssertionError(f"{code}/{variant} failed to compile: {exc}") from exc
            for sym in cp.symbols:
                assert sym in cp.defines, f"{code}/{variant}: {sym} has no DEFINE"


def test_syndromes_fire_on_their_own_signature():
    for code, fx in sorted(FIXTURES.items()):
        cat = CATALOGUE[code]
        cp = compile_pattern(cat["pattern_text"], cat["define_text"])
        matches = match_partition(cp, fx["positive"])
        assert len(matches) == 1, (
            f"{code} should fire exactly once on its own signature, got {len(matches)}"
            f"\n  pattern: {cat['pattern_text']}"
            f"\n  states : {[r['state'] for r in fx['positive']]}"
        )
        m = matches[0]
        assert m.n_epochs == fx["expect_epochs"], (
            f"{code} matched {m.n_epochs} epochs, expected {fx['expect_epochs']}"
        )
        assert m.n_epochs >= int(cat["min_epochs"]), (
            f"{code}: min_epochs={cat['min_epochs']} exceeds the shortest possible "
            f"match ({m.n_epochs}); the confidence score would be negative"
        )


def test_syndromes_stay_silent_on_near_misses():
    for code, fx in sorted(FIXTURES.items()):
        cat = CATALOGUE[code]
        cp = compile_pattern(cat["pattern_text"], cat["define_text"])
        for why, rows in fx["negatives"].items():
            matches = match_partition(cp, rows)
            assert not matches, (
                f"{code} fired on a sequence that should not match ({why})"
                f"\n  pattern: {cat['pattern_text']}"
                f"\n  states : {[r['state'] for r in rows]}"
            )


def test_strictness_ladder_is_real():
    """'strict' must actually reject something 'tuned' accepts.

    Otherwise the sensitivity curve is three identical numbers and the chart is
    decoration.
    """
    for code, fx in sorted(FIXTURES.items()):
        cat = CATALOGUE[code]
        rows = fx["between_tuned_and_strict"]
        tuned = match_partition(
            compile_pattern(VARIANTS[(code, "tuned")]["pattern_text"], cat["define_text"]), rows
        )
        strict = match_partition(
            compile_pattern(VARIANTS[(code, "strict")]["pattern_text"], cat["define_text"]), rows
        )
        assert tuned, f"{code}: the tuned pattern should match the between-fixture"
        assert not strict, (
            f"{code}: the strict pattern also matched, so strict is not stricter "
            f"than tuned on this fixture"
        )


def test_loose_accepts_what_tuned_rejects():
    """The other end of the ladder: 'loose' must accept at least one near-miss
    that 'tuned' rejects, or relaxing the quantifiers changes nothing."""
    for code, fx in sorted(FIXTURES.items()):
        cat = CATALOGUE[code]
        loose_cp = compile_pattern(VARIANTS[(code, "loose")]["pattern_text"], cat["define_text"])
        tuned_cp = compile_pattern(VARIANTS[(code, "tuned")]["pattern_text"], cat["define_text"])
        widened = [
            why
            for why, rows in fx["negatives"].items()
            if match_partition(loose_cp, rows) and not match_partition(tuned_cp, rows)
        ]
        assert widened, (
            f"{code}: 'loose' rejected every near-miss that 'tuned' rejected. "
            f"The loose variant is not looser in any way this fixture set can see."
        )


def test_s1_reproduces_the_spec_smoke_test():
    """The build spec states the exact expected output of the S1 smoke test:
    one match, epochs 1 through 8, five scratch epochs. Reproduce it here so a
    regression in the pattern is caught without a warehouse."""
    cat = CATALOGUE["S1"]
    cp = compile_pattern(cat["pattern_text"], cat["define_text"])
    rows = seq(R(1), SH(1), SC(3), SH(1), SC(2))
    matches = match_partition(cp, rows)
    assert len(matches) == 1
    m = matches[0]
    assert m.match_id == 1
    assert m.start_index == 0 and m.end_index == 7          # rows 1..8, 0-indexed
    assert m.count("itch") + m.count("itch2") == 5          # scratch_epochs
    assert m.count("shake") + m.count("shake2") == 2        # shake_epochs
    assert m.symbol_string() == "onset shake itch itch itch shake2 itch2 itch2"


def test_classifier_output_is_the_hero_caption():
    """ALL ROWS PER MATCH + CLASSIFIER() is what colours the timeline by symbol.
    Every matched epoch must carry a symbol, and the symbols must be in pattern
    order."""
    for code, fx in sorted(FIXTURES.items()):
        cat = CATALOGUE[code]
        cp = compile_pattern(cat["pattern_text"], cat["define_text"])
        m = match_partition(cp, fx["positive"])[0]
        assert len(m.symbols) == m.n_epochs
        assert all(s in cp.defines for s in m.symbols), m.symbols
        # symbols appear in the order the pattern declares them
        order = {s: i for i, s in enumerate(dict.fromkeys(cp.symbols))}
        seen = [order[s] for s in m.symbols]
        assert seen == sorted(seen), f"{code}: symbols out of pattern order: {m.symbols}"
        # every matched epoch's state satisfies its symbol's DEFINE
        for row, sym in zip(m.rows, m.symbols):
            col, want = cp.defines[sym]
            assert row[col] in want, (
                f"{code}: {sym} bound a {row[col]!r} epoch, "
                f"which is not in {sorted(want)}"
            )


def test_partitioning_prevents_cross_session_matches():
    """PARTITION BY dog_id, test_num is a correctness requirement, not tidiness.

    Half a syndrome at the end of session 1 and half at the start of session 2
    must not join up into a phantom finding.
    """
    cat = CATALOGUE["S1"]
    cp = compile_pattern(cat["pattern_text"], cat["define_text"])
    first_half = seq(R(1), SH(1), SC(3), test_num=1)
    second_half = seq(SH(1), SC(2), test_num=2, start=T0 + timedelta(seconds=5))
    rows = first_half + second_half

    assert not match_all(cp, rows, partition_by=("dog_id", "test_num")), (
        "a match was assembled across two different sessions"
    )
    # ...and the same rows DO match when the session boundary is ignored, which
    # is exactly the bug the partition key prevents.
    assert match_all(cp, rows, partition_by=("dog_id",)), (
        "fixture is not exercising the boundary — it does not match either way"
    )


def test_quiet_on_a_long_unrelated_stream():
    """A day of ordinary behaviour must not manufacture findings.

    Not a false-positive rate — that needs the real data — but a floor: if a
    pattern fires on uniformly random posture and locomotion, it is not
    specific enough to be clinical.
    """
    rng = random.Random(20260817)
    ordinary = ["REST", "SIT", "STAND", "WALK", "TROT", "GALLOP", "SNIFF", "PLAY"]
    rows = [
        {
            "dog_id": 1,
            "test_num": 1,
            "epoch_ts": T0 + timedelta(seconds=i),
            "state": rng.choice(ordinary),
        }
        for i in range(20_000)
    ]
    noisy = []
    for code, cat in sorted(CATALOGUE.items()):
        cp = compile_pattern(cat["pattern_text"], cat["define_text"])
        n = len(match_partition(cp, rows))
        if n:
            noisy.append(f"{code}={n}")
    assert not noisy, (
        "patterns fired on 20,000 epochs of uniformly random ordinary behaviour: "
        + ", ".join(noisy)
        + ". These patterns rely on PAUSE/PACE/CIRCLE/SLOW_TRANSITION, which are "
          "absent from this stream, so any match is a specificity failure."
    )


def test_embedded_syndrome_is_found_in_noise():
    """The complement of the previous test: a real signature buried in a long
    ordinary stream must still be found, exactly once, at the right place."""
    rng = random.Random(11)
    ordinary = ["REST", "SIT", "STAND", "WALK", "TROT", "SNIFF", "PLAY"]

    def noise(n: int, t: datetime):
        return [
            {"dog_id": 3, "test_num": 1, "epoch_ts": t + timedelta(seconds=i),
             "state": rng.choice(ordinary)}
            for i in range(n)
        ]

    cat = CATALOGUE["S1"]
    cp = compile_pattern(cat["pattern_text"], cat["define_text"])

    head = noise(500, T0)
    signature = seq(R(1), SH(1), SC(4), SH(1), SC(3), start=T0 + timedelta(seconds=500))
    tail = noise(500, T0 + timedelta(seconds=500 + len(signature)))
    rows = head + signature + tail

    matches = match_partition(cp, rows)
    assert len(matches) == 1, f"expected exactly one embedded match, got {len(matches)}"
    m = matches[0]
    assert m.first_ts() == signature[0]["epoch_ts"], "match started in the wrong place"
    assert m.count("itch") + m.count("itch2") == 7


# ---------------------------------------------------------------------------
# standalone runner (so this works with no pytest installed)
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    width = max(len(n) for n, _ in tests)
    failures: list[tuple[str, BaseException]] = []

    print(f"\nTELLTAIL — offline pattern validation  ({len(tests)} tests)")
    print("=" * (width + 12))
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except BaseException as exc:  # noqa: BLE001
            print(f"  FAIL  {name}")
            failures.append((name, exc))

    print("=" * (width + 12))
    if failures:
        for name, exc in failures:
            print(f"\n--- {name} ---\n{exc}")
        print(f"\n{len(failures)} of {len(tests)} failed.")
        return 1

    print(f"\nAll {len(tests)} passed. The six patterns are satisfiable, specific,")
    print("and consistent between the views and the catalogue.")
    print("Gate E is still the warehouse — but you arrive at it knowing the")
    print("patterns say what you meant.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
