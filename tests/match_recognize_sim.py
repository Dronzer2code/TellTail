"""
An offline simulator for Snowflake row pattern recognition.

Why this exists
---------------
Gate E — "one syndrome pattern matching on real data" — is the riskiest thing in
the build, and it is normally the first moment anyone discovers that a pattern
is unsatisfiable, that a quantifier is off by one, or that two symbols in the
same pattern can never both bind. Discovering that at hour thirty, against a
live warehouse, on a trial credit budget, is the expensive way to find out.

Every pattern in REF.SYNDROME_CATALOGUE is a concatenation of pattern variables
with greedy quantifiers, and every DEFINE is a simple equality on `state`. That
subset of MATCH_RECOGNIZE is exactly equivalent to a regular expression over a
string in which each row contributes one character. So the patterns can be
compiled to Python `re` and exercised against fixtures on a laptop, with no
account, no credits and no network.

What is faithfully modelled
---------------------------
  * PATTERN concatenation, and the quantifiers  +  *  ?  {n}  {n,}  {n,m}
  * greedy matching with backtracking (Python `re` and Snowflake agree here:
    both are leftmost-first with greedy quantifiers and backtracking)
  * ONE ROW PER MATCH and ALL ROWS PER MATCH
  * AFTER MATCH SKIP PAST LAST ROW   (`re.finditer`'s non-overlapping scan)
  * PARTITION BY  (each partition is matched independently, in ORDER BY order)
  * MEASURES of the shape COUNT(sym.*), COUNT(*), FIRST/LAST(sym.col), CLASSIFIER()

What is deliberately NOT modelled
---------------------------------
  * DEFINE predicates referencing PREV/NEXT/aggregates, or any column other
    than the one the symbol tests. TELLTAIL's catalogue does not use them; if a
    pattern ever does, compile() raises rather than quietly returning a wrong
    answer.
  * alternation and grouping in PATTERN — again, unused, and refused loudly.
  * AFTER MATCH SKIP TO variants other than PAST LAST ROW.

This is a semantic cross-check, not a replacement for running the real thing.
The gate is still the warehouse. This just means you arrive at the gate knowing
the pattern is satisfiable and the quantifiers say what you meant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "CompiledPattern",
    "Match",
    "compile_pattern",
    "parse_define",
    "match_partition",
    "match_all",
]

# A pattern element: a variable name plus an optional quantifier.
_ELEMENT_RE = re.compile(
    r"""
    (?P<var>[A-Za-z_][A-Za-z0-9_]*)          # pattern variable
    (?P<quant>
        \{\s*\d+\s*,\s*\d*\s*\}              # {n,}  {n,m}
      | \{\s*\d+\s*\}                        # {n}
      | [+*?]                                # + * ?
    )?
    """,
    re.VERBOSE,
)

_DEFINE_RE = re.compile(
    r"""
    (?P<var>[A-Za-z_][A-Za-z0-9_]*)
    \s+AS\s+
    (?P<col>[A-Za-z_][A-Za-z0-9_]*)
    \s*
    (?:
        =\s*'(?P<val>[^']*)'                 # col = 'LITERAL'
      | IN\s*\(\s*(?P<vals>'[^)]*')\s*\)     # col IN ('A','B','C')
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_IN_LIST_RE = re.compile(r"'([^']*)'")

# Symbols are encoded as single characters so the whole pattern becomes a plain
# regex over a string. Private-use Unicode keeps them clear of any real value.
_ENCODE_BASE = 0xE000


class PatternError(ValueError):
    """A pattern or DEFINE clause this simulator refuses to guess about."""


def parse_define(define_text: str) -> dict[str, tuple[str, frozenset[str]]]:
    """'a AS state = ''REST'', b AS state IN (''SIT'',''STAND'')'
        -> {a: (state, {REST}), b: (state, {SIT, STAND})}

    Accepts both SQL-escaped ('') and plain ('') quoting, because the catalogue
    stores the escaped form and hand-written SQL uses the plain one.

    Both `= 'literal'` and `IN (...)` are modelled. The set form is not a
    convenience: S3's `recover` has to mean "stopped moving" rather than "lying
    down" — TROT -> REST never occurs — and S1's onset posture spans three
    states. A simulator that only understood equality would have to refuse the
    real catalogue, and a refusing simulator validates nothing.
    """
    text = define_text.replace("''", "'")
    out: dict[str, tuple[str, frozenset[str]]] = {}
    for m in _DEFINE_RE.finditer(text):
        if m.group("val") is not None:
            vals = frozenset({m.group("val")})
        else:
            vals = frozenset(_IN_LIST_RE.findall(m.group("vals")))
        if not vals:
            raise PatternError(f"empty value set for {m.group('var')!r}")
        out[m.group("var")] = (m.group("col").lower(), vals)
    if not out:
        raise PatternError(f"no DEFINE bindings parsed from: {define_text!r}")

    # Anything left over after removing what we understood is a predicate form
    # this simulator does not model. Refuse rather than silently mis-simulate.
    residue = _DEFINE_RE.sub("", text)
    residue = re.sub(r"[\s,]+", "", residue)
    if residue:
        raise PatternError(
            f"DEFINE contains predicates this simulator does not model: {residue!r}. "
            f"Only `var AS col = 'literal'` and `var AS col IN (...)` are supported."
        )
    return out


@dataclass
class CompiledPattern:
    """A MATCH_RECOGNIZE PATTERN/DEFINE pair compiled to a regex."""

    pattern_text: str
    define_text: str
    symbols: list[str]                       # in pattern order, may repeat
    defines: dict[str, tuple[str, frozenset[str]]]  # var -> (column, accepted values)
    regex: re.Pattern[str]
    _sym_to_chars: dict[str, frozenset[str]] = field(repr=False, default_factory=dict)
    _char_to_states: dict[str, str] = field(repr=False, default_factory=dict)
    _val_to_char: dict[str, str] = field(repr=False, default_factory=dict)
    _columns: tuple[str, ...] = field(repr=False, default=())

    def encode(self, rows: Sequence[dict[str, Any]]) -> str:
        """One row -> one character, keyed on the row's VALUE rather than on the
        first symbol that accepts it. With set-valued DEFINEs several symbols
        can accept the same row, so per-symbol encoding would pick one
        arbitrarily and the regex would lose the alternatives. A row no symbol
        accepts gets a character no symbol's class contains, which is exactly
        right: it can never participate in a match."""
        out: list[str] = []
        for r in rows:
            ch = ""  # the "matches nothing" character
            for col in self._columns:
                got = self._val_to_char.get(str(r.get(col)))
                if got:
                    ch = got
                    break
            out.append(ch)
        return "".join(out)


def compile_pattern(pattern_text: str, define_text: str) -> CompiledPattern:
    """Compile PATTERN + DEFINE into a regex over encoded rows."""
    if any(c in pattern_text for c in "|()"):
        raise PatternError(
            f"alternation/grouping is not modelled by this simulator: {pattern_text!r}"
        )

    defines = parse_define(define_text)

    # Every distinct STATE VALUE gets one character. Two symbols testing the
    # same value (itch and itch2 both = 'SCRATCH') therefore share a character,
    # which is correct: a row satisfies both, and the regex decides which symbol
    # consumes it exactly as the matcher would.
    # One column per pattern, enforced. The encoder gives each row a single
    # character, so a pattern whose symbols test two different columns cannot be
    # represented faithfully — one column would silently win and the other's
    # symbols would match rows they should not. S3 matches activity_class and
    # every other pattern matches state; mixing them inside one pattern is a
    # modelling error this refuses rather than mis-simulates.
    cols = {col for col, _ in defines.values()}
    if len(cols) > 1:
        raise PatternError(
            f"pattern mixes DEFINE columns {sorted(cols)}; this simulator "
            f"encodes one row as one character and cannot model that"
        )

    values = sorted({v for _, vs in defines.values() for v in vs})
    val_to_char = {v: chr(_ENCODE_BASE + i) for i, v in enumerate(values)}
    sym_to_chars = {var: frozenset(val_to_char[v] for v in vs)
                    for var, (_, vs) in defines.items()}

    parts: list[str] = []
    symbols: list[str] = []
    pos = 0
    text = pattern_text.strip()
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _ELEMENT_RE.match(text, pos)
        if not m:
            raise PatternError(f"cannot parse pattern at offset {pos}: {text[pos:]!r}")
        var = m.group("var")
        if var not in defines:
            raise PatternError(
                f"pattern variable {var!r} has no DEFINE binding "
                f"(defined: {sorted(defines)})"
            )
        symbols.append(var)
        # A character CLASS, not a single character: a set-valued DEFINE accepts
        # any of its states, and `[abc]` is how that composes with a quantifier.
        cls = "[" + "".join(re.escape(c) for c in sorted(sym_to_chars[var])) + "]"
        parts.append(cls + (m.group("quant") or ""))
        pos = m.end()

    if not parts:
        raise PatternError(f"empty pattern: {pattern_text!r}")

    return CompiledPattern(
        pattern_text=pattern_text,
        define_text=define_text,
        symbols=symbols,
        defines=defines,
        regex=re.compile("".join(parts)),
        _sym_to_chars=sym_to_chars,
        _char_to_states={c: v for v, c in val_to_char.items()},
        _val_to_char=val_to_char,
        _columns=tuple(dict.fromkeys(col for col, _ in defines.values())),
    )


@dataclass
class Match:
    """One match, in the shape MARTS.SYNDROME_MATCHES stores."""

    partition: tuple
    match_id: int
    start_index: int
    end_index: int                    # inclusive
    n_epochs: int
    rows: list[dict[str, Any]]
    symbols: list[str]                # per-row CLASSIFIER(), same length as rows

    def count(self, symbol: str) -> int:
        """COUNT(symbol.*)"""
        return sum(1 for s in self.symbols if s == symbol)

    def first_ts(self, key: str = "epoch_ts") -> Any:
        return self.rows[0][key]

    def last_ts(self, key: str = "epoch_ts") -> Any:
        return self.rows[-1][key]

    def symbol_string(self) -> str:
        """'onset shake itch itch itch shake itch itch' — the hero caption."""
        return " ".join(self.symbols)


def _assign_symbols(cp: CompiledPattern, encoded: str) -> list[str]:
    """Recover per-row CLASSIFIER() for a matched span.

    The regex tells us the span but not which element consumed which character,
    so the element sequence is replayed greedily against the span — the same
    order the matcher used. For the concatenation-only grammar this simulator
    accepts, that reconstruction is exact.
    """
    elements: list[tuple[str, int, int]] = []  # (var, min, max)
    pos = 0
    text = cp.pattern_text.strip()
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _ELEMENT_RE.match(text, pos)
        assert m is not None
        var, quant = m.group("var"), m.group("quant")
        if quant is None:
            lo = hi = 1
        elif quant == "+":
            lo, hi = 1, len(encoded)
        elif quant == "*":
            lo, hi = 0, len(encoded)
        elif quant == "?":
            lo, hi = 0, 1
        else:
            body = quant.strip("{}").strip()
            if "," in body:
                a, b = body.split(",", 1)
                lo = int(a.strip())
                hi = int(b.strip()) if b.strip() else len(encoded)
            else:
                lo = hi = int(body)
        elements.append((var, lo, hi))
        pos = m.end()

    # Greedy backtracking assignment over the already-matched span.
    out: list[str] = []

    def rec(ei: int, si: int) -> bool:
        if ei == len(elements):
            return si == len(encoded)
        var, lo, hi = elements[ei]
        chars = cp._sym_to_chars[var]
        avail = 0
        while si + avail < len(encoded) and encoded[si + avail] in chars and avail < hi:
            avail += 1
        for take in range(avail, lo - 1, -1):
            if take < lo:
                break
            out.extend([var] * take)
            if rec(ei + 1, si + take):
                return True
            del out[len(out) - take:]
        return False

    if not rec(0, 0):
        # Should be unreachable: the regex already matched this exact span.
        return ["?"] * len(encoded)
    return out


def match_partition(
    cp: CompiledPattern,
    rows: Sequence[dict[str, Any]],
    partition: tuple = (),
) -> list[Match]:
    """Run one partition, already ordered. AFTER MATCH SKIP PAST LAST ROW."""
    encoded = cp.encode(rows)
    out: list[Match] = []
    # finditer is non-overlapping and resumes after the end of each match, which
    # is precisely AFTER MATCH SKIP PAST LAST ROW.
    for i, m in enumerate(cp.regex.finditer(encoded), start=1):
        if m.start() == m.end():
            continue  # a zero-length match is not a clinical sequence
        span_rows = list(rows[m.start() : m.end()])
        out.append(
            Match(
                partition=partition,
                match_id=i,
                start_index=m.start(),
                end_index=m.end() - 1,
                n_epochs=len(span_rows),
                rows=span_rows,
                symbols=_assign_symbols(cp, encoded[m.start() : m.end()]),
            )
        )
    return out


def match_all(
    cp: CompiledPattern,
    rows: Iterable[dict[str, Any]],
    partition_by: Sequence[str] = ("dog_id", "test_num"),
    order_by: str = "epoch_ts",
) -> list[Match]:
    """PARTITION BY ... ORDER BY ... over a full row set."""
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in partition_by)
        buckets.setdefault(key, []).append(r)

    out: list[Match] = []
    for key, part in sorted(buckets.items(), key=lambda kv: [str(x) for x in kv[0]]):
        part.sort(key=lambda r: r[order_by])
        out.extend(match_partition(cp, part, partition=key))
    return out
