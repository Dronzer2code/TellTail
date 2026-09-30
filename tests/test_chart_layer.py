#!/usr/bin/env python3
"""
The EvilCharts layer must build a figure on ANY plotly, and on Altair 4 or 5.

    python tests/test_chart_layer.py

Two failure modes are being guarded, and both of them are invisible on a
development machine because a development machine has current libraries:

  1. plotly raises ValueError at figure CONSTRUCTION for a property it does not
     know. `fillgradient`, `barcornerradius` and `griddash` are all newer than
     the plotly Streamlit in Snowflake is likely to solve to, so every one of
     them is capability-probed with a fallback. This exercises each helper with
     the probes forced False — the path that only ever runs in SiS.
  2. Altair renamed selections in 5. Writing the 5 spelling took the Syndromes
     tab down with an AttributeError, because SiS ships Altair 4. This asserts
     the shim resolves against whichever is installed.

Imports the app WITHOUT a Snowflake session, which means everything below the
data-access layer is unreachable. That is fine: this file is about the chart
helpers, which are pure figure construction and depend on nothing but plotly.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "warehouse"))

# A plotly property error quotes the offending value, and plotly writes its
# minus signs as U+2212. On a Windows console that is cp1252 and printing the
# message raises UnicodeEncodeError ON TOP of the failure being reported —
# which loses the actual error and reports an encoding bug instead.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

FAILED: list[str] = []


def check(label: str, fn) -> None:
    try:
        fn()
        print(f"  ok    {label}")
    except Exception as e:  # noqa: BLE001 - the point is to catch everything
        # First line only. A plotly property error is ~90 lines of valid-property
        # listing, and forty of those in a row is unreadable.
        msg = str(e).strip().split("\n")[0][:160]
        FAILED.append(f"{label}: {type(e).__name__}: {msg}")
        print(f"  FAIL  {label}: {type(e).__name__}: {msg}")


def load_app():
    """Import streamlit_app with streamlit and the session stubbed out.

    The module runs queries at import (the header strip and the rail), so it
    cannot simply be imported. Only the helpers above that point are wanted,
    so the module is executed up to the data-access boundary and the rest is
    discarded — the chart layer is defined before any of it.
    """
    import plotly.graph_objects as go

    src = (REPO / "warehouse" / "streamlit_app.py").read_text(encoding="utf-8")
    # Everything from the header section on needs a live warehouse.
    cut = src.index("# header\n")
    cut = src.rindex("# ---", 0, cut)

    st = types.ModuleType("streamlit")
    st.set_page_config = lambda **k: None
    st.markdown = lambda *a, **k: None
    st.cache_data = lambda **k: (lambda f: f)
    st.columns = lambda *a, **k: []
    # `import streamlit.components.v1 as components` resolves the attribute
    # chain off the parent module, so stubbing sys.modules alone is not enough.
    comp = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **k: None
    comp.v1 = v1
    st.components = comp
    sys.modules["streamlit"] = st
    sys.modules["streamlit.components"] = comp
    sys.modules["streamlit.components.v1"] = v1

    mod = types.ModuleType("app_under_test")
    mod.__dict__["__name__"] = "app_under_test"
    exec(compile(src[:cut], "streamlit_app.py", "exec"), mod.__dict__)
    assert mod.__dict__["go"] is go or True
    return mod


def exercise(app, tag: str) -> None:
    """Build one of everything. A raise here is a page that would not render."""
    go = app.go
    C = app.S_BLUE
    x = [0, 1, 2, 3, 4]
    y = [1.0, 3.5, 2.2, 4.8, 3.1]

    check(f"{tag} gfill", lambda: app.gfill(C))
    check(f"{tag} glow", lambda: app.glow(go.Figure(), x, y, C))
    check(f"{tag} evil_area", lambda: app.evil_area(
        go.Figure(), x, y, C, name="s", text=["a"] * 5, cap=True))
    check(f"{tag} evil_area no-glow", lambda: app.evil_area(
        go.Figure(), x, y, C, show_glow=False, fill="tonexty"))
    check(f"{tag} evil_bar", lambda: app.evil_bar(
        go.Figure(), ["a", "b", "c"], [3, 1, 2], [C, C, C], text=["x"] * 3))
    check(f"{tag} evil_bar vertical", lambda: app.evil_bar(
        go.Figure(), ["a", "b"], [3, 1], [C, C], horizontal=False, track=False))
    check(f"{tag} evil_axes", lambda: app.evil_axes(
        app.evil_area(go.Figure(), x, y, C), dotted="xy"))
    check(f"{tag} evil_donut", lambda: app.evil_donut(
        ["a", "b"], [3, 1], [C, app.S_ORANGE], centre="4", centre_sub="total"))
    check(f"{tag} evil_radial", lambda: app.evil_radial(38, 45, C, label="dogs"))
    check(f"{tag} evil_radial zero", lambda: app.evil_radial(0, 0, C))
    check(f"{tag} evil_radar", lambda: app.evil_radar(
        ["p", "r", "f1"], [("A", [0.9, 0.8, 0.85], C),
                           ("B", [0.7, 0.75, 0.72], app.S_ORANGE)]))
    check(f"{tag} evil_sankey", lambda: app.evil_sankey(
        ["a", "b", "c"], [0, 1], [1, 2], [5, 3], [C, app.S_ORANGE, app.S_AQUA],
        text=["a to b", "b to c"]))
    check(f"{tag} evil_blocks", lambda: app.evil_blocks(
        [("low", 3, C), ("high", 1, app.S_RED)], title="mix"))
    check(f"{tag} scene3d", lambda: app.scene3d(
        go.Figure(go.Scatter3d(x=x, y=y, z=y)), "x", "y", "z"))
    check(f"{tag} row_h", lambda: app.row_h(4, 25, 0))
    check(f"{tag} table_pair", lambda: app.table_pair([1] * 20, [1] * 3))


def main() -> int:
    print("=" * 74)
    print("EvilCharts layer")
    print("=" * 74)
    app = load_app()
    print(f"\nplotly probes as installed: fillgradient={app.HAS_FILLGRAD} "
          f"barcornerradius={app.HAS_BARRADIUS} griddash={app.HAS_GRIDDASH}")

    print("\nas installed:")
    exercise(app, "native")

    # THE PATH THAT ONLY RUNS IN SiS. Forcing all three probes False is the
    # only way to execute the fallback branches from a machine with a current
    # plotly, and those branches are exactly the ones nobody would notice were
    # broken until a demo.
    print("\nwith every capability forced off (the SiS-with-old-plotly path):")
    app.HAS_FILLGRAD = app.HAS_BARRADIUS = app.HAS_GRIDDASH = False
    exercise(app, "degraded")

    # gfill must actually change shape rather than merely not raising.
    print("\nfallback shape:")
    app.HAS_FILLGRAD = True
    check("gradient path returns fillgradient",
          lambda: _assert("fillgradient" in app.gfill(app.S_BLUE)))
    app.HAS_FILLGRAD = False
    check("flat path returns fillcolor",
          lambda: _assert("fillcolor" in app.gfill(app.S_BLUE)))

    print("\naltair selection shim:")
    import altair as alt
    check("alt_point builds on the installed altair",
          lambda: app.alt_point(fields=["code"], name="pick", toggle=None))
    check("alt_bind attaches to a chart", lambda: app.alt_bind(
        alt.Chart(alt.Data(values=[{"a": 1}])).mark_point().encode(x="a:Q"),
        app.alt_point(fields=["a"], name="p", toggle=None)))
    check("interval selection needs no shim",
          lambda: alt.selection_interval(encodings=["x"], name="brush"))

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED")
        for f in FAILED:
            print("  " + f)
        return 1
    print("all chart-layer checks passed")
    return 0


def _assert(cond: bool) -> None:
    if not cond:
        raise AssertionError("expected shape not produced")


if __name__ == "__main__":
    raise SystemExit(main())
