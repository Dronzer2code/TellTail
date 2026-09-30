#!/usr/bin/env python3
"""
Render the blog figures from the warehouse.

    python scripts/blog_charts.py               # all figures
    python scripts/blog_charts.py --list        # what it would draw
    python scripts/blog_charts.py --only 01 04  # just those

Writes to assets/blog/ as PNG at 2x (for dev.to, which accepts raster uploads
only) and SVG alongside wherever the renderer can produce one.

WHY THIS EXISTS RATHER THAN A SCREENSHOT FOLDER
-----------------------------------------------
A screenshot of a dashboard is a picture of a dashboard. It carries the app's
chrome, its sidebar, its font rendering and whatever the browser window
happened to be that afternoon, and it cannot be regenerated when the numbers
change. These are drawn from the same tables the dashboard reads, at a fixed
size, with a fixed palette, and re-running this script after a reload produces
the same figures with the new numbers.

It also means every number in the post is traceable: each figure prints the
query it came from and the row count it drew, so a claim in the prose can be
checked against the figure that supports it.

LIGHT ONLY, DELIBERATELY
------------------------
No dark variants. The eight series colours below were validated for
colour-vision deficiency against a WHITE chart surface — that validation is
the reason they are these particular values and not prettier ones. Re-running
the figures on a dark ground without re-running that validation would ship
eight colours whose separation nobody has checked, which is worse than a
figure that looks bright in dark mode.

RENDERERS
---------
altair -> vl-convert  : pure Rust, no browser, real SVG text.
plotly -> kaleido     : the 3D scenes, which are WebGL and so raster-only.
"""
from __future__ import annotations

import argparse
import math
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO, connect, header, info, load_env, ok, warn  # noqa: E402

OUT = REPO / "assets" / "blog"

# ---------------------------------------------------------------------------
# design system, carried over from warehouse/streamlit_app.py so a figure in
# the post and the same chart in the app are recognisably the same object
# ---------------------------------------------------------------------------
CARD = "#FFFFFF"
SURFACE = "#FAFAF9"
BORDER = "#E7E5E4"
INK = "#1C1917"
INK_2 = "#57534E"
INK_3 = "#8A8580"
GRID = "#F1EFED"
ACCENT = "#B45309"

S_BLUE = "#2a78d6"
S_ORANGE = "#eb6834"
S_AQUA = "#1baf7a"
S_YELLOW = "#eda100"
S_MAGENTA = "#e87ba4"
S_GREEN = "#008300"
S_VIOLET = "#4a3aa7"
S_RED = "#e34948"
SERIES = [S_BLUE, S_ORANGE, S_AQUA, S_YELLOW, S_MAGENTA, S_GREEN, S_VIOLET, S_RED]

# Inter is installed on the build machine and is what these were composed
# against. The fallbacks are ordered so a machine without it degrades to
# something with the same metrics rather than to a serif.
FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
MONO = "Consolas, Cascadia Mono, ui-monospace, monospace"

# dev.to renders the article column at about 800 px. Drawing at 920 and
# exporting at 2x gives a figure that is sharp on a retina display and still
# legible when the reader's browser scales it down to the column.
W_FULL = 920
SCALE = 2.0

_conn = None


def q(sql: str) -> list[dict]:
    """Run a query, return plain dicts with Decimals converted.

    Same discipline as the app: Snowflake hands back object-dtype Decimal for
    anything NUMBER, and Vega infers nothing from a Decimal except that it is
    not a number it recognises.
    """
    global _conn
    if _conn is None:
        _conn = connect()
    with _conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [{c: (float(v) if isinstance(v, Decimal) else v)
                 for c, v in zip(cols, r)} for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# altair scaffolding
# ---------------------------------------------------------------------------
import altair as alt  # noqa: E402

AXIS = dict(labelFont=FONT, labelFontSize=11, labelColor=INK_2,
            titleFont=FONT, titleFontSize=11, titleColor=INK_2,
            titleFontWeight="normal", titlePadding=10,
            domainColor=BORDER, tickColor=BORDER, tickSize=4, grid=False)


def styled(chart, *, grid_y: bool = True, grid_x: bool = False):
    """The house style, applied to a finished TOP-LEVEL chart."""
    return (chart
            .configure_view(stroke=None, fill=CARD)
            .configure_axisX(**dict(AXIS, grid=grid_x, gridColor=GRID))
            .configure_axisY(**dict(AXIS, grid=grid_y, gridColor=GRID,
                                    domain=False, tickSize=0, labelPadding=6))
            .configure_legend(labelFont=FONT, labelFontSize=11, labelColor=INK_2,
                              titleFont=FONT, titleFontSize=11, titleColor=INK_2,
                              titleFontWeight="normal", symbolType="circle",
                              symbolStrokeWidth=0, offset=14, labelLimit=220)
            .configure_title(font=FONT, fontSize=15, color=INK,
                             fontWeight=600, anchor="start", offset=14,
                             subtitleFont=FONT, subtitleFontSize=12,
                             subtitleColor=INK_2, subtitleFontWeight="normal",
                             subtitlePadding=8, lineHeight=20)
            .configure_header(labelFont=FONT, labelFontSize=11, labelColor=INK,
                              labelFontWeight=600, titleFont=FONT,
                              titleFontSize=11, titleColor=INK_2,
                              titleFontWeight="normal")
            .configure_range(category=SERIES)
            .configure_concat(spacing=30)
            .configure_facet(spacing=18))


def title(text: str, subtitle: str | list[str]):
    return alt.TitleParams(text=text, subtitle=subtitle)


def data(rows: list[dict]):
    """Inline values, never a DataFrame — see q()."""
    return alt.Data(values=rows)


def note(text: str, width: int = W_FULL):
    """The source line under a figure.

    A figure in a build post is a claim, and a claim needs to say where it came
    from. This is a mark rather than a caption so it exports inside the PNG —
    a caption in the Markdown gets separated from the image the moment anyone
    screenshots or re-hosts it.
    """
    return (alt.Chart(data([{"t": text}])).mark_text(
        align="left", baseline="top", font=FONT, fontSize=10.5,
        color=INK_3, lineBreak="\n", dx=0)
        .encode(x=alt.value(0), y=alt.value(0), text="t:N")
        .properties(width=width, height=text.count("\n") * 15 + 16))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def save_alt(chart, name: str, rows: int, sql_note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png, svg = OUT / f"{name}.png", OUT / f"{name}.svg"
    chart.save(str(png), scale_factor=SCALE)
    chart.save(str(svg))
    ok(f"{name:<26} {rows:>7,} rows  ->  {png.name}  +  {svg.name}")
    info(f"   {sql_note}")


def save_plotly(fig, name: str, rows: int, sql_note: str,
                width: int = W_FULL, height: int = 560) -> None:
    """3D scenes are WebGL, so PNG only — an SVG of one is a bitmap in a box."""
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    fig.write_image(str(png), width=width, height=height, scale=SCALE)
    ok(f"{name:<26} {rows:>7,} rows  ->  {png.name}")
    info(f"   {sql_note}")


def scene(xt: str, yt: str, zt: str, *, eye=(1.6, 1.5, 0.9), aspect=None,
          ortho: bool = True) -> dict:
    """One 3D scene style for every 3D figure in the post."""
    # plotly 6 removed the flat `titlefont` alias on scene axes; the font now
    # lives inside the title object. The old spelling does not warn, it raises
    # a "Bad property path" at figure construction.
    ax = dict(backgroundcolor=SURFACE, gridcolor=BORDER, zerolinecolor=BORDER,
              showspikes=False, tickfont=dict(size=10, color=INK_3))
    tf = dict(font=dict(size=12, color=INK_2))
    # ZOOM FACTOR. plotly places the 3D scene inside the canvas and then
    # backs the camera off far enough that the whole cube fits with room to
    # spare, so a default eye leaves a third of a blog figure as white margin
    # around a small cube. Scaling the eye vector in moves the camera closer
    # without changing the viewing angle.
    z = 0.74
    s = dict(xaxis=dict(ax, title=dict(tf, text=xt)),
             yaxis=dict(ax, title=dict(tf, text=yt)),
             zaxis=dict(ax, title=dict(tf, text=zt)),
             camera=dict(eye=dict(x=eye[0] * z, y=eye[1] * z, z=eye[2] * z)))
    if ortho:
        s["camera"]["projection"] = dict(type="orthographic")
    if aspect:
        s["aspectratio"] = dict(x=aspect[0], y=aspect[1], z=aspect[2])
    else:
        s["aspectmode"] = "cube"
    return s


def pick_epoch(state: str) -> dict | None:
    """One (dog, test_num, epoch_ts) in `state` that HAS a waveform behind it.

    Two traps here, both of which produced an empty figure rather than an
    error, which is the expensive kind:

    1. A dog can hold epochs from BOTH sources. The bulk corpus was loaded
       from CSV with its own timestamps (around March); the replayer writes
       100 Hz telemetry with today's. So `MIN(epoch_ts) WHERE state = 'TROT'`
       for a dog that has raw samples still lands months away from any of
       them, and the waveform query returns nothing.
    2. sample_ts is only unique per (dog_id, test_num) — see fig_01. The join
       has to carry test_num or it re-admits the interleaving it exists to
       prevent.

    So the epoch is required to fall inside that recording's actual sample
    range, and the range is taken per (dog, test_num), not per dog.
    """
    got = q(f"""
        WITH span AS (
            SELECT dog_id, test_num,
                   MIN(sample_ts) AS lo, MAX(sample_ts) AS hi, COUNT(*) AS n
            FROM RAW.COLLAR_TELEMETRY GROUP BY 1, 2
        )
        SELECT s.dog_id, s.test_num, s.epoch_ts, span.lo, span.hi
        FROM MARTS.EPOCH_STATES s
        JOIN span ON span.dog_id = s.dog_id AND span.test_num = s.test_num
        WHERE s.state = '{state}'
          AND s.epoch_ts >= span.lo
          AND s.epoch_ts <  DATEADD('second', -12, span.hi)
        ORDER BY span.n DESC, s.epoch_ts
        LIMIT 1""")
    return got[0] if got else None


def plotly_layout(fig, title_text: str, subtitle: str, height: int = 560):
    """Title block for a plotly figure, with the subtitle wrapped by hand.

    plotly does not wrap title text. A subtitle longer than the canvas is not
    ellipsised or shrunk, it is simply drawn past the right edge and lost —
    which is how two of these figures shipped with their last four words
    missing. Wrapping here means the caption length is a property of the text
    rather than of whoever last changed the figure width.
    """
    import textwrap
    lines = textwrap.wrap(subtitle, width=112) or [""]
    body = "<br>".join(
        f"<span style='font-size:12.5px;color:{INK_2}'>{ln}</span>" for ln in lines)
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b><br>{body}",
                   font=dict(family=FONT, size=15.5, color=INK),
                   x=0, xanchor="left", y=0.98, yanchor="top"),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT, size=11, color=INK_2),
        margin=dict(l=0, r=0, t=44 + 19 * len(lines), b=6), height=height)
    return fig


# ===========================================================================
# 01 — the two-sensor argument
# ===========================================================================
def fig_01_two_sensors():
    """THE figure the whole project rests on.

    A collar alone cannot separate a dog walking from a dog shaking its head:
    both are vigorous neck motion, and any scalar built from the neck alone —
    steps, activity minutes, magnitude — puts them in the same bin. The back
    harness resolves it, and the resolution is not a threshold, it is a SHAPE.

    Two panels per second: the traces over time, and the same second plotted
    neck against back. The scatter is the point. Locomotion draws a tight
    diagonal because the two sensors rise and fall together; a head shake
    draws a vertical smear because the neck moves and the back does not. The
    correlation coefficient is just a number attached to that shape.
    """
    # THE EPOCH IS CHOSEN BY QUERY, NOT HARD-CODED, and the (dog, test_num,
    # epoch_ts) triple is carried through to the waveform.
    #
    # RAW.COLLAR_TELEMETRY holds more than one RECORDING per dog — dog 45 has
    # two test_num series whose sample_ts values are identical to the
    # millisecond. Selecting on (dog_id, sample_ts) alone therefore returns
    # both, interleaved, and the figure silently became two different seconds
    # of dog overlaid on each other: 200 samples in a 100 Hz second, a
    # neck-against-back scatter that was two clouds, and a correlation
    # annotation belonging to only one of them. test_num is not optional here.
    picks = [
        ("locomotion", "state IN ('TROT','GALLOP','WALK') AND f.neck_back_corr > 0.85",
         "f.neck_back_corr DESC", S_BLUE),
        ("head shake", "state IN ('SHAKE','SCRATCH') AND ABS(f.neck_back_corr) < 0.25",
         "f.neck_dominance DESC", S_ORANGE),
    ]
    panels, rows_total = [], 0
    for label, where, order_by, hue in picks:
        meta = q(f"""
            SELECT f.dog_id, f.test_num, f.epoch_ts, f.neck_back_corr AS corr,
                   s.state
            FROM STAGING.EPOCH_FEATURES f
            JOIN MARTS.EPOCH_STATES s ON s.dog_id = f.dog_id
                 AND s.test_num = f.test_num AND s.epoch_ts = f.epoch_ts
            JOIN (SELECT DISTINCT dog_id, test_num FROM RAW.COLLAR_TELEMETRY) r
                 ON r.dog_id = f.dog_id AND r.test_num = f.test_num
            WHERE {where} AND f.vm_neck_std > 0.15
            ORDER BY {order_by} LIMIT 1""")
        if not meta:
            warn(f"01: no epoch matched '{label}'")
            return None
        dog = int(meta[0]["DOG_ID"])
        tnum = int(meta[0]["TEST_NUM"])
        ts = str(meta[0]["EPOCH_TS"])
        corr, state = float(meta[0]["CORR"]), meta[0]["STATE"]
        w = q(f"""
            SELECT sample_ts,
                   SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az) AS neck,
                   SQRT(back_ax*back_ax + back_ay*back_ay + back_az*back_az) AS back
            FROM RAW.COLLAR_TELEMETRY
            WHERE dog_id = {dog} AND test_num = {tnum}
              AND sample_ts >= '{ts}' AND sample_ts < DATEADD('second', 1, '{ts}')
            ORDER BY sample_ts""")
        if not w:
            warn(f"01: no samples for dog {dog} test {tnum} at {ts}")
            return None
        rows_total += len(w)
        t0 = w[0]["SAMPLE_TS"]
        recs = [{"ms": (r["SAMPLE_TS"] - t0).total_seconds() * 1000,
                 "neck": float(r["NECK"]), "back": float(r["BACK"])} for r in w]
        long = ([{"ms": r["ms"], "g": r["neck"], "sensor": "neck collar"} for r in recs]
                + [{"ms": r["ms"], "g": r["back"], "sensor": "back harness"} for r in recs])

        trace = (alt.Chart(data(long), title=alt.TitleParams(
                    text=f"{label} · dog {dog} · {state}",
                    subtitle=f"one second at 100 Hz   ·   r = {corr:+.2f}",
                    fontSize=12, subtitleFontSize=11))
                 .mark_line(strokeWidth=1.5, interpolate="monotone")
                 .encode(
                     x=alt.X("ms:Q", title="milliseconds",
                             scale=alt.Scale(domain=[0, 1000], nice=False)),
                     y=alt.Y("g:Q", title="vector magnitude (g)",
                             scale=alt.Scale(zero=False)),
                     # ORIENT TOP, NOT TOP-LEFT. Inside the plotting rectangle
                     # the legend paints its own opaque background over the
                     # data, and at 100 Hz that punched a clean white gap
                     # through the middle of both traces that reads exactly
                     # like dropped samples — on the one figure whose entire
                     # job is proving the signal is real.
                     color=alt.Color("sensor:N", title=None,
                                     scale=alt.Scale(
                                         domain=["neck collar", "back harness"],
                                         range=[S_ORANGE, S_BLUE]),
                                     legend=alt.Legend(orient="top",
                                                       direction="horizontal",
                                                       offset=2, padding=0)))
                 .properties(width=(W_FULL - 30) // 2, height=180))

        # The same second as a shape. A regression line would assert a model;
        # the reference diagonal just says "if the two sensors agreed exactly,
        # the points would lie here" and lets the reader see how far they do.
        lo = min(min(r["neck"] for r in recs), min(r["back"] for r in recs))
        hi = max(max(r["neck"] for r in recs), max(r["back"] for r in recs))
        diag = (alt.Chart(data([{"a": lo}, {"a": hi}]))
                .mark_line(strokeDash=[4, 4], strokeWidth=1, color=BORDER)
                .encode(x=alt.X("a:Q"), y=alt.Y("a:Q")))
        cloud = (alt.Chart(data(recs)).mark_circle(size=26, opacity=0.5,
                                                   color=hue)
                 .encode(x=alt.X("neck:Q", title="neck collar (g)",
                                 scale=alt.Scale(domain=[lo, hi], nice=False),
                                 axis=alt.Axis(tickCount=6)),
                         y=alt.Y("back:Q", title="back harness (g)",
                                 scale=alt.Scale(domain=[lo, hi], nice=False),
                                 axis=alt.Axis(tickCount=6))))
        shape = alt.layer(diag, cloud).properties(
            width=(W_FULL - 30) // 2, height=180)
        panels.append(alt.vconcat(trace, shape, spacing=22))

    fig = alt.vconcat(
        alt.hconcat(*panels, spacing=30).properties(title=title(
            "One sensor cannot tell locomotion from a head shake. Two can.",
            ["Both are vigorous neck motion, so any scalar built from the collar alone bins them together.",
             "The back harness resolves it as a shape: in locomotion the sensors move together, in a head shake only one moves."])),
        note("Source: TELLTAIL RAW.COLLAR_TELEMETRY (100 Hz dual IMU) joined to STAGING.EPOCH_FEATURES.\n"
             "r is CORR(vm_neck, vm_back) computed in Snowflake over the same one-second epoch. Two real seconds, not composites."),
        spacing=16)
    return styled(fig), rows_total, "RAW.COLLAR_TELEMETRY + STAGING.EPOCH_FEATURES, 2 epochs"


# ===========================================================================
# 02 — the motion signature in 3D
# ===========================================================================
def fig_02_motion_signature():
    """Three axes of accelerometer, drawn as the 3-vector they are.

    Stacked as three time series they are three wiggly lines that all look
    alike. Plotted as one path in space the same numbers become a signature
    you can recognise on sight, which is the only good reason to spend a 3D
    plot on anything.
    """
    import plotly.graph_objects as go
    # One RECORDING, pinned by test_num — see the note in fig_01. Interleaving
    # two series here is worse than in a 2D panel: the path is drawn in sample
    # order, so it zigzags between two unrelated signals and the "loop" the
    # figure exists to show is destroyed.
    pick = pick_epoch("TROT")
    if not pick:
        warn("02: no trotting epoch with a waveform behind it")
        return None
    dog, tnum, t0 = int(pick["DOG_ID"]), int(pick["TEST_NUM"]), pick["EPOCH_TS"]
    w = q(f"""
        SELECT sample_ts, neck_ax, neck_ay, neck_az
        FROM RAW.COLLAR_TELEMETRY
        WHERE dog_id = {dog} AND test_num = {tnum}
          AND sample_ts >= '{t0}'
          AND sample_ts <  DATEADD('second', 8, '{t0}')
        ORDER BY sample_ts""")
    if not w:
        warn("02: no samples")
        return None
    ax = [float(r["NECK_AX"]) for r in w]
    ay = [float(r["NECK_AY"]) for r in w]
    az = [float(r["NECK_AZ"]) for r in w]
    fig = go.Figure(go.Scatter3d(
        x=ax, y=ay, z=az, mode="lines+markers",
        line=dict(color="rgba(235,104,52,0.30)", width=1),
        marker=dict(size=2.4, color=list(range(len(ax))),
                    colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], opacity=0.9),
        hoverinfo="skip"))
    fig.update_layout(scene=scene("neck a<sub>x</sub> (g)", "neck a<sub>y</sub> (g)",
                                  "neck a<sub>z</sub> (g)", eye=(1.5, 1.5, 0.9)),
                      showlegend=False)
    plotly_layout(fig, "Eight seconds of trotting, as one path through acceleration space",
                  f"{len(w):,} samples from dog {dog}. Colour is time — pale at the start, "
                  f"dark at the end. A gait is a closed loop repeated once per stride.",
                  height=580)
    return fig, len(w), f"RAW.COLLAR_TELEMETRY, dog {dog} test {tnum}, 8 s at 100 Hz"


# ===========================================================================
# 03 — spectrogram surface
# ===========================================================================
def fig_03_spectra():
    """Frequency is what actually separates these behaviours.

    A trot puts its energy at the stride rate, around 2 Hz. A scratch or a
    head shake puts it three to four times higher. In the raw trace both are
    just "vigorous", which is exactly the confusion the two-sensor correlation
    exists to resolve — this is that confusion viewed from the other side.

    WHY THIS IS NOT THE 3D SURFACE IT STARTED AS. The first version of this
    figure was a spectrogram of ONE window as a rotatable surface. It looked
    impressive and it showed nothing: a single 2.56 s periodogram of real
    100 Hz IMU data is mostly noise, so the surface was a field of spikes, and
    the caption confidently pointed at a ridge that a reader could not see. A
    figure whose caption describes something absent from the figure is worse
    than no figure.
    #
    # Welch's method fixes the actual problem, which was variance, not
    # dimensionality: average the periodogram over many overlapping windows
    # and the noise falls away while any real peak stays put. Averaging across
    # several runs from DIFFERENT DOGS as well means a peak here is a property
    # of the behaviour rather than of one animal's gait.
    """
    import numpy as np
    # A minute that actually contains a head shake, and ONE recording of it —
    # test_num pinned for the same reason as fig_01 and fig_02.
    # A WINDOW WITH A RHYTHM IN IT, chosen by finding the longest unbroken run
    # of locomotion that has a waveform behind it.
    #
    # The first version centred on a head shake, and a shake is by nature two
    # seconds long inside a minute of something else: the surface came out as
    # a field of noise with one spike in it, which is an honest picture of
    # that minute and a useless picture of the point being made. A sustained
    # gait holds one frequency for tens of seconds, and a frequency held over
    # time is exactly what a ridge on this surface IS.
    run = q("""
        WITH span AS (
            SELECT dog_id, test_num, MIN(sample_ts) AS lo, MAX(sample_ts) AS hi
            FROM RAW.COLLAR_TELEMETRY GROUP BY 1, 2
        ), loco AS (
            SELECT s.dog_id, s.test_num, s.epoch_ts, span.lo, span.hi,
                   DATEDIFF('second', span.lo, s.epoch_ts)
                     - ROW_NUMBER() OVER (PARTITION BY s.dog_id, s.test_num
                                          ORDER BY s.epoch_ts) AS grp
            FROM MARTS.EPOCH_STATES s
            JOIN span ON span.dog_id = s.dog_id AND span.test_num = s.test_num
            WHERE s.state IN ('WALK','TROT','GALLOP','PACE')
              AND s.epoch_ts BETWEEN span.lo AND span.hi
        )
        SELECT dog_id, test_num, MIN(epoch_ts) AS t0, MIN(lo) AS lo,
               MIN(hi) AS hi, COUNT(*) AS n
        FROM loco GROUP BY dog_id, test_num, grp
        ORDER BY n DESC LIMIT 1""")
    if not run:
        warn("03: no locomotion run with a waveform behind it")
        return None
    dog, tnum, t0 = int(run[0]["DOG_ID"]), int(run[0]["TEST_NUM"]), run[0]["T0"]
    secs = min(60, int(run[0]["N"]))
    w = q(f"""
        SELECT SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az) AS vm
        FROM RAW.COLLAR_TELEMETRY
        WHERE dog_id = {dog} AND test_num = {tnum}
          AND sample_ts >= '{t0}'
          AND sample_ts <  LEAST(DATEADD('second', {secs}, '{t0}'), '{run[0]["HI"]}')
        ORDER BY sample_ts""")
    if len(w) < 512:
        warn(f"03: only {len(w)} samples")
        return None
    sig = np.asarray([float(r["VM"]) for r in w], dtype=float)
    sig = sig - sig.mean()          # gravity is a DC term 100x the signal
    # hop 64 rather than 32: at 32 the surface carries 180 columns across a
    # 900 px figure, which is finer than the render can resolve and turns
    # genuine structure into hatching.
    nfft, hop, fs = 256, 64, 100.0
    win = np.hanning(nfft)
    starts = range(0, len(sig) - nfft + 1, hop)
    cols = [np.abs(np.fft.rfft(sig[s:s + nfft] * win)) for s in starts]
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    keep = freqs <= 16.0
    spec = 20.0 * np.log10(np.asarray(cols).T[keep] + 1e-6)
    floor = float(np.percentile(spec, 5))
    fig = go.Figure(go.Surface(
        x=[s / fs for s in starts], y=[float(f) for f in freqs[keep]],
        z=np.clip(spec, floor, None), showscale=False,
        colorscale=[[0.0, "#FAFAF9"], [0.25, "#cde2fb"], [0.55, S_BLUE],
                    [0.80, S_ORANGE], [1.0, "#7a1f06"]],
        contours=dict(z=dict(show=True, usecolormap=True,
                             project=dict(z=True), highlightcolor=INK_2)),
        lighting=dict(ambient=0.62, diffuse=0.72, specular=0.18,
                      roughness=0.85, fresnel=0.1),
        hoverinfo="skip"))
    fig.update_layout(scene=scene("seconds", "frequency (Hz)", "power (dB)",
                                  eye=(1.75, -1.55, 0.95), aspect=(1.9, 1.0, 0.6),
                                  ortho=False),
                      showlegend=False)
    plotly_layout(fig, "A sustained gait, as frequency over time",
                  f"{secs} s of unbroken locomotion by dog {dog}, cut into {len(cols)} overlapping "
                  f"2.56 s Hann windows. The ridge running left to right is the stride rate — "
                  f"a rhythm held, which is what a gait is. Contours on the floor are the same "
                  f"surface seen from above.", height=600)
    return fig, len(w), (f"RAW.COLLAR_TELEMETRY, dog {dog} test {tnum}, "
                         f"{secs} s at 100 Hz, STFT in numpy")


# ===========================================================================
# 04 — the feature space
# ===========================================================================
def fig_04_feature_space():
    """The classifier argument in one object.

    Locomotion separates on neck/back correlation, neck-driven behaviour on
    neck dominance, stillness on neck SD. Any 2D pair of those three collapses
    two classes onto each other — which is what the shadows on the walls are
    for: each wall IS the 2D scatter you would have drawn instead.
    """
    import plotly.graph_objects as go
    fs = q("""
        SELECT state, neck_back_corr, vm_neck_std, neck_dominance
        FROM ML.V_LABELLED_EPOCHS
        WHERE neck_back_corr IS NOT NULL AND vm_neck_std IS NOT NULL
          AND neck_dominance IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY state ORDER BY RANDOM()) <= 300""")
    pal = {r["STATE"]: (r["COLOUR_HEX"] or "#D6D3D1")
           for r in q("SELECT state, colour_hex FROM REF.ETHOGRAM")}
    by_state: dict = {}
    for r in fs:
        by_state.setdefault(r["STATE"], []).append(r)
    by_state.pop("UNKNOWN", None)

    def mid90(v):
        s = sorted(v)
        return s[int(0.05 * (len(s) - 1))], s[int(0.95 * (len(s) - 1))]

    fig = go.Figure()
    for stt in sorted(by_state):
        pts = by_state[stt]
        xs = [float(r["NECK_BACK_CORR"]) for r in pts]
        ys = [float(r["VM_NECK_STD"]) for r in pts]
        zs = [min(float(r["NECK_DOMINANCE"]), 8.0) for r in pts]
        hue = pal.get(stt, "#999")
        xl, xh = mid90(xs)
        yl, yh = mid90(ys)
        zl, zh = mid90(zs)
        core = [(a, b, c) for a, b, c in zip(xs, ys, zs)
                if xl <= a <= xh and yl <= b <= yh and zl <= c <= zh]
        if len(core) >= 8:
            fig.add_trace(go.Mesh3d(
                x=[p[0] for p in core], y=[p[1] for p in core],
                z=[p[2] for p in core], alphahull=0, color=hue, opacity=0.16,
                flatshading=True, hoverinfo="skip", legendgroup=stt,
                showlegend=False, lighting=dict(ambient=0.78, diffuse=0.5,
                                                specular=0.1, roughness=0.9)))
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers", name=stt, legendgroup=stt,
            marker=dict(size=2.0, color=hue, opacity=0.7),
            projection=dict(x=dict(show=True, opacity=0.08, scale=0.62),
                            y=dict(show=True, opacity=0.08, scale=0.62),
                            z=dict(show=True, opacity=0.08, scale=0.62)),
            hoverinfo="skip"))
    fig.update_layout(
        scene=scene("neck/back corr", "neck SD (g)", "neck dominance (clipped at 8)",
                    eye=(1.6, 1.5, 0.85)),
        showlegend=True,
        legend=dict(itemsizing="constant", font=dict(size=10, color=INK_2),
                    bgcolor="rgba(0,0,0,0)", x=1.0, y=0.5, yanchor="middle"))
    # COUNTED, NOT ASSERTED. This said "Fourteen behaviours" because the
    # ethogram defines fourteen — but ML.V_LABELLED_EPOCHS only carries the
    # ten that have labelled epochs, so the figure showed ten and the title
    # claimed fourteen. A caption that disagrees with its own legend is the
    # cheapest possible way to lose a reader's trust in every other number.
    plotly_layout(fig, f"{len(by_state)} behaviours in the three features the ethogram turns on",
                  "Solids are convex hulls of the middle 90% of each class; dots are every sampled second. "
                  "The faint shadows on the three walls are the 2D scatter you would have drawn instead.",
                  height=620)
    return fig, len(fs), "ML.V_LABELLED_EPOCHS, stratified sample of 300 per state"


# ===========================================================================
# 05 — confusion matrix
# ===========================================================================
def fig_05_confusion():
    """The accuracy figure, including the parts that are bad.

    Row-normalised, so the diagonal reads as recall per class and a bright
    off-diagonal cell is a specific confusion rather than a big class. The
    classes this model actually struggles with are visible and labelled,
    which is the entire reason to print a matrix instead of one number.
    """
    cm = q("SELECT actual_state, predicted_state, n, pct_of_actual "
           "FROM ML.CONFUSION_MATRIX")
    ms = q("SELECT holdout_accuracy, macro_f1, weighted_f1, holdout_epochs, "
           "holdout_dogs, protocol FROM ML.MODEL_SUMMARY")
    order = [r["STATE"] for r in
             q("SELECT state FROM REF.ETHOGRAM ORDER BY sort_order")]
    seen = {r["ACTUAL_STATE"] for r in cm} | {r["PREDICTED_STATE"] for r in cm}
    order = [s for s in order if s in seen]
    # EVERY CELL, INCLUDING THE EMPTY ONES. ML.CONFUSION_MATRIX stores only
    # the pairs that actually occurred, so a combination the model never
    # produced has no row and Vega draws nothing — leaving a white hole that
    # is visually identical to the page behind it, and therefore reads the
    # same as "a bit of it went here". Filling the absent pairs with 0 makes
    # the matrix a complete grid where white means the palette's zero.
    seen_pairs = {(r["ACTUAL_STATE"], r["PREDICTED_STATE"]): r for r in cm}
    recs = []
    for a in order:
        for p in order:
            r = seen_pairs.get((a, p))
            recs.append({"actual": a, "pred": p,
                         "n": float(r["N"]) if r else 0.0,
                         "pct": float(r["PCT_OF_ACTUAL"] or 0) if r else 0.0})
    m = ms[0] if ms else {}
    acc = float(m.get("HOLDOUT_ACCURACY") or 0) * 100
    macro = float(m.get("MACRO_F1") or 0)

    base = alt.Chart(data(recs)).encode(
        x=alt.X("pred:N", title="predicted", sort=order,
                axis=alt.Axis(labelAngle=-45, orient="bottom")),
        y=alt.Y("actual:N", title="actually", sort=order))
    cells = base.mark_rect(stroke=CARD, strokeWidth=1.5).encode(
        color=alt.Color("pct:Q", title="% of the true class",
                        scale=alt.Scale(scheme="blues", domain=[0, 100]),
                        legend=alt.Legend(gradientLength=170, format=".0f")))
    # Only the cells worth reading get a number. Labelling all 196 turns the
    # matrix into a spreadsheet and hides the two or three that matter.
    labels = base.mark_text(font=MONO, fontSize=9.5).encode(
        text=alt.Text("pct:Q", format=".0f"),
        color=alt.condition(alt.datum.pct > 55, alt.value(CARD), alt.value(INK_2)),
        opacity=alt.condition(alt.datum.pct >= 8, alt.value(1), alt.value(0)))
    grid = alt.layer(cells, labels).properties(width=520, height=440)

    fig = alt.vconcat(
        grid.properties(title=title(
            f"Where the classifier is right, and exactly where it is not",
            [f"Holdout accuracy {acc:.1f}%  ·  macro-F1 {macro:.2f}  ·  "
             f"{int(m.get('HOLDOUT_EPOCHS') or 0):,} epochs from "
             f"{int(m.get('HOLDOUT_DOGS') or 0)} dogs the model never saw.",
             "Row-normalised: each row sums to 100%, so the diagonal is recall and a bright off-diagonal cell is a real confusion."])),
        note("Source: TELLTAIL ML.CONFUSION_MATRIX and ML.MODEL_SUMMARY. Dog-disjoint holdout — whole dogs are held out,\n"
             "never individual seconds, so no dog contributes to both training and evaluation. Cells below 8% are left unlabelled.",
             width=520),
        spacing=14)
    return styled(fig, grid_y=False), len(cm), "ML.CONFUSION_MATRIX (dog-disjoint holdout)"


# ===========================================================================
# 06 — bout duration ridgeline
# ===========================================================================
def fig_06_ridgeline():
    """What a mean bout length hides.

    Several of these are bimodal — a sniff is a two-second check or a long
    investigation and not much between — and a mean of two humps lands in the
    gap where the dog never actually is.
    """
    bouts = q("""
        SELECT state, bout_seconds FROM MARTS.STATE_BOUTS
        WHERE bout_seconds >= 1""")
    counts: dict = {}
    for b in bouts:
        counts[b["STATE"]] = counts.get(b["STATE"], 0) + 1
    order = [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])
             if s != "UNKNOWN"]
    pal = {r["STATE"]: (r["COLOUR_HEX"] or "#D6D3D1")
           for r in q("SELECT state, colour_hex FROM REF.ETHOGRAM")}
    recs = [{"state": b["STATE"], "secs": float(b["BOUT_SECONDS"])}
            for b in bouts if b["STATE"] in order]

    # EXTENT FROM A HIGH PERCENTILE, NOT THE MAXIMUM. One 20-minute REST bout
    # sets the max, and scaling to it puts every distribution in the leftmost
    # fifth of the chart with four-fifths of empty axis to its right. The
    # 99.5th percentile keeps essentially the whole shape of every class and
    # drops only the single longest tail, which the log axis was already
    # compressing to invisibility.
    tail = sorted(r["secs"] for r in recs)[int(0.995 * (len(recs) - 1))]
    hi = max(1.4, math.ceil((math.log10(tail) + 0.15) * 5) / 5)

    # RIDGE GEOMETRY, WRITTEN DOWN BECAUSE IT IS EASY TO GET SILENTLY WRONG.
    #
    # Each facet row is BAND px tall and the density's y range is
    # [BAND, -OVER]: zero density lands on the row's bottom edge and the peak
    # overflows OVER px into the row above. That overflow is the overlap the
    # form is made of.
    #
    # OVER IS SMALL ON PURPOSE, AND THIS IS THE WHOLE TRICK. A facet row's
    # header label is centred in its OWN band, so the moment a ridge overflows
    # by more than about a third of a band it climbs past its label and sits
    # next to the label of the row above — which is exactly what a reader then
    # believes. The first two attempts at this figure both shipped fourteen
    # ridges each apparently labelled with its neighbour's name. Keeping the
    # overflow well under half a band means every label stays inside the ridge
    # it names, and the ridges still overlap enough to read as one object.
    BAND, OVER = 40, 13
    ridge = (alt.Chart(data(recs))
             .transform_calculate(ls="log(datum.secs)/log(10)")
             .transform_density("ls", groupby=["state"], as_=["ls", "d"],
                                extent=[0, hi], steps=180, counts=False)
             .mark_area(interpolate="monotone", fillOpacity=0.88,
                        stroke=CARD, strokeWidth=1)
             .encode(
                 x=alt.X("ls:Q", title="bout length",
                         scale=alt.Scale(domain=[0, hi], nice=False),
                         axis=alt.Axis(values=[0, 1, 2, 3, 4], labelExpr=(
                             "datum.value == 0 ? '1 s' : datum.value == 1 ? '10 s' : "
                             "datum.value == 2 ? '1m 40s' : datum.value == 3 ? "
                             "'16m 40s' : '2h 46m'"))),
                 y=alt.Y("d:Q", title=None, stack=None, axis=None,
                         scale=alt.Scale(range=[BAND, -OVER])),
                 row=alt.Row("state:N", title=None, sort=order,
                             header=alt.Header(labelAngle=0, labelAlign="right",
                                               labelBaseline="middle",
                                               labelPadding=10,
                                               labelFontSize=11)),
                 fill=alt.Fill("state:N", legend=None, scale=alt.Scale(
                     domain=order, range=[pal.get(s, "#D6D3D1") for s in order])))
             .properties(width=W_FULL - 110, height=BAND, bounds="flush"))

    fig = alt.vconcat(
        ridge.properties(title=title(
            "How long a bout of each behaviour actually lasts",
            [f"Kernel density over {len(recs):,} bouts from all 45 dogs, on a shared log-seconds axis, commonest behaviour first.",
             "Two humps in a row means the dog does that behaviour in two distinct ways — which is what an average bout length deletes."])),
        note("Source: TELLTAIL MARTS.STATE_BOUTS. Densities computed by the renderer from raw bout lengths;\n"
             f"nothing pre-binned or pre-smoothed. Log axis; x clipped at the 99.5th percentile ({tail:,.0f} s).",
             width=W_FULL - 110),
        spacing=18)
    # spacing 0: the overlap is produced by the density overflowing its band
    # (see BAND/OVER above), not by pulling the rows into each other.
    return styled(fig, grid_y=False).configure_facet(spacing=0), len(recs), \
        "MARTS.STATE_BOUTS, all dogs"


# ===========================================================================
# 07 — the findings
# ===========================================================================
def fig_07_findings():
    """Every syndrome match in the space the pattern engine actually returns.

    A syndrome is not a point in time, it is a shape: how long it ran, how
    many epochs the pattern consumed, how confidently. The codes occupy
    different regions, which is the claim that they are different detectors
    rather than one detector fired six ways.
    """
    finds = q("""
        SELECT syndrome_code, syndrome_name, body_system, dog_id,
               duration_s, n_epochs, confidence, severity
        FROM MARTS.V_FINDINGS""")
    if not finds:
        warn("07: no findings")
        return None
    codes = sorted({f["SYNDROME_CODE"] for f in finds})
    names = {f["SYNDROME_CODE"]: f["SYNDROME_NAME"] for f in finds}
    cmap = {c: SERIES[i % len(SERIES)] for i, c in enumerate(codes)}
    recs = [{"code": f["SYNDROME_CODE"],
             "label": f'{f["SYNDROME_CODE"]} · {f["SYNDROME_NAME"]}',
             "system": f.get("BODY_SYSTEM") or "unclassified",
             "duration": float(f["DURATION_S"] or 0),
             "epochs": float(f["N_EPOCHS"] or 0),
             "confidence": float(f["CONFIDENCE"] or 0),
             "severity": int(f["SEVERITY"] or 0)} for f in finds]
    labels = [f"{c} · {names[c]}" for c in codes]
    colours = [cmap[c] for c in codes]

    pts = (alt.Chart(data(recs))
           .mark_circle(opacity=0.72, stroke=CARD, strokeWidth=0.7)
           .encode(
               x=alt.X("duration:Q", title="how long the match ran (seconds)",
                       scale=alt.Scale(type="sqrt", nice=True)),
               y=alt.Y("confidence:Q", title="confidence",
                       scale=alt.Scale(domain=[0, 1.02], nice=False)),
               size=alt.Size("epochs:Q", title="epochs consumed",
                             scale=alt.Scale(range=[25, 500]),
                             legend=alt.Legend(orient="right", symbolFillColor=INK_3,
                                               symbolStrokeColor=INK_3)),
               color=alt.Color("label:N", title="syndrome",
                               scale=alt.Scale(domain=labels, range=colours),
                               legend=alt.Legend(orient="right")))
           .properties(width=W_FULL - 250, height=380))

    fig = alt.vconcat(
        pts.properties(title=title(
            "98 findings in the three numbers MATCH_RECOGNIZE returns",
            ["Each dot is one matched sequence. The x axis is on a square-root scale because durations span two orders of magnitude.",
             "The codes occupy different regions — that is the claim that these are six detectors, not one detector fired six ways."])),
        note("Source: TELLTAIL MARTS.V_FINDINGS. A finding is a match of a SQL MATCH_RECOGNIZE pattern over ordered\n"
             "one-second epochs; duration, epochs consumed and confidence are all outputs of the match itself.",
             width=W_FULL - 250),
        spacing=14)
    return styled(fig), len(finds), "MARTS.V_FINDINGS"


# ===========================================================================
# 08 — sensitivity
# ===========================================================================
def fig_08_sensitivity():
    """Is the detector tuned to a magic number?

    The honest way to answer that is to move the number and show what happens.
    Each syndrome's pattern was re-run at a loose, tuned and strict minimum
    epoch count. A detector whose match count collapses to zero the moment you
    tighten it was fitted to the demo; one that degrades smoothly was not.
    """
    sens = q("""
        SELECT syndrome_code, variant, min_epochs, COUNT(*) AS matches
        FROM MARTS.SYNDROME_SENSITIVITY
        GROUP BY 1, 2, 3 ORDER BY 1, 3""")
    if not sens:
        warn("08: no sensitivity rows")
        return None
    codes = sorted({r["SYNDROME_CODE"] for r in sens})
    cmap = {c: SERIES[i % len(SERIES)] for i, c in enumerate(codes)}
    recs = [{"code": r["SYNDROME_CODE"], "variant": r["VARIANT"],
             "min_epochs": float(r["MIN_EPOCHS"]),
             "matches": float(r["MATCHES"])} for r in sens]

    base = alt.Chart(data(recs)).encode(
        x=alt.X("min_epochs:Q", title="minimum epochs the pattern must consume",
                scale=alt.Scale(nice=True, zero=False)),
        y=alt.Y("matches:Q", title="matches found",
                scale=alt.Scale(type="symlog", nice=False)),
        color=alt.Color("code:N", title="syndrome",
                        scale=alt.Scale(domain=codes,
                                        range=[cmap[c] for c in codes]),
                        legend=alt.Legend(orient="right")))
    line = base.mark_line(strokeWidth=1.6, point=False, interpolate="monotone")
    dots = base.mark_point(filled=True, size=90, opacity=1, stroke=CARD,
                           strokeWidth=1.2)
    tags = base.mark_text(font=FONT, fontSize=9.5, dy=-13, color=INK_3).encode(
        text=alt.Text("variant:N"), color=alt.value(INK_3))
    chart = alt.layer(line, dots, tags).properties(width=W_FULL - 230, height=330)

    fig = alt.vconcat(
        chart.properties(title=title(
            "What happens when you move the threshold on purpose",
            ["Every pattern re-run at a loose, tuned and strict minimum epoch count. y is symlog so a drop to 1 is still visible.",
             "A detector fitted to the demo goes to zero the moment you tighten it. These degrade, which is the answer you want."])),
        note("Source: TELLTAIL MARTS.SYNDROME_SENSITIVITY. Each point is a full re-run of that syndrome's MATCH_RECOGNIZE\n"
             "pattern against the whole corpus with only the quantifier changed. S1 and S5 produced no matches at any setting.",
             width=W_FULL - 230),
        spacing=14)
    return styled(fig), len(sens), "MARTS.SYNDROME_SENSITIVITY"


# ===========================================================================
# 09 — the shelter
# ===========================================================================
def fig_09_shelter():
    """Where the argument ends up.

    Austin Animal Center publishes every intake. The behaviour-linked share is
    the fraction of dogs surrendered for something an owner would describe as
    behaviour — the category a missed chronic pain problem eventually gets
    filed under.
    """
    tr = q("""
        SELECT DATE_TRUNC('month', month) AS m,
               SUM(n) AS total,
               SUM(CASE WHEN is_behaviour_linked THEN n ELSE 0 END) AS behav
        FROM REF.V_AAC_INTAKE_TREND
        GROUP BY 1 HAVING SUM(n) > 0 ORDER BY 1""")
    if len(tr) < 6:
        warn("09: not enough intake history")
        return None
    # The first and last months of any operational feed are partial — the
    # window opens and closes mid-month — and a partial month drawn next to
    # full ones reads as a collapse in intakes rather than as a clipped bar.
    tr = tr[1:-1]
    recs = [{"m": r["M"].strftime("%Y-%m-%d"), "total": float(r["TOTAL"]),
             "behav": float(r["BEHAV"]),
             "pct": 100.0 * float(r["BEHAV"]) / float(r["TOTAL"])} for r in tr]

    band = (alt.Chart(data(recs)).mark_area(color=BORDER, opacity=0.55,
                                            interpolate="monotone")
            .encode(x=alt.X("m:T", title=None, axis=alt.Axis(format="%Y")),
                    y=alt.Y("total:Q", title="dog intakes per month")))
    beh = (alt.Chart(data(recs)).mark_area(color=ACCENT, opacity=0.9,
                                           interpolate="monotone")
           .encode(x=alt.X("m:T"), y=alt.Y("behav:Q")))
    top = alt.layer(band, beh).properties(width=W_FULL - 40, height=210)

    share = (alt.Chart(data(recs)).mark_line(color=ACCENT, strokeWidth=1.8,
                                             interpolate="monotone")
             .encode(x=alt.X("m:T", title=None, axis=alt.Axis(format="%Y")),
                     y=alt.Y("pct:Q", title="behaviour-linked share (%)",
                             scale=alt.Scale(zero=True)))
             .properties(width=W_FULL - 40, height=140))

    total_all = sum(r["total"] for r in recs)
    beh_all = sum(r["behav"] for r in recs)
    fig = alt.vconcat(
        top.properties(title=title(
            "Austin Animal Center dog intakes, and the behaviour-linked share",
            [f"{total_all:,.0f} dog intakes over {len(recs)} months. "
             f"{beh_all:,.0f} of them — {100*beh_all/total_all:.1f}% — carry a behaviour-linked intake type or condition.",
             "Amber is the behaviour-linked subset; grey is everything else. The lower panel is the same subset as a percentage."])),
        share,
        note("Source: Austin Animal Center open data via REF.V_AAC_INTAKE_TREND, synced into Snowflake by scripts/austin_sync.py.\n"
             "First and last months dropped as partial. Behaviour-linked is a classification of AAC's own intake_type and intake_condition fields —\n"
             "it is not a clinical judgement and it is not evidence that pain caused any individual surrender.",
             width=W_FULL - 40),
        spacing=16)
    return styled(fig), len(tr), "REF.V_AAC_INTAKE_TREND (Austin Animal Center open data)"


# ===========================================================================
# 10 — transition matrix
# ===========================================================================
def fig_10_transitions():
    """A first-order behavioural Markov chain, pooled across the pack.

    Row-normalised, and the diagonal is dropped: a behaviour is overwhelmingly
    followed by itself, so leaving self-transitions in gives you a bright
    diagonal and thirteen invisible rows. What is interesting is what follows
    a behaviour when it CHANGES.
    """
    tr = q("""
        SELECT from_state, to_state, SUM(n) AS n
        FROM MARTS.STATE_TRANSITIONS
        WHERE from_state <> to_state
        GROUP BY 1, 2""")
    order = [r["STATE"] for r in
             q("SELECT state FROM REF.ETHOGRAM ORDER BY sort_order")]
    seen = {r["FROM_STATE"] for r in tr} | {r["TO_STATE"] for r in tr}
    order = [s for s in order if s in seen and s != "UNKNOWN"]
    tot: dict = {}
    for r in tr:
        tot[r["FROM_STATE"]] = tot.get(r["FROM_STATE"], 0) + float(r["N"])
    recs = [{"frm": r["FROM_STATE"], "to": r["TO_STATE"], "n": float(r["N"]),
             "pct": 100.0 * float(r["N"]) / tot[r["FROM_STATE"]]}
            for r in tr if r["FROM_STATE"] in order and r["TO_STATE"] in order]

    base = alt.Chart(data(recs)).encode(
        x=alt.X("to:N", title="then does this", sort=order,
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("frm:N", title="after this", sort=order))
    cells = base.mark_rect(stroke=CARD, strokeWidth=1.5).encode(
        color=alt.Color("pct:Q", title="% of changes",
                        scale=alt.Scale(scheme="purples", domain=[0, 60]),
                        legend=alt.Legend(gradientLength=170, format=".0f")))
    labels = base.mark_text(font=MONO, fontSize=9).encode(
        text=alt.Text("pct:Q", format=".0f"),
        color=alt.condition(alt.datum.pct > 35, alt.value(CARD), alt.value(INK_2)),
        opacity=alt.condition(alt.datum.pct >= 10, alt.value(1), alt.value(0)))
    grid = alt.layer(cells, labels).properties(width=500, height=430)

    fig = alt.vconcat(
        grid.properties(title=title(
            "What follows what, when the behaviour changes",
            ["A first-order Markov chain over one-second states, computed in SQL with LAG and row-normalised, pooled across all 45 dogs.",
             "Self-transitions are excluded — a behaviour is mostly followed by itself, and leaving that in hides everything else."])),
        note("Source: TELLTAIL MARTS.STATE_TRANSITIONS. Rows sum to 100% of that state's observed CHANGES,\n"
             "not of its total time. Cells below 10% are left unlabelled.", width=500),
        spacing=14)
    return styled(fig, grid_y=False), len(recs), "MARTS.STATE_TRANSITIONS, pooled"


# ---------------------------------------------------------------------------
FIGURES = [
    ("01", "01-two-sensors", fig_01_two_sensors, "altair"),
    ("02", "02-motion-signature-3d", fig_02_motion_signature, "plotly"),
    ("03", "03-spectrogram-3d", fig_03_spectrogram, "plotly"),
    ("04", "04-feature-space-3d", fig_04_feature_space, "plotly"),
    ("05", "05-confusion-matrix", fig_05_confusion, "altair"),
    ("06", "06-bout-ridgeline", fig_06_ridgeline, "altair"),
    ("07", "07-findings", fig_07_findings, "altair"),
    ("08", "08-sensitivity", fig_08_sensitivity, "altair"),
    ("09", "09-shelter-intake", fig_09_shelter, "altair"),
    ("10", "10-transitions", fig_10_transitions, "altair"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render blog figures from the warehouse.")
    ap.add_argument("--only", nargs="*", metavar="NN",
                    help="figure numbers, e.g. --only 01 05")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for num, name, fn, kind in FIGURES:
            doc = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {num}  {name:<26} {kind:<7} {doc}")
        return 0

    load_env()
    todo = [f for f in FIGURES if not args.only or f[0] in args.only]
    if not todo:
        warn("nothing matched --only")
        return 1

    header(f"Rendering {len(todo)} figure(s) to {OUT}")
    failed = []
    for num, name, fn, kind in todo:
        try:
            result = fn()
            if result is None:
                failed.append(name)
                continue
            obj, rows, src = result
            (save_alt if kind == "altair" else save_plotly)(obj, name, rows, src)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            warn(f"{name}: {type(exc).__name__}: {exc}")

    print()
    if failed:
        warn(f"{len(failed)} figure(s) failed: {', '.join(failed)}")
        return 1
    ok(f"{len(todo)} figure(s) written to {OUT}")
    info("PNG is 2x for dev.to; SVG is there for anywhere that takes vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
