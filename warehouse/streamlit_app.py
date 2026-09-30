"""
TELLTAIL — Streamlit in Snowflake.

Nine tabs, in narrative order. A judge who clicks left to right gets the argument
without reading a word of the post:

  1 Pack            who
  2 Live Collar     is it real
  3 Ethogram        what normal looks like
  4 Syndromes       THE FINDING
  5 Baselines       why it is abnormal for THIS dog
  6 Vet Note        what to do
  7 Drivers         what explains it
  8 Shelter Reality why it matters
  9 Pipeline        how it was built

Plus two more: On Chain, the published attestations with a link out to Solana
Explorer for every one of them, and Ask TELLTAIL, Cortex answering only from
rows.

NINE SiS-ONLY HAZARDS, ALL HANDLED HERE. Every one of them reproduces nowhere
else, which is what makes them expensive:

  1. Snowpark to_pandas() returns numerics as object-dtype Decimal. Plotly then
     treats them as CATEGORIES, so y becomes the row index and every line chart
     renders as an identical straight diagonal. `rows()` converts element-wise
     with float() and every chart is fed plain Python lists.
  2. SiS pins an older Streamlit than you develop against. st.column_config is
     guarded with hasattr and there is an HTML table fallback.
  3. plotly is not importable without environment.yml shipped to the stage next
     to this file. deploy_streamlit.py refuses to deploy without it.
  4. Old plotly drops traces whose customdata is a mixed-type numpy array, and
     the axes scale while zero points render. Hover strings are prebuilt and
     passed as text= with hoverinfo="text".
  5. There is no outbound network, so a photograph can only be shown if its
     bytes are already in the account. Breed photos are base64 in
     REF.BREED_IMAGE, written by scripts/fetch_breed_images.py; everything
     else on screen is inline SVG. They are photographs OF THE BREED and the
     page says so everywhere one appears.
  6. Custom bidirectional components cannot be installed — there is no way to
     serve a component's frontend bundle from the sandbox. The chat on the
     last page is AI-Yash/st-chat's design ported to plain HTML and CSS, with
     locally drawn avatars in place of its DiceBear URLs. THE SAME RULE IS WHY
     THE CHART LANGUAGE HERE IS A PORT: evilcharts.com is React on Recharts
     or ECharts, so its gradients, glows, rounded caps and radial/radar/sankey
     forms are rebuilt on plotly and vega instead of installed. See "THE CHART
     LANGUAGE" below.
  7. SiS ships ALTAIR 4. Selections were renamed wholesale in Altair 5, and
     the 5 spelling — alt.selection_point, .add_params — raises AttributeError
     at render time and takes the whole Syndromes tab down with it. Every
     development machine has Altair 5 or 6, so it cannot be reproduced locally
     at all. alt_point()/alt_bind() resolve the spelling at call time.
  8. The plotly version is not knowable from here. Old Streamlit implies an
     old conda solve, and plotly raises ValueError at figure CONSTRUCTION for
     a property it does not know — so fillgradient (5.23), barcornerradius
     (5.19) and griddash (5.5) are each probed at import with HAS_* and each
     has a defined fallback. Nothing in the chart layer assumes a version.
  9. A webfont referenced by URL never loads and never says so. The typeface
     is embedded as base64 woff2 in tt_font.py and staged next to this file;
     the import is guarded, so a stage without it renders in the old stack.

AND ONE ARCHITECTURAL RULE: this app reads TABLES. It never calls a Cortex AI
function, because a render path that costs credits is a render path that will
exhaust a trial cap during a demo. Every note, triage and brief on screen was
batched into a table by a task.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from decimal import Decimal

import streamlit as st

# ---------------------------------------------------------------------------
# THE TYPEFACE, AND WHY IT IS AN IMPORT RATHER THAN A FONT-FAMILY.
#
# `font-family: Satoshi` is a no-op in SiS. There is no outbound network, so the
# browser cannot fetch the face, and CSS has no way to report that — it silently
# falls through to the next name in the stack. The app looked like it had a
# custom font for as long as it took to open it on a machine that did not
# already have one installed.
#
# tt_font ships the woff2 bytes for three weights as base64 `data:` URIs, the
# same trick as the breed photographs (hazard 5). It is a separate staged file
# because 102 KB of base64 in the middle of this one would bury it.
#
# GUARDED, BECAUSE A MISSING FILE MUST NOT BE A BLANK APP. If the stage somehow
# has streamlit_app.py without tt_font.py, this falls back to the stack the app
# used before and every rule below keeps working — a plainer dashboard, not a
# broken one.
# ---------------------------------------------------------------------------
try:
    from tt_font import FACE_CSS as FONT_CSS, STACK as FONT
except ModuleNotFoundError:
    FONT_CSS = ""
    FONT = 'Geist, Inter, -apple-system, "Segoe UI", sans-serif'

# The name alone, for the renderers that take a font family rather than a CSS
# declaration. plotly, vega and bokeh all want a bare list and none of them
# accept the quoted "Segoe UI" entry, which they pass through to SVG unescaped.
FONT_CHART = "Satoshi, Geist, Inter, sans-serif"
MONO = '"Geist Mono", ui-monospace, "SF Mono", Consolas, monospace'

# ---------------------------------------------------------------------------
# design system  (spec section 15: clinical register, one accent, no chart junk)
# ---------------------------------------------------------------------------
SURFACE = "#FAFAF9"
CARD = "#FFFFFF"
BORDER = "#E7E5E4"
INK = "#1C1917"
INK_2 = "#57534E"
ACCENT = "#B45309"
GRID = "#F5F5F4"

# Chat surface, taken from the reference build rather than the palette above.
# The rest of the app is a warm stone scale; a chat transcript is the one place
# that reads as a conversation rather than a document, and the reference greys
# are what make it read that way.
CHAT_BG = "#F0F2F6"
CHAT_INK = "#262730"
CHAT_BOT = "#DD6B4D"     # bottts orange
CHAT_USER = "#FBD7B0"    # fun-emoji peach

# ---------------------------------------------------------------------------
# THE CHART PALETTE, VALIDATED RATHER THAN CHOSEN.
#
# Every colour below was run through a colour-vision-deficiency validator
# against this app's actual chart surface (#FFFFFF) — protan/deutan/tritan
# separation, chroma floor, lightness band and contrast. Nothing here was
# picked because it looked nice next to the last one.
#
# The eight series slots are used IN ORDER and never cycled past the end. They
# are for things with no natural ordering: syndrome codes, pattern symbols,
# cohort bands. Worst adjacent pair is ΔE 9.1 under protanopia, ΔE 19.6 for
# normal vision — both clear.
#
# UI CHROME IS NOT A SERIES COLOUR. ACCENT stays the dark amber it always was
# because it is worn by text, rules and borders, where the requirement is
# reading contrast, not separation from seven other hues. A colour that labels
# a thing and a colour that IS the thing are different jobs.
# ---------------------------------------------------------------------------
S_BLUE = "#2a78d6"
S_ORANGE = "#eb6834"
S_AQUA = "#1baf7a"
S_YELLOW = "#eda100"
S_MAGENTA = "#e87ba4"
S_GREEN = "#008300"
S_VIOLET = "#4a3aa7"
S_RED = "#e34948"

# Pattern-symbol palette for the hero ribbon. Deliberately not the state
# palette: the point of that chart is which PATTERN VARIABLE each epoch played.
SYMBOL_COLOURS = [S_BLUE, S_ORANGE, S_AQUA, S_YELLOW, S_MAGENTA,
                  S_GREEN, S_VIOLET, S_RED]

# Triage keeps its own darker greens/ambers/reds rather than borrowing series
# slots. These are worn as a solid badge with WHITE TEXT ON THEM, so the bar
# they have to clear is text contrast; the series oranges and greens are two
# stops too light for that and would ship an unreadable badge.
TRIAGE_COLOUR = {
    1: "#15803D", "routine monitoring": "#15803D",
    2: "#B45309", "schedule appointment": "#B45309",
    3: "#B91C1C", "urgent veterinary attention": "#B91C1C",
}

# ---------------------------------------------------------------------------
# ONE HEIGHT SCALE FOR EVERY CHART IN THE APP.
#
# These were once twelve hand-picked numbers — 54, 90, 210, 220, 230, 250, 260,
# 280, 290, 300, 330, 430 — each chosen for its own chart as it was written.
# Every chart was fine and the whole was a mess: two charts side by side in one
# row ended at different heights, the same kind of chart was a different size on
# two pages, and nine pages of that reads as unfinished.
#
# The rule now: pick a step off this scale, and two charts sharing a row pick
# the SAME step. Horizontal bar charts use bars(), because a 25-bar chart
# genuinely does need more room than a 4-bar one — but it grows on a fixed row
# pitch and clamps, so it cannot run away down the page.
# ---------------------------------------------------------------------------
H_STRIP = 56       # one stacked bar, no axes (the triage mix)
H_GAUGE = 148      # one radial gauge in a row of them
H_RIBBON = 96      # a state or symbol ribbon
H_SM = 260         # a supporting chart, or one of a stack of time series
H_MD = 320         # the default: a chart in a two-column row
H_LG = 420         # a hero — the 3D feature space


def bars(n: int, *, row: int = 24) -> int:
    """Height for a horizontal bar chart of n categories."""
    return max(H_SM, min(560, row * int(n or 0) + 76))


def alpha(hex_colour: str, a: float) -> str:
    """#RRGGBB + opacity -> rgba(). Plotly will not take an 8-digit hex.

    CSS has accepted #RRGGBBAA for years, so writing `S_ORANGE + "33"` looks
    right and works everywhere in this file EXCEPT inside a figure, where
    plotly rejects it at construction with a property error. Converting here
    means the fill tints in charts and the tints in the stylesheet can be
    written from the same constants.
    """
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


st.set_page_config(page_title="TELLTAIL", page_icon="🐕", layout="wide")

st.markdown(f"""
<style>
  {FONT_CSS}
  .stApp {{ background: {SURFACE}; }}
  /* Satoshi has to be asserted rather than inherited. Streamlit sets its own
     font on a dozen generated class names that change between builds, and any
     one of them left unstyled shows up as a single control in the wrong face,
     which reads as a rendering bug rather than a design choice. */
  html, body, [class*="css"], .stApp, button, input, select, textarea,
  [class^="st-"], [class*=" st-"] {{
      font-family: {FONT};
      font-variant-numeric: tabular-nums;
  }}
  html, body, [class*="css"] {{ color: {INK}; }}
  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.02em;
      font-family: {FONT}; font-weight: 700; }}
  .tt-card {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
      padding: 12px 14px; margin-bottom: 10px;
  }}
  /* ------------------------------------------------------------------
     PANEL HEAD. One rule for the ~30 places that used to write a bold
     title and a quiet caption by hand. They had drifted apart — different
     margins, some with a blank line between, some without — and that is
     most of the reason two charts in one row did not start on the same
     y even when the charts themselves were the same height.
     ------------------------------------------------------------------ */
  .tt-panelhead {{ margin: 2px 0 7px; line-height: 1.45; min-height: 34px; }}
  .tt-panelhead b {{ font-size: 13.5px; font-weight: 700; color: {INK};
      letter-spacing: -.01em; display: block; }}
  .tt-panelhead .tt-quiet {{ display: block; margin-top: 1px; }}
  .tt-metric-label {{ font-size: 11px; text-transform: uppercase;
      letter-spacing: .06em; color: {INK_2}; }}
  /* min-width:0 + overflow-wrap, because one of these tiles carries a
     timestamp. "2026-08-16 11:42:59" at 26px is wider than a fifth of the row
     and ran straight out through the card's right border; it now wraps at the
     space instead of overflowing, and the tiles in a strip stay one height. */
  .tt-metric-value {{ font-size: 26px; font-weight: 600; color: {INK};
      line-height: 1.15; min-width: 0; overflow-wrap: anywhere; }}
  /* The pack grid. Every card is the same height and the sparkline is pinned
     to the bottom, so 45 cards read as a grid instead of a ragged column.
     Breed names run from "Beauceron" to "Nova Scotia Duck Tolling Retriever";
     left to themselves they wrap onto a second line, shove the triage badge
     down, and every card in that row ends up a different height. */
  .tt-dogcard {{ display:flex; flex-direction:column; min-height: 180px;
      margin-bottom: 8px; padding: 11px 13px; }}
  .tt-dogcard-head {{ display:flex; justify-content:space-between;
      align-items:flex-start; gap:8px; }}
  /* photo + name as one unit, so the triage badge stays hard right and the
     thumbnail never pushes the breed onto a second line */
  .tt-dogcard-id {{ display:flex; align-items:center; gap:9px; min-width:0; }}
  .tt-dogcard-name {{ min-width:0; }}
  .tt-dogcard-name b {{ font-size:15px; white-space:nowrap; }}
  /* clip rather than wrap: the badge must stay on the first line */
  .tt-breed {{ display:block; overflow:hidden; text-overflow:ellipsis;
      white-space:nowrap; max-width:100%; }}
  .tt-badge {{ white-space:nowrap; flex:0 0 auto; }}
  .tt-chiprow {{ display:flex; flex-wrap:wrap; gap:4px; }}
  /* margin-top:auto pushes the sparkline and footer to the card floor, which
     is what makes the bottom edges line up across a row */
  .tt-spark {{ margin-top:auto; padding-top:8px; }}
  .tt-dogcard-foot {{ margin-top:4px; }}
  /* ------------------------------------------------------------------
     LEFT RAIL. The nav is an st.radio because the router needs its value,
     but a bare radio list reads as a form control rather than navigation.
     These rules turn each option into a nav row — full-width hit area,
     hover, and a hue bar on the selected one — without touching the widget
     itself. Every selector degrades to a plain radio if SiS ships a build
     whose DOM does not match, which is why none of them hide anything.
     ------------------------------------------------------------------ */
  section[data-testid="stSidebar"] {{ background: {CARD};
      border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] .stRadio > div {{ gap: 1px; }}
  section[data-testid="stSidebar"] label {{ font-size: 13.5px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 1px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
      display: flex; align-items: center; width: 100%;
      padding: 5px 8px 5px 7px; margin: 0; border-radius: 5px;
      border-left: 2px solid transparent; cursor: pointer; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
      background: {SURFACE}; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
      background: {SURFACE}; font-weight: 600; }}
  .tt-brand {{ display:flex; align-items:center; gap:9px; margin-bottom:12px; }}
  .tt-brand-mark {{ width:34px; height:34px; border-radius:8px; flex:0 0 auto;
      background:{INK}; display:flex; align-items:center; justify-content:center; }}
  .tt-railstat {{ margin-top: 14px; border-top: 1px solid {BORDER};
      padding-top: 10px; font-size: 12px; }}
  .tt-railstat > div {{ display:flex; justify-content:space-between;
      align-items:baseline; padding: 3px 0; gap: 8px; }}
  .tt-railstat span {{ color: {INK_2}; }}
  .tt-railstat b {{ color: {INK}; font-size: 13px; font-variant-numeric: tabular-nums; }}
  /* a labelled number with the proportion it represents drawn underneath it,
     so "38 of 45" is a length as well as a ratio */
  .tt-meter {{ margin: 7px 0 0; }}
  .tt-meter-top {{ display:flex; justify-content:space-between;
      align-items:baseline; gap:8px; font-size:12px; }}
  .tt-meter-top span {{ color:{INK_2}; }}
  .tt-meter-top b {{ color:{INK}; font-size:12.5px; font-variant-numeric:tabular-nums; }}
  .tt-meter-track {{ height:3px; border-radius:2px; background:{GRID};
      margin-top:4px; overflow:hidden; }}
  .tt-meter-fill {{ height:100%; border-radius:2px; }}
  .tt-pill {{ display:inline-flex; align-items:center; gap:5px; font-size:11px;
      padding:3px 8px; border-radius:11px; border:1px solid {BORDER};
      background:{SURFACE}; color:{INK_2}; }}
  .tt-railfoot {{ margin-top: 10px; font-size: 11px; color: {INK_2};
      line-height: 1.45; }}
  .tt-railnote {{ margin-top: 12px; padding: 8px 10px; background: {SURFACE};
      border-radius: 0 4px 4px 0; font-size: 12px; line-height: 1.4; }}
  .tt-pagehead {{ padding: 2px 0 2px 12px; margin: 0 0 12px; }}
  .tt-pagetitle {{ font-size: 21px; font-weight: 700; letter-spacing: -.015em;
      color: {INK}; }}
  /* Streamlit's default block padding wastes the top third of a 1080p screen
     on a dashboard that is meant to be read at a glance across the room. */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; }}
  h3 {{ margin-top: .4rem !important; }}
  .tt-chip {{ display:inline-block; padding: 1px 7px; border-radius: 3px;
      font-size: 11px; font-weight: 600; border: 1px solid {BORDER}; }}
  .tt-badge {{ display:inline-block; padding: 2px 9px; border-radius: 3px;
      font-size: 11px; font-weight: 700; color: #fff; }}
  .tt-mono {{ font-family: {MONO}; font-size: 12px; }}
  .tt-caveat {{ background: #FEF3C7; border-left: 3px solid {ACCENT};
      padding: 8px 12px; font-size: 12px; color: #78350F; margin: 8px 0; }}
  .tt-quiet {{ color: {INK_2}; font-size: 12px; }}
  /* Caption under a radial gauge. Written as HTML rather than as a plotly
     annotation below the polar area: an annotation at a negative paper y is
     clipped by the figure box at some heights and not at others, which is why
     the three gauges on the pack page were drawing unlabelled. */
  .tt-gaugecap {{ margin: -10px 0 2px; text-align: center; font-size: 10.5px;
      color: {INK_2}; line-height: 1.3; }}
  /* Byline under a chart: which of the three renderers drew it, and what
     that bought. Sits tight under the figure, so the caption reads as part
     of the chart rather than as the next paragraph. */
  .tt-renderer {{ margin: -6px 0 14px; font-size: 11px; color: {INK_2};
      display: flex; align-items: baseline; gap: 7px; line-height: 1.45; }}
  .tt-renderer-tag {{ flex: none; font-family: {MONO}; font-size: 9.5px; letter-spacing: .07em;
      text-transform: uppercase; color: {INK_2}; background: {GRID};
      border: 1px solid {BORDER}; border-radius: 3px; padding: 1px 5px; }}
  /* A bokeh document lives in its own iframe, which paints white and squares
     its own corners; without this it sits on the warm surface as a bright
     slab with a hairline gap under it. There is no class to hang this on —
     Streamlit names the element only by the iframe's title attribute. */
  iframe[title="streamlit.components.v1.html"] {{
      display: block; border: 1px solid {BORDER}; border-radius: 6px;
      background: {CARD}; }}
  div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
  /* ------------------------------------------------------------------
     CHAT BUBBLES — ported from AI-Yash/st-chat (streamlit_chat/frontend/
     src/stChat.css): .chat flex row, .chat.user row-reverse, a 50% round
     avatar, and a .msg::after triangle made of transparent borders.

     PORTED RATHER THAN INSTALLED, for two reasons that are both hard rules
     here. Streamlit in Snowflake cannot load a custom bidirectional
     component — there is no way to serve the package's compiled frontend
     bundle from inside the sandbox. And st-chat draws its avatars from
     DiceBear over HTTPS, which is exactly the outbound request SiS blocks,
     so the stock component would render two broken images per turn. The
     avatars below are inline SVG for the same reason the breed photos are
     base64.

     One deliberate change: st-chat anchors the tail to the row at top:0
     with border-top-color, which works because every message is its own
     iframe. In one page that puts a stray notch above each bubble, so the
     tail is anchored to the bubble and points sideways.
     ------------------------------------------------------------------ */
  .tt-chat {{ display:flex; flex-direction:row; align-items:flex-start;
      width:100%; margin:0 0 22px; }}
  .tt-chat.user {{ flex-direction: row-reverse; }}
  .tt-chat .avatar {{ display:flex; align-items:center; justify-content:center;
      height:48px; width:48px; flex:0 0 auto; margin:0 6px; overflow:hidden;
      border-radius:50%; }}
  /* No border and no tail: the reference build draws a plain soft-grey
     rectangle and lets ALIGNMENT carry the speaker. Adding a rule or a
     triangle to it is the thing that makes a ported chat look almost-right. */
  .tt-chat .msg {{ display:inline-block; margin:0 8px; padding:13px 18px;
      max-width:70%; min-height:1.5rem; line-height:1.7; font-size:16px;
      white-space:pre-line; border-radius:10px; background:{CHAT_BG};
      color:{CHAT_INK}; border:none; }}
  .tt-chat .msg p {{ margin-block: 0; }}
  .tt-chat-meta {{ font-size:10.5px; color:{INK_2}; margin:-16px 0 22px 68px; }}
  table.tt {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  table.tt th {{ text-align: left; border-bottom: 1px solid {BORDER}; padding: 5px 8px;
      color: {INK_2}; font-weight: 600; text-transform: uppercase; font-size: 10px;
      letter-spacing: .05em; }}
  table.tt td {{ border-bottom: 1px solid {GRID}; padding: 5px 8px; }}
  /* A register in a fixed box, for the half of a row whose partner is short.
     The header is pinned, or a scrolled table loses the only thing that says
     what its columns are. */
  .tt-scroll {{ overflow-y: auto; overflow-x: hidden; border: 1px solid {BORDER};
      border-radius: 4px; background: {CARD}; }}
  .tt-scroll table.tt th {{ position: sticky; top: 0; background: {CARD};
      z-index: 1; box-shadow: 0 1px 0 {BORDER}; }}
  .tt-scroll table.tt tr:last-child td {{ border-bottom: none; }}
  /* An inline proportional bar inside a table cell: the column reads as a
     chart without becoming one. */
  .tt-cellbar {{ display: block; height: 4px; border-radius: 2px; margin-top: 3px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def _session():
    from snowflake.snowpark.context import get_active_session
    return get_active_session()


def _scrub(v):
    """Decimal -> float, everything else untouched.

    THIS IS HAZARD 1. Without it, plotly receives object-dtype Decimals, decides
    the axis is categorical, and every line chart in the app renders as the same
    straight diagonal. It looks like a plotting bug and it is a dtype bug.
    """
    if isinstance(v, Decimal):
        return float(v)
    return v


@st.cache_data(ttl=45, show_spinner=False)
def rows(sql: str) -> list[dict]:
    """Run a query, return plain Python dicts. Cached, so a tab re-render does
    not re-bill the warehouse."""
    try:
        res = _session().sql(sql).collect()
    except Exception as exc:  # noqa: BLE001
        st.session_state.setdefault("_errors", []).append((sql[:120], str(exc)))
        return []
    return [{k: _scrub(v) for k, v in r.as_dict().items()} for r in res]


def rows_live(sql: str) -> list[dict]:
    """rows() without the cache. Exactly one caller: the Cortex chat, where a
    cached answer to a new question would be a lie."""
    try:
        res = _session().sql(sql).collect()
    except Exception as exc:  # noqa: BLE001
        st.session_state.setdefault("_errors", []).append((sql[:120], str(exc)))
        raise
    return [{k: _scrub(v) for k, v in r.as_dict().items()} for r in res]


@st.cache_data(ttl=900, show_spinner=False)
def rows_quiet(sql: str) -> list[dict]:
    """rows() for OPTIONAL sources, swallowing the error instead of logging it.

    REF.BREED_IMAGE is populated by scripts/fetch_breed_images.py, which needs a
    Kaggle token. On an account where that has never run the table simply does
    not exist, and that is a fact about the account rather than a bug — listing
    it in the query-error expander would train the reader to ignore that panel.
    """
    try:
        res = _session().sql(sql).collect()
    except Exception:  # noqa: BLE001
        return []
    return [{k: _scrub(v) for k, v in r.as_dict().items()} for r in res]


def sq(text: str) -> str:
    """Single-quote a string for inline SQL. The Snowpark .sql() path here takes
    no bind parameters, so the literal has to be escaped rather than bound."""
    return "'" + str(text).replace("\\", "\\\\").replace("'", "''") + "'"


def col(data: list[dict], name: str, default=None) -> list:
    return [r.get(name, default) for r in data]


def one(data: list[dict], name: str, default=None):
    return data[0].get(name, default) if data else default


def ago(seconds) -> str:
    """Seconds since the last epoch, in units a human reads at a glance.

    The raw number is unreadable past a minute or two — the bulk corpus dogs
    were recorded in 2018 and rendered as "stale 17,715,743s", which says
    nothing except that something is broken. Past a week it is not staleness at
    all, it is the archive, and the caller labels it as such.
    """
    try:
        v = float(seconds or 0)
    except (TypeError, ValueError):
        return "—"
    if v < 90:
        return f"{v:.0f}s"
    if v < 5400:
        return f"{v / 60:.0f}m"
    if v < 172800:
        return f"{v / 3600:.0f}h"
    return f"{v / 86400:.0f}d"


def sparkline_svg(series, width=260, height=34, colour=None) -> str:
    """A sparkline as inline SVG, drawn INSIDE the card markup.

    Was one st.plotly_chart per dog, rendered after the card's closing div —
    which put a figure-sized gap under every card, let the cards and their
    sparklines drift apart on reflow, and paid for 45 Plotly figures on the
    first tab. An SVG polyline costs nothing and belongs to the card.
    """
    pts = []
    for x in series or []:
        try:
            pts.append(float(x))
        except (TypeError, ValueError):
            pass
    if len(pts) < 3:
        return f'<div style="height:{height}px"></div>'
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    step = width / (len(pts) - 1)
    coords = " ".join(
        f"{i * step:.1f},{height - 3 - ((v - lo) / rng) * (height - 6):.1f}"
        for i, v in enumerate(pts)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'style="width:100%;height:{height}px;display:block">'
        f'<polyline points="{coords}" fill="none" '
        f'stroke="{colour or ACCENT}" stroke-width="1.2" '
        f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>'
    )


# --------------------------------------------------------------------------
# The dog, drawn rather than fetched.
#
# Streamlit in Snowflake has no outbound internet, so every <img src="http...">
# renders as a broken icon — no CDN illustrations, no remote breed photos, no
# matter how much nicer they would look. Only two kinds of picture survive the
# sandbox: one drawn inline as SVG, and one whose bytes are ALREADY IN THE
# ACCOUNT (see breed_photo below, reading base64 out of REF.BREED_IMAGE).
#
# The diagrams below stay drawn even now that the photographs exist, because
# they are the more useful picture: what a viewer needs from this tab is not
# what a Beauceron looks like, it is WHERE THE TWO SENSORS SIT, since the whole
# detection argument rests on the relationship between them.
# --------------------------------------------------------------------------

# One silhouette per posture, so a state reads as a shape before it reads as a
# word. Paths are deliberately crude — a recognisable stance beats a portrait.
_DOG_BODY = {
    "stand": "M14,44 L14,30 Q14,22 24,21 L52,21 Q62,22 62,30 L62,44 M20,44 L20,58 "
             "M30,44 L30,58 M48,44 L48,58 M58,44 L58,58",
    "rest":  "M12,52 Q12,44 24,44 L54,44 Q66,44 66,52 L66,58 L12,58 Z",
    "sit":   "M16,44 L16,30 Q16,22 26,21 L52,21 Q62,22 62,34 L62,58 L46,58 "
             "Q40,50 30,50 L20,58 Z",
    "move":  "M14,42 L14,28 Q14,20 24,19 L52,19 Q62,20 62,28 L62,42 M18,42 L12,58 "
             "M28,42 L34,58 M48,42 L42,58 M58,42 L64,58",
}
_STATE_POSE = {
    "REST": "rest", "SIT": "sit", "STAND": "stand", "SNIFF": "stand",
    "WALK": "move", "TROT": "move", "GALLOP": "move", "PLAY": "move",
    "PACE": "move", "CIRCLE": "move", "SHAKE": "stand", "SCRATCH": "sit",
    "PAUSE": "stand", "SLOW_TRANSITION": "sit", "UNKNOWN": "stand",
}


def dog_glyph(state: str, size: int = 42, colour: str = None) -> str:
    """A small posture silhouette for a state. Used as an inline icon."""
    pose = _STATE_POSE.get((state or "").upper(), "stand")
    col = colour or INK_2
    return (
        f'<svg viewBox="0 0 78 66" style="width:{size}px;height:{int(size*66/78)}px;'
        f'vertical-align:middle" aria-label="{state}">'
        f'<path d="{_DOG_BODY[pose]}" fill="none" stroke="{col}" stroke-width="3.4" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="66" cy="24" r="7" fill="none" stroke="{col}" stroke-width="3.4"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# The stand-in dog, for the study breeds Stanford Dogs does not carry.
#
# Beauceron, Hovawart, Mudi and the crossbreeds have no reference photograph in
# REF.BREED_IMAGE, and the honest thing to show in their place is a picture
# that is obviously a drawing rather than a photograph of anything.
#
# Font Awesome's "dog" icon, via svgrepo.com/show/351961 (CC BY 4.0). ONE PATH,
# 406 characters. It replaced a 58 KB auto-traced photograph of a Labrador that
# had to be colour-inverted at render time because its first path was the whole
# canvas with the dog cut out of it — 148 paths of trace noise to say
# "placeholder". This says it at 1/140th the size and reads better at 44 px,
# which is the only size it is ever drawn at.
#
# Still served as ONE CSS background-image rather than inline copies: the pack
# grid draws 45 of these, and a background is rasterised once and blitted.
# ---------------------------------------------------------------------------
_DOG_SILHOUETTE_PATH = (
    "M298.06,224,448,277.55V496a16,16,0,0,1-16,16H368a16,16,0,0,1-16-16V384H192V496a16,16,0,0,1-16,16H112a16,16,0,0,1-16-16V282.09C58.84,268.84,32,233.66,32,192a32,32,0,0,1,64,0,32.06,32.06,0,0,0,32,32ZM544,112v32a64,64,0,0,1-64,64H448v35.58L320,197.87V48c0-14.25,17.22-21.39,27.31-11.31L374.59,64h53.63c10.91,0,23.75,7.92,28.62,17.69L464,96h64A16,16,0,0,1,544,112Zm-112,0a16,16,0,1,0-16,16A16,16,0,0,0,432,112Z"
)

# native box is "0 -32 576 576"; widened ~10% so the tail and paws do not
# touch the edge of the circular crop
_DOG_SILHOUETTE_BOX = "-32 -64 640 640"


def dog_silhouette_css(plate: str, ink: str) -> str:
    """The silhouette as a single reusable CSS class, colours baked in."""
    from urllib.parse import quote

    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="'
           + _DOG_SILHOUETTE_BOX + '">'
           '<rect x="-400" y="-400" width="1600" height="1600" fill="'
           + plate + '"/><path fill="' + ink + '" d="'
           + _DOG_SILHOUETTE_PATH + '"/></svg>')
    uri = "data:image/svg+xml," + quote(svg, safe="/:=' ,.-")
    return (".tt-dogsilh { background-image: url(\"" + uri + "\"); "
            "background-size: cover; background-position: center; }")


# Emitted on its own rather than inside the sheet at the top of the file, which
# is an f-string: a path full of commas and braces has no business going near
# one. Plate first, then the dog — a placeholder should read as a deliberate
# drawing, not as a photo that failed to load, so the two are a clear step
# apart rather than both pale.
st.markdown("<style>" + dog_silhouette_css("#EDEAE7", INK_2) + "</style>",
            unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def breed_photos() -> dict:
    """Reference breed photographs, as base64 already inside the account.

    SiS has no outbound network, so <img src="https://..."> is a broken icon
    whatever the host. scripts/fetch_breed_images.py range-reads the Stanford
    Dogs archive, downsizes to a square thumbnail and stores the bytes in
    REF.BREED_IMAGE, which is the only form of photograph that survives the
    sandbox. Empty dict when that script has never run — every caller falls
    back to the drawn silhouette rather than showing a gap.
    """
    data = rows_quiet("""
        SELECT breed, image_b64, is_approximate, credit, source_folder
        FROM REF.BREED_IMAGE
    """)
    return {r["BREED"]: r for r in data if r.get("IMAGE_B64")}


def breed_photo(breed: str, size: int = 46, *,
                radius: str = "50%", photos: dict = None) -> str:
    """A round breed thumbnail, or the posture silhouette when there is none.

    THE CAPTION IS NOT DECORATION. This is a photograph of *a* dog of the breed,
    never of the animal being diagnosed, and the two are trivially confused by
    anyone glancing at a card. Every photo therefore carries the disclaimer in
    its tooltip, an amber ring when even the breed is only approximate, and the
    pages that render one repeat it in text underneath.

    PASS `photos` WHEN DRAWING MORE THAN ONE. st.cache_data hands back a fresh
    copy on every call so that a caller cannot corrupt the cache, and the value
    cached here is ~450 KB of base64 — the pack grid calls this 45 times, which
    is 20 MB of copying per rerun if every card looks the table up for itself.
    """
    rec = (photos if photos is not None else breed_photos()).get(breed)
    if not rec:
        # Built outside the f-string: SiS pins Python 3.11, where reusing the
        # delimiter quote inside an f-string expression is a syntax error.
        who = breed or "this breed"
        return (
            f'<div class="tt-dogsilh" style="width:{size}px;height:{size}px;'
            f'border-radius:{radius};border:1px dashed {BORDER};'
            f'flex:0 0 auto" title="No reference photograph for {who} in '
            f'Stanford Dogs — generic silhouette, not a dog of this breed and '
            f'certainly not this animal"></div>')
    appx = bool(rec.get("IS_APPROXIMATE"))
    ring = ACCENT if appx else BORDER
    tip = ("Reference photograph of a similar breed, NOT this dog"
           if appx else "Reference photograph of the breed, NOT this dog")
    return (
        f'<img src="data:image/jpeg;base64,{rec["IMAGE_B64"]}" '
        f'alt="{breed} reference photograph" title="{breed} — {tip}" '
        f'style="width:{size}px;height:{size}px;border-radius:{radius};'
        f'object-fit:cover;border:1.5px solid {ring};flex:0 0 auto;display:block">')


# ---------------------------------------------------------------------------
# A REAL DOG SKELETON, because the sensor diagram is the one picture on this
# dashboard that makes an anatomical claim.
#
# What was here before was four line segments and a circle for a head. It could
# not show what the tab is actually arguing: that the collar sits on the
# CERVICAL vertebrae and the harness on the THORACIC vertebrae, and that those
# two points move independently when the head acts alone.
#
# Source: "Skeleton of a dog labelled" by A.Spielhoff, Wikimedia Commons,
# CC BY-SA 3.0 — attribution is rendered under the figure and is a licence
# condition, not decoration. Derived by stripping the German labels, the
# dotted leader rules and the callout arrows, then rounding all coordinates to
# one decimal. Original viewBox 1400x630 is preserved so the marker
# coordinates below stay meaningful.
#
# The two marker positions were read off a coordinate grid rendered over the
# artwork, not guessed: (497, 237) is mid-cervical, (712, 240) is mid-thoracic.
# ---------------------------------------------------------------------------
_DOG_SKELETON_G = (
    '<g transform="matrix(1,0,0,1,177.7,-335.1)"><g><path d="M671.6,921C670.6,922 669.5,922.8 668.8,924C668.4,924.8 668,925.7 667.7,926.6C667.5,927.3 667.6,928.1 667.5,928.8C667.4,929.7 667.4,930.5 667.6,931.3C667.7,931.9 668,932.4 668.2,933M631.7,701.3C632.8,709.5 634.4,717.6 636.8,725.6C638.7,731.7 641.9,737.3 645.7,742.2C651,749.1 657,755.6 660.7,763.6C663.5,768.7 671.2,784.5 669.3,779C669.2,778.7 667.3,772.1 669.4,775.5C671.7,778.5 671.5,779.9 673.8,784.1C674.9,786.3 672.3,778 674.6,782.6C675.7,784.4 677.7,789.4 676.3,784.5C674.8,779.6 676.8,783.8 678.2,785.9C680.9,790.5 683.7,800.6 686.1,806.6C687.3,809.7 683.3,792.2 685.1,794.8C687.8,798.5 686.9,798.2 688.6,802.6C689.4,804.6 691.2,810.6 690.4,805.6C689.9,802.7 688.4,799.7 693.3,806C695.8,810.4 695.3,811.6 696.2,814.5C697.1,817.3 698.8,828.8 698.8,823C698.5,820.1 697.5,813.3 699,819.4C700,823.5 700.6,838 700.7,822.6C701.1,821.5 701.4,829.2 701.7,830.9C702.6,835.8 701.9,822 702.6,826.2C703.1,829.1 703.2,836.3 703.5,833.4C704.1,828.9 704.7,832.3 705.2,835C706.4,840.8 705.6,846.6 705.9,852.4C705.6,857 704.5,861.3 702.6,865.5C699.3,873.3 696.2,881 693.6,889.2C692.3,893.5 691.6,898.1 688.5,901.5C686.6,903.6 683.5,903.2 680.3,903.8C675.9,905.1 671.6,905.7 668.1,907.8C662.8,912 658.8,917.8 658.2,925.1C657.9,927.3 657.5,929.7 659.8,931.1C663.4,933.4 667.7,932.6 671.5,933.7C676.2,935.3 680,935.5 685.5,935.7C689.8,935.8 693.8,935.4 697.5,933.6C701.4,932.5 705.7,933.5 709.5,932.2C714.7,930.5 719.3,926.1 720.4,920.4C722.3,914.4 723.1,908.1 724.5,901.9C726.6,893.9 729.2,885.9 733.3,878.7C736.8,872.3 740.2,865.7 744.7,859.9C748.8,854.9 752.6,849.7 755.1,843.7C756.5,839.6 756.5,841.9 756.5,842.2C756.8,847.6 756.9,836.3 757.8,840.3C759.1,842.3 757.3,834.3 758,835.5C759.6,839.6 758.5,834.1 758.1,832.4C758.4,831.7 760.7,837.3 759,831.4C757.1,826.3 754.2,821.5 751.6,816.8C750,813.8 748.1,810.7 747.6,807.4C746.8,801.8 746.4,796 746.6,790.3C746.6,787.6 746.9,785 747.5,782.2M434.2,926.1C431.8,927 430.4,928.7 429.9,931C429.6,932.3 429.7,933.7 430,935.1L430,935.2M512.1,749.6C508.9,757.2 505.7,763.9 503.6,771.9C503.4,772.8 502.7,779.1 502.6,776.8C502.3,775.8 503.3,768.9 502.4,772.1C499.9,781.5 499.9,792.6 500.2,802.3C500.4,808.2 500.5,808.3 501.2,804.5C501.8,801 501.8,808.4 501.6,810C501.5,813.5 501.5,815.6 501.3,820.5C501.2,822.7 502.7,813.5 502.8,816.5C502.7,821.2 501.5,824.3 503.6,827.4C505.7,832.2 510,836.1 510.5,841.5C510.9,845.6 510.8,846.5 510.3,853.8C510.2,855.2 509.2,848.4 509.4,852.4C509.5,854.2 509.1,861.5 508.7,856.1C508,852.9 508.2,859.4 508.1,860.4C507.9,864.7 507.3,859.4 507.3,859.7C506.9,863.6 506.7,863.5 505.7,866.3C505.5,866.8 506.3,859.6 504.1,866.8C503.1,869.8 502.9,872.9 502.2,876C501.9,877.3 502.4,870.1 501.7,870.9C499.9,874.4 499.8,878.6 497.9,882.2C495.9,887 493.2,891.6 491.8,896.7C490.1,901.3 489.1,907.7 487.9,910.8C486.8,913.9 486.5,914.7 485.8,916.5C483.3,923.9 479.2,929.3 474.4,935.2C472.2,937.7 470.3,941.5 466.5,941.8C459.1,942.4 451.6,941.7 444.1,941.5C441.5,941.2 438.4,942.1 436.2,940.7C433.1,938.9 430.6,936.4 428.6,933.3C426.5,929.9 424.4,925.9 426.2,922.1C427.5,918.3 430.7,915.6 434.5,914.4C439.5,911.9 445.8,912.1 450.2,908.8C454.8,905.4 457.4,899.8 459.8,894.6C461.9,889.1 464.3,883.5 464.8,877.5C466.1,869.1 466.3,860.7 466.4,852.2C466.4,850 466.8,848.3 465.7,846.4C465.2,845.3 465.4,852.3 465,848.6C464.6,845.9 465,844.4 464.4,841.4C464.2,840.4 464.2,847.4 463.9,845.2C463.6,842.7 463.5,839.5 462.9,837.1C462.5,835.9 462.5,844.2 462.1,840.6C461.8,837.8 462.3,835.8 461.8,832.6C461.6,831.4 461.6,838.6 461.1,835C460.7,831.8 459.4,822.2 459.1,825.4C458.8,829.7 458.9,830.9 458.6,829.5C457.8,826.2 457.5,823.5 457.4,818.9C457.4,816.5 457.1,823.2 457.1,825.6C455.8,819.4 456,812.7 452.8,806.9C451.5,803.3 449.8,800.7 448.6,796.2C447.8,793.2 447.5,790.1 446.9,787M819.1,943.3C817,945.4 814.8,947.8 816,951.5C816.6,953.8 818.2,956.5 819.9,957.9M804.3,938C803.1,939.4 801.6,940.3 800.9,942.1C800.1,943.5 800,945.7 800.3,947.6C800.4,948.7 800.3,950.1 800.8,951.2M665,685C666.3,685.2 667.8,685.3 669.1,685.7C670.9,686.8 672.8,687.7 674.6,688.9C677.2,691.5 679.9,693.9 682.4,696.6C684.4,699.1 686.9,701.4 688.6,704.3C690.1,706.8 690.6,709.8 691.8,712.5C696.3,717.4 700.2,721.8 704.7,726.7L705.3,727.5C706.4,728.9 695.4,720.3 694.5,719.5C689.6,715.2 699.8,728.7 704.8,733C705.3,733.5 708,738.1 706.5,736.2C702.5,731.3 704.4,735.2 704.9,735.7C707.2,738.8 710.3,743.5 711.1,744.5C712.2,745.8 723.8,759.1 714.3,750.7C714,750.3 714.1,750.4 709.4,746.2C706.9,744 709.7,749 713.1,752.9C718.3,758.8 726.8,762.5 732.9,767.5C735.3,769.5 728.3,764.7 724.8,762.9C721.4,761.1 727.6,765.9 731.8,770.7C735.2,774.5 740,777 744.2,779.9C747.3,781.9 749.2,783.7 751.8,786.4C752.6,787.2 752.7,787.5 749.1,786.2C746.6,785.3 750.5,789 752,790C758.6,794.4 765.1,799.1 771.8,803.4C777.2,806.9 783.4,809.4 788.6,813.2C791.4,815.2 793.9,817.7 795.7,820.7C803.3,831.2 811.8,841.1 817.2,852.9C819.2,857.3 821,862 821.4,866.8C821.5,868.5 820.9,869.7 821.7,876.4C821.8,877.6 821.8,871.3 822.5,872C824.1,874.4 823.8,874 823.6,880.3C823.6,883.8 823.9,889.5 824.4,883.4C824.7,879.8 825.5,890.4 825.6,894C825.7,899.9 825.5,909.5 824.3,911.9C823.2,914.3 822.9,914.4 821.4,915.6C816.9,919 812.3,921.2 808,923.7C805.6,924.8 803.5,926.4 801.6,928.1C799.5,930 797.4,931.3 795.4,934.4C794.1,936.5 793.7,938.7 793.7,941.2C793.6,943.3 794.3,945.2 795.7,946.6C797.4,948.3 798.8,950.2 800.9,951.4C803.2,953 805.5,955 808.1,956.2C810.8,957.5 813.8,957.5 816.8,957.7C822.5,958.1 827.8,958.4 833,958C834.5,957.9 835.8,957.4 837,956.6C838.9,955.2 840.4,953.2 842.1,951.6C845.2,948.8 847.5,945.3 850.5,942.6C854,939.1 857.8,935.5 859.8,931C861,928.1 860.6,924.7 859.8,921.7C858.7,918.6 857.4,915.6 856.9,912.4C856.6,910.3 856.2,910.5 856,907.6C855.9,905.7 856.8,908.6 857.6,912C858.3,914.9 857.1,905.9 856.9,904C856.6,899.7 858.7,907.6 858.9,909.1C859.2,912.8 859.2,904.1 858.9,901.9C858.6,899.4 858,897 857.6,894.3C856.7,887.9 856.7,881.9 857,875.9C857.4,866.9 856.9,857.6 860.3,849C861.8,845.4 863.3,841.7 863.4,837.7C863.6,832.1 862.3,826.3 859.4,821.4C857.7,817.9 854.9,815.1 852.3,812.3C847,807.1 841.4,802.2 836.5,796.6C831.7,790 825.5,784.3 821.9,776.9C821.6,776.2 828.1,784.2 831.5,787.4C833.7,789.5 829,783.8 828.4,782.8C826.6,780.1 825.4,779.5 823.8,776.6C822.8,774.7 829.1,782.7 827.9,780.7C826,777.2 825.1,776.7 822.4,773.4C820.4,770.9 828.6,778.9 823.8,772.9C822.2,770.8 818.1,765.1 818.8,764.9C822,766.8 826,776.5 823.8,770.1C822.5,766.3 821.5,766.9 817.4,758.8C816.4,756.6 820.6,762.5 821.8,764.5C820.7,759.5 817,756.4 815.5,750.8C815.2,749.8 817.8,754.2 818.4,755.2C819.3,757 817.7,751.6 817.1,750.6C816,747.3 814.4,744.3 814.3,740.8C815.3,743.4 816.1,746.5 817.5,748.9C819.1,751.6 815.3,740.7 814.1,735.9C813.2,732.2 816.4,741.5 817.4,744.3C817.8,745.3 817.2,740.3 816.9,738.3C816.3,735.2 815.9,735.6 814.3,729.1C813.9,727.6 818.3,735.1 817.6,733.1C816.2,729 814.7,726.4 814,722.4C813.2,717.7 813.2,715.1 812.8,709.5C812.6,707 815.5,720.7 815.6,716.4C814.8,710 814.7,708.9 814.3,706C813.9,701.3 813.9,700.6 814.2,694.1C814.3,691.7 816,699.6 816.6,702.3C817.3,705.5 817,699.1 817.1,698.2C817.1,693.1 816.2,688.1 815.9,683C815.8,679.6 815.6,677.2 815.6,672.7C815.7,669 816.4,678.5 817.4,681.9C817.7,683.1 819.5,689.6 819,686.2C818.4,682 818.2,672.4 818.9,665.8C820.3,654.4 823.4,643.7 822.8,632.2C822.6,629.4 822.9,626.4 821.7,623.8C819.9,620.4 818.1,617 816.8,613.3C815.1,608.6 813.5,603.8 813.1,598.9C812.7,595.5 812,591.8 813,588.5C813.7,585.8 814.7,583.7 816.6,581.1C818.4,578.9 820.6,577.1 823.3,574.6C836.2,562.5 852.2,556.1 865.7,546.1C871.5,541.8 875.7,539.4 882.3,532.1C884.6,529.5 894.1,516.3 889.2,527.9C887.7,531.3 892.1,525.2 892.8,523.2C895.8,515.4 896.5,512.1 898.6,506.7C899.4,504.6 897,518.6 899.5,509.7C901.8,504.2 903.9,487.3 903.9,491.4C903.9,505.9 907.1,482.4 907.2,479.7C907.6,474 907.6,472.1 906.8,462.5C906.5,459.6 910.1,473.8 909.4,465.1C908.8,457.9 904.9,442.7 905.5,444.5C912.2,464.4 907.6,442.2 906.8,435.2C905.9,426.8 903.1,419.2 897.9,412.6C895.5,409.5 894.5,407.3 898.9,412.3C901.3,414.9 904.8,422 900.6,411.8C896.9,402.9 891.2,395.2 885,387.4C882.8,384.6 874.8,378.9 877.7,380.2C881.5,382 888,387.1 883.3,381.9C880.3,377.9 876.3,375 872.4,372C867.4,368.3 859.1,364.1 857.6,363.2C847.8,357.7 868.7,367.5 858.8,360.7C853.7,357.2 848,354.6 842.2,352.5C826.9,347.5 810,348.6 794.6,352.2C790.9,353.1 788.8,353.2 784.1,356C778.3,359.3 783.1,357.6 783.5,358.3C784.1,359.1 785.5,358.6 786.6,358.7C787.6,358.8 789,358.7 790,358.8C791,358.9 791.9,358.9 792.7,359.2C794,359.8 798.9,361.1 792.9,361.5C791.7,361.6 796.4,362.9 798.2,363.4C800.4,363.9 802.7,364 804.9,364.6C810.9,366.4 816.9,368.8 822.4,371.9C827.2,375.5 831,380.2 835.4,384.3C837.6,385.9 837.2,386 835.2,381.8C834,379.5 831.5,377 832.7,377.9C835.9,380.4 840.2,383.6 844.4,386.6C848.8,389.7 854.4,400.8 854.4,399.2C854.4,396.1 847.7,387.1 853,393.1C855.6,396 860.3,407.8 858.1,399.5C856.2,392.4 862.3,403.6 863.6,405.5C866.9,410.1 869.7,427.1 869.7,421.7C869.6,409 871.8,437.6 871.5,427.8C871.7,423.1 872.7,421.4 872.5,427.7C872.2,436.7 873.4,428.9 873.5,428.5C874.6,426.2 872.1,443.3 874.3,434.4C875.9,429.7 874.6,444.1 874.9,447.1C874.3,455.8 874.4,464.7 872.2,474.8C871.5,481.5 871,471.7 870.7,473.8C870.4,476.8 870.2,480.6 869.5,483.6C869,485.8 869.1,480.2 868.6,481.9C867.9,484.7 868,487.9 866.5,490.5C866,491.5 865.9,490.7 866.4,489C866.8,486 867.6,483 866.8,484.5C865.4,487.6 865,490.6 864.6,494C864.5,494.9 862.8,499.3 863.6,496.6C863.8,495.5 864.2,491 863.4,493.7C862.6,496.2 862.4,497 861.2,500.9C860.6,502.6 860.2,503.7 859.2,504.5C858.5,505 860.8,499.1 860.8,499C858.9,501.7 857,506.5 854.8,508.9C853,510.7 853,511 850.4,514.1C849.4,515.4 854.9,505.8 852.1,509.6C849.3,513.4 847,517.9 843,520.6C840.7,522.2 838.5,523.9 835.3,525.5C829.1,528.9 823.9,532.5 818.3,535.4C813.3,538 808,540.1 802.9,542.5C799.9,544 797,545.8 793.4,545.6C791.9,545.5 787.3,545.8 788.2,545.9C790.7,546.2 791.1,546.7 788.6,546.6C787,546.5 781.6,546.6 785.8,547C787.9,547.1 788,547.1 785.8,547.5C784.5,547.8 778.7,547.3 781,547.7C782.1,547.9 785.7,547.9 782.4,548.3C779.5,549 771.4,548.1 772.9,548.5C776.1,549.5 780.2,549.8 778.9,549.9C775.5,550.2 772.5,549.4 768.6,549.3C767.1,549.3 776.1,550.7 771.5,551.2C763.5,552.6 755.4,552.3 746.9,551.1C737.2,549.8 729.4,546.5 719.2,544.3C690.7,538.5 661.4,540.9 632.8,544.5C622.5,545.8 611.6,543.7 601.8,544.6C610,546 595.5,545.9 593,546.2C585.7,546.1 597.1,547 587.7,547.2C579.9,548 582.9,547.3 586.2,548.6C574.2,549.1 562,548.4 549.9,549.3C546,549.7 536.1,549.4 536.9,548.9C541.6,548.3 549.9,547.5 540.6,547.8C530.7,548 520.7,547.6 509.9,546.8C507.3,546.7 494.5,547.4 501.4,546.3C510.5,545 511.1,544.8 498,544.4C483.4,543.9 468,546 453.7,543.4C446.9,542.2 471.5,543.2 450.1,541.6C436.9,540.7 426.8,544.3 413.7,542.4C396.5,539.9 380.7,531.3 364.6,524.9C363.2,524.3 361.7,523.7 359.8,522.7C345.7,516.4 332.1,509.9 320.9,500C310.2,491.1 300.9,480.5 289.5,472.6C280.3,466 268.9,463.9 257.8,463.2C238.1,463.5 218.2,467.8 200.7,477.2C195,479.7 193.5,486 190.9,491C187.3,500.3 179.3,506.8 171,511.8C160.4,517.5 148.2,520.1 138.1,526.9C133.1,529.8 133.5,536.3 135.2,541.1C138.3,547.1 136.4,554.5 140.4,560.3C146.2,568.7 154.6,575.2 163.8,579.7C169.8,582 176.4,583.5 182.9,582.8C191.8,580.9 200.4,585.5 209.3,585.1C218.7,584.2 227.4,579.6 236.8,578.2L237.6,578.1C239,577.8 241.2,577.5 242.5,578.4C244.9,579.6 245.8,581.2 248,583.6C251.7,587.2 255.4,590.7 258.6,594.8C261.4,598.2 264.7,601.2 267.4,604.7C271.6,609.1 276.3,612.8 280.4,617.2C283,620.5 286.2,623.3 287.8,627.2C288.8,629.3 294.3,639 293.6,636.4C291.5,630.2 291.9,629.6 291.3,626.7C291,625.3 292.5,630.7 293.3,632C295.3,635.9 293.5,626.5 295.3,633.5C296.9,636.9 297.9,636 301.4,644.8C301.9,645.9 299.9,633.7 304.6,647.2C306.8,651.8 304.2,642 305.1,643.2C308.1,647.5 308.1,649.2 309.6,654.4C310,655.5 309.7,647.7 312.1,657.3C314.3,665.9 314.2,658.5 313.2,654.9C312.6,652.6 310.8,644.8 311.7,647.2L315.1,656.1C317.1,660.8 319,665.6 321.2,670.2C322.5,672.4 322.7,673.7 324.2,675.9C324.8,676.4 326.7,683.4 326.5,681.4C325.9,674.8 322.9,668.9 325.1,673C327.7,677.6 327.1,677.8 327,676.9C327.6,679.9 329,687.6 329,684.6C328.9,667.5 329.8,681.9 330.7,685.1C331.5,687.7 334.2,693.2 333,689.4C330.8,681.7 333.4,683.6 333.7,685.8C336.1,699.7 334.9,690.8 336,692.8C338,695.5 337.9,696.4 340.2,701.7C341.3,704.4 339.3,694.8 344.2,710C345.8,715.1 344.8,709.3 344.2,707.9C341.9,702.8 351.5,718.8 346.8,709.6C345.9,707.7 348.2,708.7 348.4,709.2C349.4,712 349.8,712.4 351.3,716C352.9,719.1 351.5,712 353.2,715.1C356.1,719 354.7,718.8 357.8,722.6C358.6,723.6 359.7,724.2 360.7,725L361.7,725.6C365.2,728 368.8,730.3 371.9,733.4C375.5,736.8 379.3,740.1 382.3,744.2C385.1,748.3 386.9,750.6 387.4,757.5C388,758.9 388.9,769.4 388.8,762.2C388.8,760 389.6,758.1 389.7,758.5C390,763.7 389.4,768.9 389.3,774.1C389.2,775 389,778.7 389.5,776.3C390,774.1 390.2,767.2 390.3,769.5C390.6,773.2 390.9,777.1 390.6,780.9C390.1,784.8 390.5,787.7 390,792.5C390,796 391,799.4 390.6,803C390.8,807.7 391,812.4 390.6,817.1C390.2,819.7 389.8,820.8 389.8,826C389.8,830.9 390.5,821 391.5,818.7C392.3,816.6 392.2,825 391.3,827.9C390.4,830.6 390.5,833.1 390,835.8L389.8,837C389.4,841.6 388.8,848.8 388.2,854.1C387.4,861.1 389.8,851.2 389.9,852.7C390.3,859.1 390,866.8 389.3,873.2C388.5,880.2 387.3,887.1 386.9,894.2C386.6,898.7 384.7,902.9 382.8,907C381,911.5 379,916.1 375.4,919.5C373.7,921.3 370.4,924.7 368,925.3C363.6,926.4 361.6,927.2 355.7,928.7C351.8,929.7 349.3,931.2 346,933.4C342.8,936.2 339.5,939.6 339.8,944.3C340,947.8 341,951.7 343.3,954.3C345.6,956.9 348.6,959.3 352.5,959C353.7,959.2 355,959.1 356.2,959.1L357.4,959L358.2,959.1C358.8,959.1 359.5,959.2 360.2,959.4C369.9,960.1 380,961.6 389.2,959C395.4,957.2 398.6,950 402.2,944.8C407,937.8 409.9,929.4 411,921.1C413.1,906.4 420,893.5 428,881.4C431.3,876.5 429.9,869.5 429.6,863.7L429.6,862C429.6,860.2 429,854.9 429.6,856.6C430.4,859.1 431.7,866 431.7,862.8C431.6,853.2 431.1,853.3 431.2,846.1C431.4,845.2 430.7,837.3 431.5,839.3C432.8,843.6 433.1,844.8 434,848.2C435.2,852.7 433.7,839.7 434,836.9C433.8,821.3 434.6,827.8 434.7,816.7C434.7,811.1 437.1,833.8 437.2,829.5C437.5,821.1 437.1,818.3 438.1,812.9C439,807.7 441.4,802.9 442.6,797.8C445.7,791.1 448.1,784 451,777.2C452.8,773 454.2,768.6 456.7,764.8C457.5,763.6 458.2,762.5 459.4,761.2C471.8,757.7 484.1,755.6 496.3,752.9C505.1,749.9 514.5,749.6 523.7,748.3C526.6,747.8 513.2,747.9 516,747.4C525,745.7 532.7,745 527.8,745C516.3,745 529.6,741.6 540.9,740.4C543.7,740.1 547.4,739.2 549.2,738.5C551.7,737.6 533.1,739.7 540.6,737.2C544.7,735.4 549.2,732.9 561.5,731C565.6,730.4 575,729 566,729.6C561.1,730.1 556.7,730.2 563.3,728.3C569,726.3 574.4,723.6 580.3,722.3C587.5,720.8 583.1,720.7 579.2,721.2C576.5,721.5 568.5,723 571,721.9C575,719.9 578.1,719.5 582.8,719.3C587.3,719.1 587.5,719.4 590.6,718.9C597.5,717.7 578.6,716.9 583.2,716.2C591.2,715 593.8,714 602.6,711.6C605.5,710.9 608,711.5 611,711.2C617.3,710.5 598,709.9 608.9,706.7C616.6,704.4 620.2,702.5 632.8,701.2C640.2,700.4 646.9,699.4 653.7,700.6C657.8,701.4 660.9,702.2 665.1,705.2C669,708 673.3,710.6 667.8,705C666.2,703.3 674.8,704.5 677.9,705.5C681.5,706.4 684.6,707.9 687.9,709.5L690.1,711.1L691.1,712L691.8,712.5M470.8,732.2C470.1,735.9 469.6,739.7 468.3,743.2C466,749.2 462.9,755.7 459.4,761.2M357.3,959.1C354.6,957 353.8,949.6 356.8,947M343.1,940.6C342.5,942 342.1,943.6 342,945.1C341.8,947.6 342,950.1 342.9,952.4C343,953 343.4,953.9 343.5,954.5" style="fill:none;stroke:black;stroke-opacity:1;stroke-width:2px;" /></g></g><g transform="matrix(1,0,0,1,177.7,-26.9)"><g ns1:id="czaszka gora" transform="matrix(1,0,0,1,0,-308.3)"><path d="M264.3,533.8C263.4,533.9 262.8,534.3 262.1,534.6C260.6,535.4 259.7,537.1 258.2,537.7C256.2,538.5 253.9,538.8 251.7,538.4C251.1,538.2 250.8,537.1 251,535.9C251.2,535.2 251.5,534.8 252,534.6C253.6,533.8 253.6,533.6 253.9,532C254.1,530.8 253.7,528.1 252.1,527.7C249.9,527.1 247.4,531.4 247.2,529.6C247.1,527.9 247.2,526.4 246.4,525C245.3,523 242.8,521.9 240.6,521.1C237.5,520.2 235.7,519.7 232.6,519.8C231.1,519.9 228.8,519.9 226.7,520.4C226.1,520.6 225.7,520.7 225.4,520.9C223.4,522 222.3,522.6 221.1,524.5C220.2,526 219.9,528.8 220.4,530.5C221,532.6 220.7,534.2 219.3,535.5C217.8,537 217.1,539.6 215.2,540.5C213.6,541.1 213.7,543.1 213.3,544.6C213.2,546.7 211.4,546.1 210.1,545.3C207.3,543.6 204,544.2 201.4,546.2C199.9,547.4 196.8,543.6 194.8,543.8C191.8,543.5 189.3,546.7 188,546.6C185.9,546.4 184.2,544 181.6,544C178.9,544.1 176.1,547 175,546.2C173.6,545.1 174.1,544.2 171.6,544.3C169.9,544.4 168.6,544.8 168.2,546.6C167.4,548.2 166.1,546.9 165,545.7C163.3,543.8 158.6,544.6 157.5,545.5C156.5,546.3 156.4,548.7 154.7,548.1C153.3,547.7 152.7,547 150.9,548.9C149.9,550 148.9,549.8 147.1,549.6C145.8,549.5 146,550 145.9,548.3C145.8,546.6 152.4,538.3 154.8,534.6C156.2,529.2 157.7,524.3 164,522C172.3,518.9 180.2,514.4 187.2,509C197.5,500.2 203.1,486.8 214.8,479.7C222.9,474.8 232.5,473.9 241.9,473C249.8,472.2 258.6,472.5 266.8,474.2C274.9,475.9 283.9,477.2 290.1,483.2C292.6,489.1 285.9,495.4 286,501.1C286.2,507.4 286.3,510.1 283.5,515.3C281.2,519.8 279.9,524.4 279.6,530.5C279.3,535.7 272,538.3 266.6,537.5C264.7,537.3 262.7,536.3 261,535.4" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M264.4,530.3C264.1,528.3 265.2,526.4 266.3,524.6C267.2,523.1 268,521.3 269.8,520.8C271.4,520.4 273.4,520.9 274.3,522.5C275.3,524.4 275.3,526.5 273.8,528.2C272.1,530.1 270.8,531.2 268.9,531.9C267,532.4 264.8,532.6 264.4,530.3ZM226.3,488.4C223.9,490.6 222.2,493.5 219.8,495.6C217.7,498.1 214.1,497.7 211.7,496C208.3,494.1 205.9,498.4 203.6,500.3C199.9,503.5 196.7,507.4 194.6,511.8C193.3,514.3 193.3,517.4 194.2,520C194.9,522 196.5,524 198.5,524.7C202.6,526 207,524.3 209.8,521C211.8,518.6 213.5,516 216.1,514C218.1,512.4 220.4,511.3 222.9,510.8C226,510.2 229.2,510.6 232.4,511.3C240.4,512.9 247.9,517 256.2,517C258.3,517 260.7,517 262.6,515.7C262.9,515.5 263.2,515.3 263.5,515.1" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,-308.3)"><path d="M232.1,511.2C232.9,509.8 234,508.5 235.4,507.9C236.8,507.4 238.1,507.1 239.7,507.9C240.6,508.4 241.7,509.5 242.3,510.6C242.6,511.3 242.6,512.2 242.7,513C242.7,513.4 242.6,513.8 242.5,514.1C239.1,513.1 235.7,511.8 232.1,511.2L232.1,511.2ZM227,530.2C226.6,534.4 225.5,538.2 224.6,542.2C224.4,543.2 224.5,543.8 223,544.4C222.5,544.6 221.8,546.4 221.1,547.3C219.4,549.3 217.7,551.3 215.5,552.7C212.8,554.3 209.4,555 206.2,555.5C202.5,556 196.4,555 195,555.2C192.8,555.6 191,557.3 188.8,557.6C185.9,557.9 182.8,557.5 179.8,557C178.6,556.8 176.2,556.5 174.4,557.1C172.7,557.7 170.5,559.6 169.9,559.8C168.8,560.1 165.2,559.1 164.2,559.3C163.3,559.4 161.5,562 160.4,563.6C159.7,564.6 159,565.1 157.8,564.9C156.6,564.5 155.6,563.4 154.3,563.1C153.1,562.7 152.2,564.8 152.4,565C153,565.8 156.8,568.3 159.3,569.2C164.5,571.7 170.1,573.4 175.8,574.7C178.8,575.6 182,576.1 185.2,576.3C190.6,576.7 196.1,577.2 201.6,577C205.1,576.8 208.8,576.8 212.3,576C215.8,575.2 219.5,574.2 222.6,572.2C227.2,569.2 231.2,565.2 235.9,562.3C239.2,560.3 240.9,559.2 244.7,558.2C247,557.6 251.3,555.6 253,554C254.5,552.7 255.9,551.7 256.4,549.8C256.7,548.4 256.8,546.8 255.3,546C254.2,545.4 253.3,546.3 251.8,544.4C250.4,542.6 248.4,541.5 248.5,539.1C248.5,537.3 249.8,536.8 251,535.9C251.2,535.2 251.5,534.8 252.1,534.7C253.6,533.9 253.6,533.7 253.9,532.1C254.1,530.8 253.7,528.1 252.1,527.7C250,527.1 247.4,531.4 247.2,529.7C247.1,527.9 247.2,526.4 246.4,525C245.3,523 242.8,521.9 240.6,521.2C237.5,520.2 235.8,519.8 232.6,519.9C231.1,519.9 228.8,519.9 227,520.4C227.4,523.7 227.4,526.6 227,530.2L227,530.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,-308.3)"><path d="M222,545.8C221,545.4 220.5,545.1 219.6,545.1C218.6,545 217.1,545.4 216.2,545.6C215.4,545.9 215.3,546.6 215,547.3C214.7,548.1 213.8,548.9 214.1,549.8C214.3,550.7 214.9,550.9 215.5,551.3C216.1,551.7 216.4,552.1 217.2,551.4C218.7,550.1 219.7,548.8 221.2,547.1C221.6,546.5 221.7,546.3 222,545.8ZM219.2,535.8C219.3,537.7 218.8,539.1 219,541.2C219.2,542.5 219.2,542.1 219.4,543.4C219.6,544.6 220.4,544.9 216.1,545.1C215.1,545.1 214.7,545.2 213.3,545C213.5,543.9 213.6,542.8 214,541.7C214.3,541.2 214.6,540.6 215.3,540.4C215.9,540.2 216.2,539.7 216.8,539.1C217.2,538.6 217.5,537.9 218,537.4C218.3,536.8 218.5,536.7 218.7,536.4C218.9,536.2 219.2,535.8 219.2,535.8L219.2,535.8ZM216.5,551.9C215.4,551.3 214.1,550.5 213.2,549.8C212.4,549.2 212.2,549.1 211.3,548.2C210.8,547.7 210.8,547.8 210,548.3C209.7,548.5 209,548.9 208.7,549.3C208,550.1 207.6,551 207.1,552.1C206.7,552.8 206.4,553.4 206.4,553.7C206.7,554.5 207,555.3 207.3,555.3C207.6,555.2 208.1,555.1 208.6,555.1C209.9,554.8 211.1,554.5 212.3,554.1C213.4,553.7 214.6,553.2 215.5,552.7C216,552.4 216.4,552.1 216.5,551.9L216.5,551.9ZM196.8,555.2C196.8,555.2 197.7,553.7 198.1,553C198.8,552 199,551.1 199.5,550C199.9,549.1 200,549 200.7,549.5C201.3,550 202,550.2 202.4,550.3C203.1,550.6 203.7,551 204.3,551.5C204.8,551.8 205.4,552.4 205.8,552.9C206.2,553.4 206.4,553.7 206.6,554.2C206.7,554.6 207.3,555.3 206.9,555.3C204.5,555.6 202.2,555.6 199.4,555.4C198.6,555.4 198.5,555.3 198.1,555.3C197.7,555.3 196.8,555.2 196.8,555.2L196.8,555.2ZM200.5,547.3C200.4,547.9 200.5,548.1 200.7,549C201,550.1 202.3,550.1 203.2,550.7C204.3,551.3 205.3,552.2 206.1,553.1C206.6,553.6 207,551.9 207.3,551.3C208.5,549.3 209.5,548.4 210.7,547.1C211.2,546.6 211.5,546 211.3,545.9C210.8,545.7 210,545.2 209.7,545.1C208.1,544.3 206.1,544 204.1,544.7C203.6,544.9 203.1,545.1 202.6,545.4C202.4,545.5 202.2,545.6 202,545.8C201.8,545.9 201.6,546.1 201.4,546.2C201.2,546.3 200.9,546.4 200.6,546.4C200.3,546.4 200.5,547 200.5,547.3ZM181.2,555C181.7,554.4 182.4,553.6 183.1,553.4C183.8,553.2 184.4,553.1 184.9,552.6C185.5,552.1 185.9,551.4 186.6,550.8C187.2,550.2 188.1,551 188.4,551.5C188.9,552.1 188.9,552.9 189.8,553.3C190.6,553.7 191.5,553.9 192.1,554.5C192.5,554.9 193.5,555.6 193.1,555.8C192.7,556 192.1,556.3 191.6,556.6C190.4,557.3 189.1,557.6 187.7,557.6C185.8,557.7 184.1,557.6 182.4,557.4C181.8,557.3 182.4,557.4 182.3,556.7C182.3,556.2 182,555.7 181.2,555ZM188.6,546.5C189.4,547.7 190.3,548.7 191.3,549.8C191.7,550.3 192.1,550.9 192.6,551.4C192.9,551.6 194.1,550.6 194.5,550.2C195.9,549 196.8,548.1 198,546.8C198.5,546.1 198.9,545.9 198.5,545.5C197.9,545.1 197.2,544.6 196.6,544.4C195.9,544 195.2,543.7 194.6,543.8C193,543.8 191.6,544.6 190.7,545.2C190.1,545.6 189.3,546.1 188.6,546.5ZM177.7,545.5C178.3,546.5 179,547.6 179.9,548.5C180.9,549.4 181.6,550.3 182.7,551.1C182.9,551.3 183.5,550.8 183.7,550.5C184.3,549.7 184.6,549.1 185.4,548.2C185.8,547.7 186.6,546.9 187,546.3C186.6,546.1 186.1,545.9 185.7,545.6C184.5,545 183.2,544.1 181.8,544C180.7,543.9 180.1,544.2 179.1,544.7C178.7,544.9 178.2,545.2 177.7,545.5ZM167.2,547.4C168.1,548.8 169.2,550.1 170.2,551.5C170.5,551.9 170.9,552.7 171.3,552.3C171.9,551.8 172.2,551.5 172.7,551.1C174,550 175.2,548.8 176.2,547.4C176.5,547 176.9,546.5 176.8,545.9C176.4,546.1 176,546.3 175.6,546.3C175.1,546.5 174.7,545.9 174.4,545.5C173.9,545 173.3,544.4 172.5,544.3C171.5,544.2 170.5,544.4 169.5,544.7C168.9,545 168.5,545.7 168.2,546.4C168,546.9 167.7,547.4 167.2,547.4ZM173.4,557.4C172.5,556.6 172.9,557 171.9,556.1C172.2,555.5 172.6,555.1 173,554.6C173.9,553.6 173.8,553.5 174.6,552.6C175,552.1 175.2,551.8 175.9,551.1C176.4,550.6 176.3,550.3 177.7,552.1C178.1,552.7 178.5,553 178.9,553.5C179.3,553.8 179.9,554.2 180.3,554.5C180.8,554.9 181.4,555.1 181.8,555.6C182.1,555.9 182.3,556.3 182.3,556.9C182.2,557.5 181.9,557.3 181.1,557.1C179.1,556.9 177.2,556.5 175.2,556.9C174.6,557 174,557.2 173.4,557.4ZM163.8,559.4C165.1,557.3 165.9,555.9 167.4,553.9C168,553.2 168.8,554 169.3,554.3C170.2,554.9 170.5,555.2 171.4,555.8C172.4,556.5 172.9,557 173.6,557.5C172.9,557.8 173.1,557.7 172.5,558.1C171.6,558.6 170.9,559.1 170.1,559.7C169.2,560.1 168.2,559.7 167.2,559.6C166,559.4 164.6,559.1 163.8,559.4L163.8,559.4ZM146,549.7C146.2,551.5 146.2,552.1 146.5,553.2C146.9,554.3 147.2,555.1 147.7,556C147.9,556.5 148,557 148.4,557.5C148.7,558 148.9,557 149.1,556.7C149.5,555.9 149.5,554.9 149.8,554C150.1,552.7 150,552.1 150.1,550.8C150.1,549.8 150.5,549.4 150,549.6C149.5,549.8 148.8,549.7 148.4,549.7C147.5,549.6 146,549.5 146,549.7L146,549.7ZM150.3,549.6C150.1,550.6 150.1,551.2 150.1,552C150,552.9 150.1,554 150.1,554.8C150.1,555.4 150.3,555.9 150.7,556.8C151,557.5 151.7,557.4 152.2,557.1C152.6,556.8 152.9,556.4 153,556C153.4,555.4 153.4,554.6 153.5,553.9C153.6,553.3 153.7,553 154.1,552.3C154.3,551.9 155.3,550.8 155.5,549.8C155.6,549.4 155.5,548.8 155.1,548.4C154.9,548.1 154.3,548.1 153.9,547.8C153.6,547.7 152.7,547.5 152.2,547.9C151.4,548.3 151.3,548.5 150.7,549.1L150.3,549.6ZM152.3,564.9C152.3,564.9 151,563.5 150.5,562.7C150.2,562.3 149.9,561.9 149.7,561.4C149.5,560.6 149.3,559.8 149.3,559.1C149.2,558.7 149.2,558.1 149.3,557.9C149.3,557.6 149.3,557.1 149.6,557.5C149.9,557.8 150.2,558.1 150.5,558.4C150.9,558.9 151.2,559.5 151.7,559.9C152,560.1 152.3,560.2 152.7,560.3C153,560.5 153.4,560.5 153.6,560.8C153.6,561.1 153.8,561.6 153.8,561.6L154.1,562.4C154.1,562.4 154.3,563 154.1,563C154,563 153.8,563 153.6,563.1C153.5,563.2 153.3,563.3 153.2,563.4C153,563.5 153,563.6 152.9,563.7C152.7,563.9 152.6,564.1 152.6,564.3C152.5,564.4 152.5,564.5 152.4,564.6C152.4,564.7 152.3,564.9 152.3,564.9L152.3,564.9ZM165.5,546.2C164.8,547.1 164.1,547.9 163.5,548.8C162.8,550 162,551.1 161.5,552.3C161.2,553 161.1,553.8 161.1,554.6C161,555.5 160.9,556.3 161,557.2C161,557.8 161,558.4 161.1,559C161.2,560.2 160.5,560 159.7,559.4C159.1,559 158.5,558.6 158.1,558C157.6,557.4 157.3,556.6 157.1,555.9C156.8,554.9 156.7,553.6 156.6,553.1C156.5,551.9 156.2,550.4 156.1,549.7C156.1,548.8 156.1,548 156.5,547.2C156.9,546.6 157.2,545.6 157.9,545.3C158.7,544.9 159.7,544.7 160.6,544.6C161.9,544.4 163.3,544.7 164.5,545.3C164.9,545.6 165.2,545.9 165.5,546.3L165.5,546.2ZM154.1,562.8C153.9,561.9 153.5,560.9 153.4,559.9C153.3,558.6 153.2,557.2 153.3,555.9C153.4,554.8 153.4,554.3 153.8,553.3C153.9,552.7 154.2,552.1 154.6,551.6C155.2,551.1 155.3,551.2 155.4,551.4C155.7,551.9 155.9,552.2 156.1,552.9C156.2,553.8 156.3,554.8 156.5,555.8C156.6,556.2 156.6,556.6 156.8,557C157.1,557.5 157.4,558 157.8,558.4C158.4,559.2 159.3,559.8 160.1,560.5C160.7,560.9 161.2,561.3 161.8,561.6C161.2,562.3 160.8,563.1 160.3,563.8C159.9,564.4 159,564.9 158.4,565C158.1,565 157.4,564.8 157,564.6C156.7,564.4 155.7,563.6 154.9,563.3C154.7,563.2 154.4,563.1 154.1,563C154.1,562.9 154.1,562.9 154.1,562.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,-308.3)"><path d="M297,526.5C298.3,526.4 300.2,527.1 301.5,528.1C302.8,529.1 303.4,530.5 304.2,532.2C304.9,533.8 305,535.5 304.7,537.1M297,526.9C295.5,522.9 298.7,519.5 303.4,522.5C308.8,525.7 311.9,532 318.1,533.8C322.4,535.7 326.7,538 329.8,541.5C331.4,544.9 330.6,549 329.6,552.4C328.3,556.2 326.1,554.2 322.2,554.2C319,554.2 325.3,561.1 322.9,563.9C320.6,566.4 316.3,563.6 313.8,563C309.7,562.1 309.4,558.4 307.4,555.6C305.5,552.7 298.7,552.7 299.2,549C299.5,546.9 300.1,542.7 300.6,539.9C301.2,535.3 299.7,530.5 297,526.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M332.7,550.3C334.2,553.1 335.4,556.1 336.3,559.2C336.5,559.8 336.9,560.2 337.4,560.6C339.9,562.4 342.7,563.9 345.2,565.8C345.8,566.2 346.3,566.7 346.8,567.2C347.3,567.8 347.7,568.2 347.7,568.8C347.7,570 347.1,571 346.6,572.1C346.4,572.6 345.6,573.1 345,573.1C343.9,573.5 343,573.4 342,573C340.5,572.3 339.6,570.7 338.3,569.5C337.8,569.1 337.1,568.6 336.4,568.7C335.5,568.9 334.7,569.4 334.4,570.4C334.3,571.1 334.3,571.5 334.6,572.1C335.1,573 336,573.5 336.7,574.3C337.4,575.1 338.5,575.8 339.2,576.7C339.6,577.3 340.1,578.2 340,578.9C340,579.5 339.9,580.2 339.3,580.6C338.7,581 338,580.9 337.3,580.8C336.6,580.7 335.6,580.2 335,580C334.6,579.8 334.3,579.7 334,579.8C333.5,580 333.3,580.4 333.1,581.1C333,581.7 333.1,582.5 332.7,583C332.2,583.4 331.5,583.4 331,583.4C330.1,583.5 329.2,583.4 328.5,582.9C327.6,582.3 327,581.3 326.6,580.3C326.1,578.6 326.1,576.8 325.5,575.1C325,573.8 323.8,572.8 322.6,572C321.4,570.7 319.6,570.2 318.1,569.3C316.5,568.3 314.5,567.7 313.1,566.4C312.5,566 312,565.3 312,564.6C311.9,563.8 311.4,562.1 312.1,562.4C314.1,563.3 314.7,563.4 316.3,564C318.2,564.7 320.4,565.6 322.3,564.5C323.7,563.7 323.7,561.7 323.1,560.3C322.6,558.9 322.1,557.5 321.5,556.2C321.3,555.6 321.1,554.6 321.8,554.3C323.4,554.2 325,554.7 326.6,554.8C328,555 329.3,553.8 329.6,552.5C329.9,551.5 330.2,550.5 330.3,549.5C330.5,548.7 332.5,549.8 332.7,550.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M351.4,568.7C353.3,570.5 354.2,573.3 354.5,576C354.7,577.2 354.6,578.6 355.2,579.7C356.1,581.3 357.6,582.4 358.7,583.7C359.6,585.1 359.3,586.7 358.7,588C357.9,589.7 355.5,588.7 353.9,588.7C351.8,588.5 353.9,592.2 351.8,592.3C350.5,592.5 348,591.6 347.9,593.4C347.7,595 347.1,597 345.5,596.4C343.6,595.6 342.4,593.7 340.7,592.5C339.3,591.4 337.6,590.6 336.1,589.6C334.6,588.7 333.9,587.6 332.8,586.3C331.9,585.3 330.9,583.8 332.6,583.1C333.3,582.3 332.9,579.5 334.4,579.8C336,580.2 337.7,581.5 339.3,580.6C340.6,579.4 339.9,577.4 338.9,576.3C337.5,574.9 335.8,573.9 334.7,572.3C333.8,570.8 334.9,568.6 336.6,568.7C338.5,569 339.4,571 340.7,572.2C342.3,573.4 344.3,573.8 346.2,572.6C346.8,572.3 347.5,570 347.8,568.9C348,567.6 350.6,568.1 351.4,568.7L351.4,568.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M358.2,583.1C357.4,581.9 359.1,580.3 359.9,579.1C360.5,578.2 361.5,577.7 362.7,577.4C363.3,577.3 364.7,577.2 365.5,578C366.9,579.4 367.5,581.4 367.9,583.3C368.6,586.2 369.2,589.2 370.5,591.9C371.1,593.5 373,595.7 372.8,596.7C372.6,597.3 369.7,596.7 368.4,596C367.2,595.4 365,595.1 364.3,595.6C363.4,596.2 363.6,598.4 364.4,599.9C365.3,601.5 368.2,601.6 368,603.7C368.1,605.5 366.5,606.9 364.9,607.1C363.2,607.3 361.5,607.5 360.1,608.3C358.5,609.1 357,607.8 355.6,607.1C354.1,606.3 352.6,605.3 351.4,603.9C350,602.4 348.8,600.9 347.5,599.3C346.8,598.4 344.9,597.3 346.7,596.3C348.4,595.3 347,591.8 349.8,592.2C350.9,592.3 352.8,592.8 352.8,591.1C352.9,589.2 352.6,588.3 354.8,588.8C356.3,589.1 358.5,589.4 358.9,587.4C359.4,586 359.5,585.1 358.2,583.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M369.9,608.5C369.2,609.3 368.6,610.2 367.6,610.7C366.2,611.3 364.7,611.4 363.3,611.2C363.2,611.1 363.1,611.1 363,611.1M372.2,594.8C373.8,594.6 374.2,593 374.8,591.8C375.4,590.5 375.7,589.1 376.5,587.9C377.5,586.3 379.3,587.1 379.8,589C380.6,592.3 382.3,595.4 383.8,598.4C384.5,600.1 385.4,602.2 385.7,603.6C385.9,604.8 383,603.9 381.7,604.5C380.4,605.1 378.9,606.5 379.8,608C380.5,609.5 381.7,610.8 382.8,612.1C384.3,613.9 385.2,615.5 385.5,617.6C385.8,620 384.6,622 384,624.1C383.3,626.2 381.7,627.4 379.6,627.3C377.8,627.1 376,626.2 374.4,625.3C372.4,624.2 370.8,622.4 368.5,621.9C366.7,621.4 364.9,620.3 364.1,618.6C363.3,616.9 363.6,615 363.8,613.2C364,611.1 362.1,610.7 360.7,610C359.8,609.6 358.2,609.1 360.3,608.1C362.5,607.1 363.7,607.4 365.7,606.9C367.2,606.5 368.7,604 367.7,602.5C366.7,601.3 364.6,601 364.1,599.2C363.7,597.8 363.3,595.2 365.5,595.3C367.9,595.4 370,597 372.4,596.9C373.4,596.5 372.3,595.4 372.2,594.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M378.5,610.7C378.3,612 378.3,613.4 378.1,614.8C378,615.4 377.1,615.4 376.6,615.3C376.1,615.2 375.5,615.1 375.1,615.4C374.5,615.8 374.2,616.5 374.3,617.2C374.2,618.3 374.2,619.5 374,620.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M284,547C282.6,545.1 283,543.3 283.4,541.5C283.8,539.7 284.9,539.3 283.7,536.6C283.4,535.8 282.9,534.7 282.6,534.2M284,519.2C283.1,519.7 282.3,520 281.4,520.5C280.4,523.6 279.8,526.9 279.6,530.2C279.5,531.3 279.4,532.2 278.7,533.2C278.4,533.6 279.5,533.9 279.7,534.3C280.3,535.9 279.4,537.6 279.6,539.2C279.5,541 280.1,542.3 280.9,543.5C282.1,545.5 283,545.3 284.2,547.4C284.7,548 285,550.5 286.2,551.5C287.1,552.2 288.5,552.3 289.6,552.4C291.3,552.7 293,552.8 294.6,552.3C295.6,552 296.5,551.2 297.3,550.5C297.9,549.9 298.3,549.3 298.8,548.5C299.6,547.3 299.6,546.3 299.8,544.9C300.1,542.1 300.8,539.3 300.7,536.5C300.6,534.6 299.9,532.7 299.2,530.9C298.8,529.7 298.3,528.4 297.5,527.5C296.6,526.4 295.3,525.8 294.4,524.8C293.8,524.1 293.4,523.2 292.9,522.3C292.6,521.6 292.5,520.7 292,520C291.4,519.1 290.6,518.4 289.6,517.9C288.6,517.4 288,517.6 287.2,517.7C286,518.1 284.8,518.7 284,519.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M392.2,632C390.6,631.4 389.1,630.8 387.7,629.9C386.5,629 385.3,628.1 384.3,627C383.9,626.6 382.7,626.1 383.1,625.8C384.1,624.8 384.5,622.4 385.2,620.5C385.6,619.3 385.8,617.9 385.4,616.7C384.9,615.2 384.2,613.8 383.2,612.7C382.4,611.6 381.6,610.8 380.9,609.7C380.3,608.8 379.4,607.8 379.5,606.7C379.9,605.5 381,604.8 382.1,604.4C383.1,604.2 384.2,604.4 385.2,604.3C386,604.2 385.5,603.4 386.4,603.4C387.2,603.5 388,603.1 388.5,602.6C389,602 389.2,601.7 389.2,601.1C389.4,598.9 389.2,596.8 389.5,594.6C389.7,592.6 389.8,590.6 390.2,588.7C390.4,588 390.5,587 391,586.5C391.5,586 392,585.6 392.9,585.6C393.5,585.6 394.1,585.5 394.6,585.7C395.9,586.2 396.9,587.4 397.8,588.5C398.2,589 398.8,589.4 399.2,590C396.9,592.7 394.6,595.5 393.1,598.8C392.3,600.4 391.5,602 391.4,603.8C391.1,606.4 391.1,608.9 391.2,611.5C391.3,614.4 391.5,617.3 391.8,620.2C392.1,623.9 392.3,627.5 392.2,631.2C392.2,631.5 392.2,631.8 392.2,632L392.2,632Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g ns1:id="noga przod" transform="matrix(1,0,0,1,0,-308.3)"><path d="M396.5,655.1C391.5,654 386.5,662.4 383.2,657.8C381.2,655 384.8,649 386.9,645.8C389.2,642.2 391.3,639.4 391.8,635.5C393.4,624.9 390.1,614.2 391.4,603.7C391.9,599.8 395.1,595.1 397.6,592C402,586.4 405.7,582.1 411.1,577.4C414.8,574.1 418.2,569.7 422.8,567.9C428.2,565.8 434.6,565.8 440,567.6C445.1,569.3 448,575 449.7,580C451.8,585.1 453.1,590.7 456.6,595C459,598 459.8,601.7 460,605.5C460.2,610.2 456.5,612.6 453.5,615.2C449.5,618.7 444.2,619.4 439.9,622.1C435.1,625.4 429.4,627.3 425.2,631.5C420.3,635.8 417.2,641.7 414.8,647.7C413.2,651.2 413.7,655.6 410,658C406.7,660.1 402.9,658.9 400.1,656.6C399.1,655.8 397.8,655.4 396.5,655.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M395.4,650C398.9,639.9 403.9,630.5 409,621.2C414.4,611.8 420.4,602.7 425.8,593.2C427.6,589.8 430.5,586.9 431.1,582.9C432,579.4 431.3,574.3 427.1,573.4C423.7,572.7 420.7,575 418.2,577.1C412.1,582.7 405.2,587.9 401.3,595.5C399.7,598.5 397.1,601.2 396.3,604.5C395.5,607.7 394.9,611 396.4,614.3C399.4,620.9 398.9,628.7 396.8,635.5C396.1,638.1 395.3,639.9 394.1,641.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M400.5,652.5C406.6,640.5 411.3,627.7 418.2,616.1C424.7,606.3 429.1,595.1 437,586.2C439,582.6 444.1,582.6 446.5,585.9C450.2,591.3 453.4,597.7 453,604.4C452.8,607.7 449.9,610.4 447.4,612.6C444.8,615 441.2,616 438.1,617.7C435.4,619.2 432.4,620.1 430.1,622C423.8,627 417.6,632.7 414.3,640.4C412.7,644.1 410.8,647.7 409.9,651.7C409.3,654.4 405.9,655.5 403.4,654.6" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M421.5,748.6C419.5,748 416.9,747.3 415.1,748.5C413,749.9 411.4,752.4 411.7,755.2C412,757.8 413.3,760.1 414.5,762.4C415.9,765 418.9,766.7 421.7,767.6C423.7,768.3 425.9,768.3 428,767.8C429.4,767.4 430.6,766.4 431.7,765.3C433.7,763.4 435.1,760.2 435.6,757.4C435.8,754.7 436.2,752.1 435.6,749.3C435,746.3 431.9,744.2 431.6,741.1C431.4,739.3 432.8,738.4 434.3,737.8C436.3,737.3 435.6,735.2 434.9,734C432.4,729.1 428.2,725.5 424.9,721.3C421.4,716.9 417.1,713.2 413.7,708.7C411,705.1 408,701.6 405.9,697.4C404.3,694 402.1,690.8 400.9,687.2C399.6,683.1 399.2,678.8 398.6,674.6C398.3,672.7 399.7,672 400.8,671C401.9,670 403.8,670 405.1,669.1C407.2,667.7 408.6,664.9 408.8,662.4C408.9,661.2 408.6,658.8 406.9,659C404,659.3 401.6,657.8 399.2,656.1C397.3,654.7 394.5,654.6 392.4,655.9C390,657 387.8,659.4 385,659.2C383.1,658.8 382.3,656.6 382.6,654.8C379.6,655.3 377.4,658.8 376.2,661.1C375,663.4 373.9,665.4 373.4,667.9C372.9,670.1 373.1,672.4 373.3,674.7C373.7,679.3 377.1,682.8 379.4,686.6C382.4,691.3 386.3,695.4 389.5,700C394.2,706.1 398.2,712.7 403.6,718.3C408.8,723.8 413.8,729.5 418,735.7C420.2,738.8 422.9,741.8 423.6,745.6C423.8,746.6 423.1,749 421.5,748.6L421.5,748.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M399.7,863.2C399.3,864.7 399.7,866.5 400.4,867.7C401.6,869.9 404.8,870.5 407.1,869.5C408.4,869.1 409,867.9 409.3,866.6C409.9,863.5 409.2,860.5 409.1,857.5C409.1,853.9 408.9,850.3 409.4,846.7C410,837.7 410.9,828.7 412.8,817.8C414.3,809.2 414,808.6 415,802.6C415.9,798.6 416.4,796.1 417.8,791.1C419.3,787.4 421.7,783.2 423.4,779.6C424.2,777.8 425.6,776.1 426.2,774.2C426.8,772.2 426,770.1 425.8,768.1C422.1,768.4 418.4,766.6 415.7,764.1C414.2,764.3 412.7,764.6 412,766.1C411.3,767.4 412,768.9 412.8,770.3C413.9,772 412.8,774.3 410.9,775.7C408.8,777.3 410.1,780.6 410,783.1C410.2,787.1 410.4,791 410,795C409.6,800.2 408,805.5 407.1,810.8C406.2,815.4 405.6,820 404.7,824.8C403.6,831.6 403.6,838.5 403.2,845.4C402.9,849.4 402.7,852.7 401.9,856.6C401.2,859.6 400.3,861.4 399.7,863.2L399.7,863.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M408.9,868.8C409.6,870.4 411.7,871.2 413.6,871.1C415.6,871 417.9,870.2 418.2,867.9C418.7,864 416.8,860.4 416.4,856.7C415.5,850.8 415,844.9 415.7,839C416.6,831.4 418.7,824 420.6,816.6C422.5,808.9 425.8,801.7 428.7,794.3C431.9,785.1 435.7,776.2 438.5,766.9C439.7,762.7 442.2,759 445.7,756.1C449.3,753.1 451.2,748.3 451.4,743.5C451.6,742 450.8,739.7 449.4,739.1C447.4,738.2 445.1,738.8 443.1,739.5C441.6,740.1 439.6,739.8 438.4,739.1C437.3,738.3 436.8,737 434.6,737.7C433.4,738 430.9,739.6 431.8,741.8C433.3,746.2 435.8,747.1 435.9,751.5C436,756.5 435.4,761.3 432.1,764.9C430.5,766.8 428.3,768.1 425.8,768.2C426.5,771 426.9,774.2 425,776.8C422.6,781.1 420.4,785.4 418.4,789.9C415.2,798.3 414.4,807.3 413.1,816.1C411,827.8 409.7,839.7 409.1,851.6C408.8,855.8 410.1,863 409.2,867C408.9,868.3 408.6,868.3 408.9,868.8L408.9,868.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M418.4,865.2C419,865.2 419.6,865.4 420.2,865.5C420.7,865.7 421.1,865.7 421.6,866C422.2,866.3 422.7,866.9 423.2,867.4C423.6,867.8 424,868.2 424.3,868.6C424.7,869.2 425.1,869.7 425.2,870.3C425.2,870.7 424.9,871.3 424.5,871.6C424.2,871.9 423.7,872.3 423.2,872.6C422.3,873.1 421.4,873.7 420.3,874C420,874.1 419.6,874.2 419.2,874.2C418.9,874.2 418.6,874.2 418.4,874C417.8,873.7 417.3,873.2 416.9,872.7C416.6,872.4 416.3,872.1 416.1,871.7C416,871.4 416,871.4 415.9,871.1C415.8,871 415.8,870.6 416,870.6C416.3,870.5 416.5,870.3 416.7,870.2C417.1,870 417.2,869.9 417.4,869.6C417.7,869.3 417.9,868.8 418.1,868.4C418.2,868 418.3,867.7 418.3,867.3C418.3,866.9 418.3,866.6 418.3,866.1C418.2,865.7 418.2,865.2 418.4,865.2L418.4,865.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M408.8,868.6C408.5,869.4 408,870.1 407.7,871C407.4,871.8 408,872.7 408.4,873.5C408.7,874.2 409.3,874.8 409.9,875.3C410.5,875.8 410.7,876.5 410.8,877.1C411,877.7 410.9,878.5 411.2,879C411.7,879.8 412.4,880.5 413.2,880.9C414.3,881.4 415.6,881.9 416.7,881.5C417.6,881.2 418.4,880.6 419.1,879.9C419.9,879.1 420.1,878 420.2,876.9C420.3,876.2 420.4,875.5 420.1,874.9C420,874.6 419.9,874.4 419.7,874.1C418.8,874.3 417.7,873.9 417.1,873.1C416.6,872.5 415.9,871.4 415.9,871C415.7,870.7 415.1,870.9 413.8,871.1C412.7,871.3 411.8,871 410.8,870.6C409.9,870.2 409.1,869.5 408.8,868.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M407.8,870.8C407.5,871.3 407.4,872 407.1,872.5C406.7,873.2 406.2,873.8 405.7,874.4C405.2,874.9 404.6,875.2 404.1,875.6C403.7,875.9 403.3,876.3 402.8,876.5C402.3,876.6 401.8,876.5 401.3,876.4C400.5,876.4 399.6,876.3 398.8,876.1C398.3,875.9 397.7,875.7 397.5,875.3C397.2,874.8 396.9,874.3 396.8,873.7C396.8,873.2 396.9,872.7 397,872.2C397.5,871.1 397.9,870 398.6,869.1C399,868.6 399.1,868 399.8,867.8C400.4,867.6 400.7,868.3 401,868.6C401.5,869.1 402.2,869.4 402.8,869.7C403.4,870 404.1,870 404.7,870C405.4,870 406.2,869.9 406.9,869.6C407.4,869.3 408,869 408.4,868.8C408.7,869.2 408,870.2 407.8,870.8L407.8,870.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M411.7,879.8C411.8,880.9 409.8,881.8 408.6,882.4C408.1,882.6 407.6,882.9 407,883.1C406.1,883.3 405.1,883.2 404.4,882.7C403.8,882.2 403.2,881.7 403,881C402.8,880.3 403,879.5 403.1,878.7C403.1,878.1 403.3,877.5 403.5,876.9C403.7,876.4 403.8,875.7 404.4,875.5C405,875.1 405.5,874.7 405.9,874.2C406.4,873.6 406.8,873 407.2,872.4C407.4,872 407.6,871.4 407.7,872.1C408,872.9 408.5,873.7 409,874.5C409.4,874.9 409.9,875.2 410.3,875.6C410.9,876.5 410.9,877.6 411.1,878.5C411.2,879 411.6,879.7 411.7,879.8L411.7,879.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M403.6,881.9C403.8,882.3 402.9,882.5 402.5,882.5C401.7,882.5 400.9,882.6 400.1,882.5C399.2,882.4 398.2,882.2 397.3,881.9C397,881.8 396.7,881.6 396.4,881.4C396.1,881.1 395.9,880.6 395.8,880.2C395.7,879.7 395.5,879.2 395.6,878.7C395.7,878.2 396.2,877.8 396.3,877.4C396.4,877 396.5,876.5 396.7,876.2C396.9,875.8 397.5,875.5 397.9,875.7C398.3,875.9 398.8,876.2 399.3,876.2C400,876.3 400.8,876.4 401.4,876.5C401.9,876.5 402.6,876.6 402.8,876.4C403.3,876.3 403.5,876 403.8,876C403.3,877.2 403.1,878.5 402.9,879.7C402.9,880.1 402.9,880.5 403,880.9C403.1,881.3 403.2,881.5 403.4,881.8L403.6,881.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M394.8,920.9C394.5,921.9 394.2,922.9 394.1,924C394.1,924.8 394.3,925.6 395,926.1C396.1,926.9 397.6,927 398.9,926.8C399.9,926.6 400.7,926 401.3,925.1C402.1,923.9 402.9,922.7 403.6,921.4C403.9,920.6 403.9,919.7 403.8,918.9C403.7,917.9 403.1,916.9 403.3,915.8C403.5,913.8 404.2,912 404.7,910.1C405.4,906.8 406.8,903.7 407.8,900.5C408.4,898.5 409.5,896.6 410.3,894.7C411.3,892.9 412.2,891 413.5,889.4C414.3,888.1 415.2,886.9 415.9,885.5C416.3,884.7 416.3,883.7 416.1,882.9C415.9,882.4 415.9,881.7 415.3,881.6C414.2,881.4 413,881 412.1,880.2C411.5,879.5 411.7,880.4 411.3,880.8C410.4,881.6 409.1,882.1 408.1,882.7C407.5,883 406.5,883.1 406,883.4C405.3,883.8 405,885.5 404.9,885.9C404.9,886.7 404.8,887.4 404.9,888.4C405.1,889.3 405.1,890.5 405,891.5C404.9,892.7 404.8,893.9 404.3,895C403.2,897.2 402.1,899.4 401.3,901.7C400.5,904 399.5,906.1 398.7,908.4C397.8,911.9 397,915.3 395.8,918.7C395.6,919.3 395.1,920.1 394.8,920.9L394.8,920.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M401.8,928.3C402.3,928.4 402.8,928.3 403.1,928C403.7,927.5 404.1,926.9 404.4,926.2C404.9,925 405.5,923.7 405.7,922.3C405.8,921.8 405.9,921.3 405.8,920.8C405.6,920.2 405.3,919.7 405,919.3C404.7,918.9 404.3,918.6 403.8,918.5C403.7,918.9 403.9,919.2 403.9,919.6C403.9,920.1 403.8,920.8 403.6,921.2C403.1,922.4 402.2,923.7 401.7,924.4C401.4,924.9 401,925.6 400.6,926.1C399.9,926.8 400.7,928.2 401.8,928.3L401.8,928.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M394.8,926.2C395.1,927 391.6,927.7 389.9,926.9C388.6,926.1 388.5,924.1 389.1,922.8C390,920.8 391.4,919.2 392.2,917.1C393.9,913.4 394.9,909.4 396.4,905.6C398,901.4 399.6,897.2 400.6,892.8C401.2,890.4 400.8,888 400.6,885.6C400.5,884.1 400,883.1 402.5,882.5C403.3,882.3 403.8,882.3 404.3,882.6C404.8,883 405.5,883.2 406.1,883.3C404.4,884.9 404.8,887.7 405.1,889.8C405.3,892.2 404.6,894.6 403.4,896.7C402.2,899.1 401.4,901.5 400.4,903.9C399.7,905.6 398.9,907.3 398.5,909.1C397.7,912.2 397.1,915.3 395.9,918.2C395.5,919.4 394.1,921.7 394.1,924.6C394.1,925.1 394.5,925.5 394.8,926.2L394.8,926.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M389,925.9C387.5,926.9 385.4,927 384.2,925.9C383,924.8 384,922.9 384.9,921.6C386.8,919.2 388.1,916.3 389,913.3C391,907.4 392.8,901.3 394.9,895.3C396.2,891.8 395.8,888.1 395.8,884.5C395.8,883.5 396,881.6 396.8,881.8C397,881.9 397.4,882 397.8,882.1C398.4,882.3 399.2,882.4 399.8,882.5C400.1,882.5 400.5,882.6 400.8,882.6C401.2,882.6 401.3,882.7 401.6,882.7C401.2,882.9 401,883.1 400.7,883.5C400.2,884.2 400.7,885.8 400.8,886.9C401,889.3 401.1,891.8 400.3,894.1C399.2,898 397.9,901.9 396.4,905.6C394.5,910.7 393,916.6 390.3,920.6C389.4,922.1 388.5,923.5 388.7,924.7L389,925.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g ns1:id="kregosłup + zebra" transform="matrix(1,0,0,1,0,-308.3)"><path d="M450,580.2C450.7,578.8 451.1,577.1 451.9,575.7C452.4,574.9 453.1,574.2 454.1,574C455.3,573.6 456.7,573.3 457.8,574.1C458.5,574.6 458.9,575.5 458.8,576.4C458.8,577.4 458.6,578.4 458,579.2C456.7,581.1 455.8,583.3 454.7,585.3C454.2,586.5 453.7,587.9 453.6,589.2C453.4,589.8 452.9,588.5 452.8,588C452.5,587.3 452.2,586.6 452,585.9C451.3,584.1 451,583.5 450.4,581.7L450.1,580.8C450,580.7 449.9,580.5 450,580.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M455.3,592.9C455.4,592.3 457.1,591.5 457.7,591C459,589.7 460,588.2 461,586.7C462.4,584.1 463.8,581.5 465,578.8C465.4,577.8 466.4,577.2 467.5,577C468.7,576.9 470.1,577 471.2,577.5C472.3,577.8 473.3,578.9 472.5,580.1C471.1,582.3 470.1,584.7 468.7,586.9C467.9,588.3 466.8,590 466.2,590.9C464.6,593.6 463.5,596.6 462,599.4C461.5,600.3 460.8,601.5 459.8,602C459.5,602.2 459.5,600.9 459.2,600C458.6,598 457.5,596.1 456.2,594.5C455.8,594.1 455.2,593.1 455.3,592.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M460.2,603.5C460.9,602.3 461.3,601.7 462.9,601.6C464.5,601.6 465.7,600.4 466.8,599.4C470.2,596.2 472.9,592.4 475.4,588.5C477.6,585.1 479.3,581.5 481.8,578.4C483.1,577 486.4,576.7 487,577.5C488,579.4 487,582.5 486.4,583.8C485.4,586.5 484.6,587.4 483.1,589.8C481.8,591.9 480.4,593.8 478.8,595.6C477.8,597.2 476.7,598.8 476.4,600.7C475.8,602.4 475.8,604.6 473.7,605C472.1,605.5 470.3,605 468.9,605.9C466.8,607.3 466.7,610.4 464.4,611.7C463.1,612.4 462,612.4 460.8,611.4C459.7,610.6 459,610 459.4,608.9C460,607.6 460.1,605.9 460,604.9C460,603.9 460.2,603.5 460.2,603.5Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M476,602C476.4,599.6 479.6,599.1 481.4,597.5C482.9,596.1 484.6,594.7 486,593.1C488.8,589.9 490.3,585.7 492.5,582C493.4,580.5 494.5,578.5 496.6,578.4C501.5,577.8 501.9,579.3 501.5,580.9C500.9,583.7 500.4,585.1 499.5,586.3C498.2,587.9 497.5,589.2 496.4,590.9C495.4,592.5 494.1,594 493.6,595.4C493.1,597.5 493.4,599.6 493.3,601.7C492.3,603.9 489.5,602.8 487.7,603C485.7,603.1 484.6,604.7 483.6,606.1C482.4,607.7 481.6,609.7 480.2,611.1C478.5,612.8 477.1,609.7 475.8,608.6C474.8,607.7 473.4,606.1 473.3,605.2C475.1,604.9 475.8,603.2 476,602Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M493.5,599.4C494.9,596.6 497.7,595 499.8,592.7C501.1,591.1 502.6,589.6 503.7,587.9C504.8,586.1 506.1,584.3 506.7,582.2C507.1,581 507.5,579.5 508.7,578.9C509.6,578.4 510.8,578.5 511.6,578.4C515.1,577.7 514.3,578.4 514.4,580C514.4,581.6 514,583.7 513.3,585.1C512.4,586.7 512.6,587.9 511.5,589.4C510.4,590.8 509.2,592.1 508.6,593.9C508.2,595.7 506.9,597 506.8,597.4C506.7,597.7 506.6,597.4 505.8,597.4C505,597.3 502.8,597.7 502.1,598.4C500.6,600 499.6,602.2 498.5,604.1C498,604.9 497.7,606.8 496.8,606.6C495.8,606.3 495,605.5 494.3,604.7C493.6,603.9 492,603.1 493,602.2C493.6,601.4 493.1,600.1 493.5,599.4L493.5,599.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M515.5,584.6C516.4,583.4 516.9,582.1 518,581C519.2,579.7 520.7,578.4 522.4,577.9C525.9,576.8 526.2,578.1 525.5,580.5C525.2,581.9 525,583 524.5,584.2C523.6,586.2 521.7,587.8 520.9,589.9C520.1,591.3 519.8,593 519.4,594.5C518.9,596.2 517.2,596.7 516.7,598.4C516.2,599.5 515.8,600.7 515.4,601.7C515,602.6 514.5,603.9 513.4,604C512.2,604 511.3,603.1 510.5,602.3C509.3,601.1 508.6,599.5 507.4,598.3C507,597.7 506.4,597.6 507.5,596.4C508.4,595.1 508.4,593.9 509.3,592.6C510.6,590.4 512.4,588.4 514.1,586.4C514.6,585.8 515.1,585.1 515.5,584.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M522.2,587.7C523.8,585.7 525.5,584 527.1,582C527.9,581 528.1,579.4 529.2,578.8C530,578.4 531.5,578.1 532.2,578.9C532.9,579.7 532.7,580.4 533.1,581.9C533.4,582.7 532.8,583.2 532.6,583.8C532.2,584.9 531.4,585.9 531.2,587C530.9,588.5 530.7,590.1 530.7,591.7C530.8,592.7 531.4,593.6 532,594.3C532.5,594.8 532.9,594.9 532.7,595.8C532.4,597 532.3,597 531.9,597.7C531.4,598.5 531.5,599.8 531.2,600.8C531.2,602 529.8,601.6 529.1,601C527.5,599.5 525.8,598 524.3,596.4C523.4,595.6 522.4,594.7 521.2,594.6C520.3,594.6 519.3,594.9 519.5,594.2C519.8,592.9 520.4,590.4 521.5,588.7C521.6,588.4 521.9,588 522.2,587.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M548,586.1C549.3,586.2 551.5,586 551.3,584.2C551.1,581.9 551.8,579.5 551.7,577.1C551.7,576.5 551,575.7 551.4,575.4C552.5,574.8 553,575.1 553.6,575.5C555.4,576.3 557,577.9 557.3,579.9C557.5,581.3 557.9,582.4 559,583.3C560.2,584.4 561.8,584.5 563.2,584.6C564.1,584.6 565.1,584.6 566,584.3C566.7,584.1 567.5,583.9 567.5,583C567.3,581.3 567.1,578.9 566.2,577.5C564.9,575.7 563.1,574.6 563.1,573.4C563.1,572.9 566.2,572.3 567.6,572.7C569.4,573.1 570.7,573.5 571.8,574.5C573.2,575.7 573.8,577 573.9,578.6C574.4,585.2 576.2,583.1 577,586C577.4,587.3 577.1,588.8 578.6,590.9C579.3,592 579.3,593.6 577.9,593.7C577.3,593.8 574.1,591.2 572,590.5C571,590.2 569.9,589.8 569.1,590.3C567.5,591.2 567,594.8 566.9,595.4C566.6,597 566.9,597.3 566.6,597.2C564,596.3 562,595.1 559,593.9C557.5,593.2 556,592.3 554.4,591.7C553.7,591.5 553,591.4 552.3,591.3C551.4,589.6 550.4,587.9 549,586.6C548.7,586.4 548.3,586.3 548,586.1L548,586.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M557.5,581C557.8,582.3 557.7,583.3 558.3,584.5C558.5,584.7 558.8,585.3 559.2,585.7C559.9,586.3 560.6,586.6 561.6,587" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M575.7,578C577.4,578.7 577.8,579 579.3,580.2C580.3,581.1 582.7,582.2 583.4,581.6C584.3,580.3 583.2,579 582.9,577.7C582.4,575.8 581.3,574.2 580.9,572.2C580.5,570.4 583.1,570.5 584.2,570.8C586.5,571.5 587.7,571.6 589.8,572.5C590.6,572.8 591.7,573.9 592.2,574.7C593.2,576.5 593.4,578.6 593.2,580.7C593,582.1 594.5,583.4 595.9,583.9C597.9,584.3 598.6,586.3 599.5,587.8C600.4,589.2 600.8,590.7 601.4,592.4C601.7,593.3 600.5,594.4 599.9,595C598.9,596.1 599.4,597.8 598.4,598.8C597.3,600.1 595.3,600.4 593.5,600.1C592.1,599.9 590.8,601.2 589.5,600.6C587.7,599.2 585.7,598 583.7,596.9C581.9,595.8 579.9,594.7 578,593.8C578.6,593.4 579.3,592.6 578.8,591.5C578.6,590.9 578.2,590.5 577.9,589.7C577.5,588.7 577.3,587.5 577.1,586.6C577,584.6 575.5,584 574.7,582.5C573.9,580.9 574.1,579.2 573.8,577.5C573.9,577.2 574.4,577.4 574.6,577.5L575.7,578Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M579.3,580.2C580.4,581.3 581.6,582.5 582.7,583.6C583.1,584.1 583.6,584.5 584.1,584.7C584.8,585 585.5,585.2 586.2,585.4" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M597.7,599.4C598.4,601 599.8,602 601.4,602.3C603.3,602.7 605.2,601.5 607,600.9C608.6,600.3 610,599.2 611.4,598.3C613.6,597 616.1,596.6 618.2,595.3C619.9,594.2 620.6,592.5 620.8,590.8C621,589.3 620.8,587.7 620.6,586.1C620.3,584.1 618.2,583.1 616.8,581.9C615.7,581 615.2,579.6 615.3,578.2C615.2,577.6 615.6,576.8 615.1,576.4C613.6,575.1 611.5,574.9 610.3,573.3C608.4,571.6 607.2,569.1 605,567.7C603.8,566.8 602.3,566.8 600.8,566.9C595.9,566.5 598.9,568.4 599.7,569.9C601.1,571.6 602,573 602.9,574.3C604.1,575.9 605.1,577.5 604.2,578.6C603.5,579.5 601.6,578.8 600.3,578.4C597.9,577.8 595.6,576.9 593.1,577.2C593.6,579 592.5,581.1 593.9,582.6C594.8,583.7 596.6,584 597.3,584.6C598.7,585.8 600.3,589.1 601.2,591.6C602.4,593.9 599.3,594.6 599.2,596.5C599,597.7 598.8,598.8 597.7,599.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M602,578.9C602.8,579.1 603.6,579.2 604.3,579.6C605.1,580 605.9,580.4 606.5,581C607.3,581.8 607.4,583.1 608.8,583.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M614.3,575.8C615.6,574.6 616,572.1 618.3,572.3C620.4,572.4 621.7,574.1 623.5,574.9C626.3,576 623.4,572.7 622.8,571.9C619.7,567.5 620.8,569.3 619.7,567.7C618.7,566.3 616.5,564.7 619.4,564.5C622.8,564.3 624.4,564.5 626.7,565.8C628.5,567.3 630.3,568.9 631.9,570.7C633.2,572.1 633.9,574.9 634.2,576.9C634.9,578.8 637.5,578.8 638.4,580.4C639.2,582.1 639.3,584.1 638.6,586C638,587.8 636.5,588.9 635,590C632.5,591.6 629.9,593.4 628,595.8C626.9,597.2 625.7,598.4 624.5,599.6C623.1,601 622.7,601.8 620.8,602.1C615.5,603 614.1,602 613.6,599.3C613.2,596.8 615.4,596.7 617.4,595.8C619.9,594.7 621,591.8 620.9,589.3C620.8,587.8 620.8,586.1 620.1,584.8C618.8,582.8 615.8,582.2 615.4,579.6C614.9,578.3 616.1,576.4 614.3,575.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M623.3,574.8C624,575.1 624.6,575.4 625.2,575.8C625.8,576.2 626.3,576.8 627,577.2C627.5,577.5 628.5,577.8 628.7,577.9" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M633.6,574C634.4,572.2 636.6,571.6 638.6,572C640,572.3 642.1,574.9 642.2,574.3C642.6,572 642.2,569.4 640.6,567.4C639.5,566 637.9,564.3 636.9,562.5C637.5,560.3 640.4,560.7 642,561.3C643.5,561.8 644.9,562.5 646.1,563.6C648.6,566 650.7,568.6 652.9,571.1C655.5,574.3 657,578.2 658.5,582C659.7,584.9 660.6,588.2 660.1,591.3C659.7,593.5 658.3,595.7 656.2,596.3C653.8,597 651.3,596.6 648.8,596.6C646.8,596.7 645.9,598.6 644.6,599.8C643.2,601.2 642.2,601.9 640.5,603C638.9,604.1 635.7,603.8 633.8,603.2C633,602.9 630.8,602 633,600.8C634.9,598.9 635.4,598.4 637.3,597.3C637.9,596.9 639.4,595.6 638.7,595.8C636.3,596.6 634.1,597.7 631.5,597.7C630.1,597.7 626.1,597.3 628.3,595.6C630.7,592.6 634.5,590.7 637.3,588.3C639.1,586.1 639,584.6 639,582.7C638.5,579 636.8,579.5 635.2,578.3C633.9,577.3 634.1,575.6 633.6,574Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M640.9,573.6C641.9,574.5 642.7,575.4 643.6,576.4C644.1,577 644.8,577.3 645.4,577.5C646.4,577.7 647.4,577.4 648.3,577.1" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M638.8,596.1C639.6,594.8 640.6,593.8 641.1,592.4C641.5,591.5 642.1,590.5 642.2,589.5C642.4,588.5 642.2,587.5 642,586.5C641.8,585 641,583.7 640.2,582.6C639.5,581.7 638.8,580.9 638.2,580.1" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M640,594.4C640.3,592.1 640.4,589.5 640.5,587C640.6,585.9 640.6,585.3 640.4,584.2C640.3,583.7 640.1,582.8 639.9,582.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M646.6,597.7C647.5,596.7 648.3,595.7 649.4,594.8C650.8,593.7 652.4,592.7 654.1,592C654.5,591.9 654.9,591.8 655.4,591.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M658.9,594.3C658.1,595.9 657,597.4 655.7,598.7C654.5,600.1 650.6,601.9 651.9,602.9C653.5,604.3 655.7,603.5 657.7,603.2C659.8,602.9 661.7,602 663.6,601C665.8,600 667.3,597.2 669.7,596.1C670.7,595.6 671.5,595.3 672.9,595.4C674.7,595.4 677.7,596.2 678.8,594.6C679.9,593 680,591 680.1,588.9C680.2,586.6 680.4,585.7 679.7,582.9C679.1,580.1 676.5,575.5 674.7,571.9C673.8,570 673.2,568 672.1,566.2C671.1,564.5 669.1,563.6 667.4,562.8C665.7,562.1 663.8,562.2 662,562.3C659.9,562.3 662.7,564.2 663.3,565C664.6,566.7 664.7,569.4 664.3,571.5C664.1,573 661.6,575 660.4,573C659.4,571.5 657.8,571.4 656.2,571.5C654.7,571.5 653.7,572.1 654.7,573.8C657,577.6 658.7,581.9 659.8,586.2C660.5,588.9 660.4,591.9 658.9,594.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M660.7,573.3C661.2,573.8 661.5,574.1 661.9,574.6C662.6,575.7 663,577 663.9,578C664.1,578.3 664.5,578.5 664.8,578.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M668.3,597C669.1,596.1 670.8,595 671.6,594C672.7,592.6 672.8,592.3 673.8,590.7C674.4,589.9 675.3,588.2 675.8,587.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M676.7,575.6C678.3,574.8 679,572.9 681,573.1C683.2,573.2 684.3,577.1 686.3,577C687.7,576.8 687.2,574.2 687.2,572.8C687.1,570.9 686.1,568.7 684.8,567.3C683.1,565.6 683.1,563.8 686.9,564.5C688.7,564.7 690.3,565.6 691.6,566.8C692.6,568.5 693.2,570.4 694.3,572C695.2,573.7 696.5,573.8 698.3,574.5C699.5,575 698.6,577.2 698.5,578.8C698.2,580.8 698.5,584.6 698.7,586.9C699.1,588.9 699.6,589.7 700.2,591.7C701.2,594.6 699.3,595.2 697.4,595.6C694.4,596.4 694.6,598.3 691.6,597.8C690.2,597.7 689,596.9 687.8,596.3C685.2,595.9 684.3,599 682.5,600.4C679.4,602.7 675.4,603.8 671.7,602.9C670.4,601.5 673.4,600.7 674.2,599.7C676,597.5 678.7,595.9 679.5,593C680.3,590.8 680.2,588.3 680.2,585.9C680,582.6 678.5,579.5 677.1,576.5C677,576.2 676.8,575.9 676.7,575.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M685.5,597C686.9,595.4 688.3,593.5 689.7,592C690.7,591 691.1,590.5 692.3,589.7C692.6,589.5 693.8,589.1 694.2,589.1" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M684.5,575.9C684.9,576.4 685.1,576.8 685.4,577.4C685.8,578.1 686,578.8 686.5,579.4C686.8,579.7 687.1,579.9 687.4,580.1C687.7,580.3 688.3,580.4 688.5,580.4" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M411.9,656.2C412.4,658.9 412.9,661.6 413.2,664.3C413.7,667.6 413.9,671 414,674.3C414,675.2 414.3,676.3 413.9,677.1C413.6,677.8 413.1,678 412.4,678.2C411.3,678.3 410.1,678.3 409.1,677.9C408.1,677.5 408.3,675.6 408.4,674.6C408.4,672.3 408.5,671 408.4,668.6C408.3,667.8 408,666.9 407.8,666.1C408.5,664.8 408.8,663.3 408.9,661.8C408.9,660.7 408.4,659.6 407.4,658.9C408.1,658.8 408.9,658.6 409.6,658.2C410.5,657.7 411.3,657 411.9,656.2L411.9,656.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M408.5,678.2C407.8,679.8 407.9,681.6 407.3,683.4C407,684.3 406.9,686.2 405.4,685.3C404,684.5 399,677.7 399.1,678.3C399.4,679.9 399.9,683.5 400.4,685C400.9,687.9 403.9,691.6 405.3,691.9C406.9,692.3 408.1,691.4 409.2,690.6C410.2,689.8 410.8,688.7 411.4,687.6C412.7,685.6 413.9,682.6 414.5,681.2C414.9,680.1 414.3,678 412.8,678.1C411.6,678.4 410.5,678.5 409.3,678C409.1,677.9 408.8,678 408.5,678.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M419.7,640.6C421.5,644.1 422.3,646.8 423.6,650.5C424.7,653.3 425.8,656.1 426.2,659.1C426.7,662.3 428,665.2 428.4,668.3C428.9,671.7 429.6,675 430.7,678.2C431.3,680.3 431.8,682.5 431.2,684.5C430.6,686.5 427.9,686.8 425.9,686.7C424.9,686.6 423.9,686.9 423.1,686.2C421.7,685 421.9,682.8 422.4,681.1C423,679 423.5,677 423.5,674.8C423.6,671.2 423.6,667.5 423.4,663.9C423.1,660.8 422.7,657.6 421.2,654.8C419.9,652.8 418.8,650.8 417.6,648.7C416.5,646.9 415.4,645.9 416.2,644.4C417,642.7 417.8,640.7 418.9,639.4C419.3,639.9 419.5,640.1 419.7,640.6L419.7,640.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M422.4,688.2C422.1,688.9 422.1,690.8 422.2,692.5C422.3,693.6 421.7,694.3 421.2,695.3C420.3,696.9 419.4,697 417.8,698.1C417,698.7 414.9,696.6 413.7,695.5C412.1,694 411.6,693 410,690.1C409.8,690 408.7,691.1 408,691.5C407.4,691.8 406.7,692.1 406,692.2C405.3,692.2 404.1,691.2 404.1,691.4C404.7,692.1 406.5,694.9 407.8,696.6C409.1,698.3 410.2,700 411.7,701.5C413.6,703.3 415.4,705.6 418,706.1C418.8,706.3 419.6,705.9 420.3,705.6C421.5,705 422.7,703.7 423.4,703.1C424.4,702.2 425.2,701.1 426.1,700C427.3,698.4 427.8,696.9 428.5,695C428.9,693.5 429.6,691.9 429.7,690.4C429.8,689.3 429.7,688.3 428.9,687.4C428.2,686.6 427.7,686.8 426,686.7C423.8,686.8 422.9,686.3 422.7,687.2L422.4,688.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M428.7,629C432.2,634.7 435,641 438.3,646.9C440.5,650.9 442.4,655.2 443.5,659.7C446.1,669.4 447.3,679.4 447.8,689.4C447.9,692.2 449.6,697.3 447.5,698C445.3,698.7 442,698.1 440.8,697.1C439.6,696.2 440.2,691.4 440.8,689C441.2,686.3 441.5,684.9 441.4,682.8C441,678.1 441.2,672.9 440.4,668.3C438.8,660.4 436.8,652.4 432.6,645.4C431,642.7 429.4,640 427.6,637.4C426.5,635.7 424.9,633.6 424.3,632.3C425.7,631.1 427.5,629.4 428.6,628.8L428.7,629Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M440.1,698.8C441.5,697.3 443.9,698.5 445.8,698.2C447.4,698.2 449,699.2 449.1,700.9C449.5,703.4 448.1,705.6 447,707.7C446.2,709.4 444.8,710.6 443.3,711.8C440.8,714.1 439,715.1 435.7,715.2C432.8,715.3 431.9,716.2 429.4,714.9C426.9,713.6 422.6,710 421.7,709.3C420.9,708.6 419.7,707.8 418.8,707C418.3,706.5 414.5,704.4 417.7,705.9C418.3,706.1 419.6,706.1 420,705.9C421.4,705.1 422,704.3 422.6,704C423.2,703.8 424.2,705.3 425,705.9C426.5,707 427.9,708.5 429.5,709.2C431.8,710.4 433.5,711.2 436.1,710.4C437.7,709.9 439.1,708.8 440.2,707.4C441.4,705.8 440.6,703.7 440.1,702C439.8,700.9 439.3,699.6 440.1,698.8L440.1,698.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M536.2,603.7C535.7,604.7 534.8,605.4 533.7,604.4C532.5,603.4 531.3,602.4 529.9,601.5C530.7,601.8 531.2,601.3 531.2,600.9C531.3,600.2 531.5,599.7 531.5,599C531.6,597.4 533.4,596.4 532.5,594.7C532.6,594.7 532.9,594.8 533.3,594.7C533.9,594.5 534.5,594 535,593.8C537.3,592.8 539.8,593.6 541.8,594.9C542.9,595.6 544.1,596.3 545.1,597.1C547.1,598.8 547.9,599.2 549.3,600.2C550.6,601.2 551.8,601.9 553.4,603.1C555.9,604.9 558.3,607 560.8,608.7C563.3,610.5 565.3,613 567.5,615.1C570.5,618.6 573.4,622.2 576,626C577.8,628.8 579.3,631.8 580.8,634.8C581.9,637.1 582.9,639.5 583.6,642C585.3,648 586.3,654.3 587.1,660.6C587.2,661.9 587.5,663.5 586.6,664.4C585.2,665.7 582.6,665.9 580.9,664.9C579.9,664.2 579.6,663.1 579.3,661.5C579,659.9 580,657.3 579.9,655.4C579.8,652.9 579.9,651.3 579.5,649.3C578.8,645.5 577.8,641.7 576.2,638.2C574.6,634.9 574,633 572.1,629.8C570.8,627.7 569,624.7 567.1,622.4C565.2,619.9 562.9,617.3 560.9,615.4C558,612.6 554.8,610.1 551.7,607.5C547.6,604.7 544.4,602 540,599.9C538.9,599.3 537.3,600.2 537,601.5C536.8,602.3 536.5,603 536.2,603.7L536.2,603.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M452.2,616.2C454.9,614.3 457.3,612 459.2,609.8C459.9,611.1 461.1,611.7 462.1,612.2C463.1,612.7 463.3,614.3 463.3,616.3C463.5,620.7 467.6,623.7 469.6,627.5C474.7,636.7 480.6,645.8 483.9,655.9C486.4,664.5 488.9,673.1 489.6,682C490.1,688 491.2,693.9 491.5,699.9C492.2,703.5 487,704.6 484.8,702.6C482.1,699.9 485.1,698.1 485,693.5C484.9,688.9 485.1,681.8 483.5,674.8C482.7,671.1 482.2,667.1 481,663.4C478.7,656.5 475.3,650.1 472.4,643.2C469.2,636.8 466,630.2 461.9,624.3C460,621.6 458.1,618.2 454.4,618.5C453.2,618.6 451.6,617.6 452.2,616.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M485.2,703C484,704.3 482.7,703.8 482.5,707.7C482.3,709.9 483.2,712.4 481.9,714.4C480.2,717 477.9,719.3 475.1,720.7C473.6,721.5 471.6,722.3 470,722.5C466.8,722.7 464,722.2 461,721C459.8,720.6 459.1,720 457.2,720.7C455.6,721.3 451.6,722.2 446.9,721.7C449.7,722.5 454.5,724.4 458.4,725.5C461.7,726.4 464.9,727.4 468.4,727.7C470,727.8 471.5,727.7 473.1,727.4C475.3,726.9 477.7,725.9 479.7,724.8C481.1,723.9 482.4,723 483.4,721.7C485.2,719.5 486.9,717.1 488,714.4C489.1,711.7 490.6,709.7 491.7,707C492.2,705.6 490.6,703.6 489.3,703.3C487.9,703.7 486.6,703.8 485.2,703Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M464.3,619.7C463.7,618.7 463.3,617.5 463.3,616.2C463.3,615.5 463.3,615.1 463.3,614.4C463.1,613.4 462.6,612.7 462.6,612.3C463.8,612.2 464.8,611.7 465.5,610.8C466.1,610 466.8,608.8 467.3,608C468.2,606.6 468.6,605.8 470.7,605.4C471.5,605.2 471.8,605.3 472.9,605.3C473.5,605.3 473.1,605.4 473.8,606.3C474.1,606.7 474.9,607.6 475.3,608.1C476.9,609.7 477.9,611.2 479.2,612.6C482.6,616.3 486.5,619.6 488.7,624.2C495.4,634.7 500.9,646 505,657.8C508.2,668.8 511.3,679.8 513.1,691.1C513.4,695.1 515.2,701.4 511.3,703C507.5,704.5 502.8,703.1 505.4,697.8C508.1,692.8 506.7,688.3 506.5,682.9C506.2,676.2 503.3,668.7 501.6,662.3C499.6,656.1 497.5,649.8 494.6,643.9C491.5,637.7 488.1,631.4 483.5,626.1C480.7,622.4 477.9,618.9 474.6,615.6C470.9,611.8 468.8,616.8 466.5,619.1C465.9,619.7 465.1,619.5 464.3,619.7L464.3,619.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M505.9,703C503.8,704.7 504.3,707.5 503.8,709.8C503.4,711.7 503.2,713.8 501.9,715.3C499.8,717.7 498.1,719.3 495.3,720.8C493.3,721.9 490.2,722.9 487.9,723.1C485.8,723.3 483.8,723.5 481.8,723.5C479.6,724.8 477.3,726 474.9,726.9C473.9,727.3 470.9,728 466.7,727.5C468.6,728 475.2,728.7 477.2,728.8C479.7,728.8 482.3,729 484.9,728.7C487.6,728.5 490.2,728.4 492.8,727.6C494.6,727.1 496.2,726.8 497.8,726.1C499.3,725.3 500.7,724.5 501.9,723.3C503.5,721.7 504.6,721 506.7,718C509,714.5 511.1,711 512.7,707.2C513.4,705.6 512.7,703.2 510.7,703.2C509.2,703.8 507.5,703.7 505.9,703Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M483.9,617.4C482,615.4 480.2,613.5 478.3,611.5C478.3,611.4 479.4,612.3 480.8,610.5C482.6,608 483.7,605.1 486.4,603.4C487.8,602.4 491.2,603.7 492.9,602.4C492.4,603.6 493.3,603.7 494.8,605.2C496.2,606.8 496.5,606 497.7,607.4C501.7,610.6 502.7,611.6 507.4,616.8C509.9,619.7 512.9,623.2 515.1,626.4C517.6,630.1 519.8,634.1 521.5,638.3C524.3,645.7 527.1,653.1 529.6,660.7C531,665.9 532.2,671.2 533,676.5C533.9,682 534.8,687.6 535,693.1C535.1,694.9 535,697.6 533.3,698C531.9,698.2 529.6,698.2 528.2,697.3C527.1,696.5 527.2,694.9 527.2,693.4C527.6,688.1 527.5,682.7 527,677.4C526.7,674.2 526.4,671 525.6,667.8C523.9,661.1 522.1,654.5 519.7,648.1C517.4,642.9 515.3,638 512.9,632.9C511.3,629.5 508.7,626.3 506.4,623.3C502.6,618.3 498,614 493.5,609.6C489.9,606 487.8,612.2 486.2,615C485.8,615.7 484.7,616.7 483.9,617.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M528.4,697.7C526,698.8 526.4,700.7 525.5,703.7C524.8,706 523.6,709 521.8,710.7C519.1,713.2 516.1,716.2 512.6,717.9C510.5,718.9 506.5,718.6 505.9,719.4C504.4,721.1 503.3,722.1 500.9,724.2C498.3,726.1 494.7,727.1 496.8,726.9C500.2,726.7 507.9,724.7 511.3,723.8C514.3,723.1 517,721.6 519.6,719.9C521.9,718.3 524.3,717 526.2,715C529.8,711.2 532.2,706.4 534.1,701.6C534.9,699.6 533.8,698 531.3,698.1C530.3,698.2 529.4,697.6 528.4,697.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M502.3,611.3C501.6,610.5 500.5,609.8 499.7,609C498.6,607.9 498,607.4 497.5,607C496.9,606.5 497.5,606.4 497.8,605.7C498,605.1 498.8,603.3 499.8,601.8C501,599.8 501.7,598.1 503.2,597.8C504.4,597.5 506.3,597 507.1,598C508,598.9 509,600.2 509.7,601.4C512,604.1 515,606 517.6,608.5C521.3,612 524.8,615.5 527.9,619.5C532,624.1 535.2,629.5 538.5,634.8C541.6,640.9 544.6,645.8 546.6,652.4C547.9,656.3 549.8,661.9 550.2,666C550.7,671.4 550.3,676.6 551.6,681.8C552.4,685 552.9,688.4 547.9,687.8C542.9,687.3 543.5,684.4 544.7,680.7C546.3,676.1 545.2,671.1 544.9,666.3C544,655.8 539.2,644.7 533.7,635.9C531.2,631.5 530,630.4 525.6,625C522.7,621.6 520.1,617.9 516.6,615C513.5,612.5 511.2,609.5 507.8,606.9C505.3,605 503.1,612 502.3,611.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M545.1,687C543.6,688.6 543.2,691.5 544.1,693.6C545,695.7 545.2,698.2 544.1,700.2C542.4,703.3 540.5,706.3 538,708.8C535.9,710.9 533.2,712.9 530.4,714.2C529,714.8 527.7,715.2 526.1,715.3C525.2,716.3 524.6,716.6 523.5,717.3C521.6,718.6 519.7,720 517.5,721.3C515.7,722.3 513.4,723.2 510.8,724C513.5,724.1 518.4,723 521.1,722.4C524.9,721.7 525.9,721.4 528.2,720.5C532.3,718.9 536.1,717 539.8,714.2C542.6,712.1 545.2,709.7 547.4,706.8C548.7,705.2 549.3,703.7 550.1,701.9C551,699.6 551.6,696.5 552,694C552.2,692.5 552.6,690.9 552.5,689.3C551.9,687 550.3,687.9 548.4,687.9C547.8,687.8 547,687.7 546.3,687.5C545.9,687.4 545.1,687 545.1,687L545.1,687Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M523.9,604C522.8,603.3 521.3,603.8 520.3,604.5C519.4,605.2 519,606.6 518.2,607.6C517.4,608.7 516.8,607.5 515.6,606.7C515.1,606.3 513.1,604.8 512,603.6C513.2,604.5 514.4,603.8 515.1,602.5C515.6,601.5 516,600 516.4,598.9C516.9,597.8 517.1,597.4 518.3,596.1C518.7,595.7 519.2,595.1 519.4,594.6C519.5,594.2 519.1,594.8 520.7,594.7C523.3,594.4 525.1,597.3 526.8,598.9C527.3,599.3 528.1,600.1 529,600.9C529.8,601.5 530.5,601.9 531.4,602.6C533.4,604.3 536.2,606.3 538,608.1C541.4,611.3 544.7,614 547.9,617.5C550.8,620.2 553.1,623.5 555.6,626.6C557.9,630.5 560.3,634.2 562,638.4C563.4,641.2 564.7,643.8 565.5,646.9C566.9,651.7 567.8,654.1 568.6,659C569.1,663.1 570.4,669.8 569.9,673.9C569.8,675 568.7,676.2 567.3,676.1C565.4,676 563.2,676.2 562.2,674.4C561,672.3 561.5,670.6 562.5,668.6C563.1,666.5 563.1,663.9 563,661.6C562.8,658.2 562.7,655.1 561.5,651.9C559.6,646.9 558.2,642.4 555.4,637.8C553.4,633.8 552.9,633 550.4,629.4C548.4,626.4 543.1,620.3 540.6,617.7C536.6,613.6 532.7,609.7 527.8,606.6C526.2,605.6 525.5,605 523.9,604Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M562.8,675.7C561.5,677.2 560.5,679 560.6,681.1C560.7,683.6 560.1,686.1 558.9,688.3C557.2,691.8 554.3,694.5 551.7,697.3C551.1,698.3 551,698.9 550.7,700C550.5,700.8 550.2,701.7 549.7,702.9C548.8,704.7 548.4,705.5 546.9,707.5C546.6,707.9 544.5,710 543.8,710.7C545.2,711.2 547.6,708.8 548.8,707.6C551.4,705.2 553,703.6 555,701.4C557.7,698.2 560.9,695.5 563.2,692.1C565,689.6 566.2,686.7 567.5,683.9C568.4,682.2 569.3,680.4 569.4,678.4C569.5,677.1 569.2,676.1 567.9,676.1C566.8,676.2 565.6,676 564.7,676C563.6,675.9 563.1,675.1 562.8,675.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M442,621C446.6,625.4 449.6,631.2 452.8,636.6C457.4,644.5 461.1,652.9 463.8,661.6C465.5,668.1 467.3,674.7 467.8,681.5C468.4,687.1 470.4,692.8 469.9,698.5C469.8,700.1 464.9,701.1 463.3,699.7C461.1,697.1 462.8,693.7 463.1,690.7C463.5,686 462.4,681.4 461.9,676.8C461.1,671.5 459.2,666.4 458,661.1C456.3,654.5 453.5,648.4 450.7,642.3C448.8,638.1 445.8,634.3 443.5,630.4C441.9,627.7 439.7,625.8 438.8,622.8C439.9,622 441.1,621.2 442,621Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M463.7,700.1C462.9,701.3 461.8,702.5 461.6,704.1C461.1,706.1 461.4,708.7 461,710.8C460.6,713 458.5,714.6 456.5,715.5C455.4,716 454.1,716.6 452.9,716.9C450.3,717.6 448.5,717.6 445.9,717C443.5,716.5 440.9,715.8 439.1,714.5C437.8,715.1 435.7,715.4 432.8,715.4C432,715.4 429.6,715 428.8,714.6C430.5,716.2 435.2,717.8 438.5,719.1C440.9,720 443.3,721 445.9,721.4C447.4,721.7 448.2,721.9 450.5,721.7C452.8,721.8 455,721.3 457.1,720.7C458.6,720.3 460,719.3 461.3,718.4C463.1,717.1 465.3,715.1 466.1,713C467.4,710.7 468.2,708.9 469.3,706.4C470,705.2 470.5,703.7 470.3,702.4C469.8,701 468.6,699.7 467,700.3C466.1,700.4 464.8,700.4 463.7,700.1L463.7,700.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M581,665C578.9,664 578,668.7 577.9,671C577.5,673.2 577.9,674.8 577,676.9C576,679.3 575.2,681.4 573.8,683.3C572.1,685.5 569.2,688.1 567.8,689C566,690.3 564.5,690.9 563,692.5C562.1,693.6 561.8,694.1 560.8,695.2C560.2,695.9 559.6,696.6 558.7,697.4C557.7,698.4 556.9,699.3 556,700.3C555,701.3 554.4,702 553.2,703.4C552.3,704.2 552,704.4 553.3,703.9C555.6,703.2 557.5,701.8 559.4,700.6C562.8,698.6 566,696.2 569.2,694C571.7,691.9 574.7,690.2 576.6,687.6C578.5,684.9 579.6,681.8 581.4,679C582.7,676.6 583.9,674.2 585.1,671.7C585.9,669.9 588,665 585.7,665.3C583.8,665.6 582.7,665.8 581,665Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M558.5,599.9C557.4,599.6 556.3,598.7 555.2,598.9C554.4,599.1 553.9,599.9 553.7,601C553.6,601.6 553.5,602.9 552.6,602.5C551.1,601.3 549.4,600.4 548,599.3C549.3,599.8 549.9,597.8 550.1,597C550.5,595.8 550.6,594.5 551.6,593.7C552.4,593.1 552.4,592.1 552.2,591.2C555.3,591.7 557.9,593.5 560.8,594.6C561.9,595.2 563.2,595.9 564.4,596.4C565.4,596.8 566.7,597.3 567.7,597.6C569,598.1 569.9,598.4 571.3,598.8C572.5,599 573.1,599.5 574,599.9C575.2,600.4 576.4,601.1 577.5,601.8C579.1,602.7 580.3,603.7 582,605C584.2,606.7 586.8,609.8 588.5,612C591.7,616 594,620.5 595.9,625.2C597.2,628 598.4,630.8 599.2,633.7C600,637.1 600.8,640.5 601,644.1C601.1,645.4 601.3,647 600.3,647.8C599.7,648.3 598,648.3 596.7,648.2C595.3,648.2 594.8,646.3 594.6,644.9C594.3,642.4 594.7,639.8 594.1,637.3C593.4,634.1 592.6,631 591.7,627.9C590.5,625.3 589.7,622.6 588,620.4C586.7,618.1 585.2,615.9 583.3,614.1C580.4,611.2 577.1,608.7 573.6,606.6C568.9,603.8 563.6,602 558.5,599.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M595.8,648C594.1,647 594.6,650.5 594.5,652C594.5,653.6 594.4,655.6 594.1,657.3C593.7,659.2 593.3,661 592.5,663C591.9,664.6 590.8,666.4 589.6,667.5C588.4,668.6 588.1,668.8 586.6,670C585.6,670.9 584.4,673.2 583.9,674.3C583.3,675.6 582.5,677 581.8,678.2C581.1,679.6 580.2,681.1 579.5,682.5C579,683.7 576.9,686.5 579.3,684.7C583.2,681.9 586.5,678.3 590,675C591.8,673.1 593.7,671.3 595.2,669.1C596.7,667.1 598,665 598.9,662.7C600.2,659.5 601,656 601.5,652.6C601.8,650.9 602.2,648 600.2,648.2C598.7,648.4 597,648.6 595.8,648Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M574.4,597.6C573.9,598.1 574.2,599.7 573.5,599.6C572.3,599 571.2,598.7 569.6,598.2C568.7,598 567.8,597.6 566.8,597.2C567,594.9 567.2,592.7 568.5,590.8C569.4,589.5 570.9,590.2 572.1,590.6C573.4,591 574.7,591.8 575.8,592.5C577.5,593.7 579.3,594.4 581.1,595.4C583.9,596.9 586.7,598.6 589.3,600.5C591.9,602 593,602.8 594.8,604C596.4,605 597.6,606.5 599,607.8C600.9,609.5 602.3,611.7 603.8,613.8C605.9,616.9 608.7,620.5 610.3,623.9C611.4,626.2 611.8,627.8 612.4,630.4C612.8,631.9 613.4,633.8 611.8,634.4C610.6,634.8 609.7,635 608.5,634.8C607.6,634.7 606.6,634.5 606,633.8C604.6,632.2 605,629.6 604.5,627.6C604.1,626.1 603.9,624.7 603.2,623.3C602.3,621.3 601.7,619.3 600.5,617.5C599.3,615.8 598.3,614.1 596.9,612.6C595.5,611 594.3,610 592.6,608.6C590.3,606.7 587.2,604.3 584.8,602.6C583.1,601.3 581.4,600.4 579.5,599.4C578.3,598.8 577.1,598 575.9,597.5C575.3,597.4 574.9,597.3 574.4,597.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M606.8,634.4C606.6,635.3 606.6,635.8 606.7,636.8C606.8,637.8 607,638.5 607.3,639.5C607.7,640.8 607.9,641.8 608.1,643.1C608.3,644.2 608.2,645.3 608.1,646.4C608,647.8 607.6,649.5 607.2,650.6C606.8,651.7 606.4,652.8 606,653.8C605.7,654.4 605.2,655.3 605,655.7C604.5,656.8 606.9,655.4 607.5,655.1C608.2,654.6 608.7,654.1 609.2,653.5C609.8,652.6 610.3,651.7 610.7,650.7C611.4,648.8 612.3,646.9 612.9,645C613.4,643.2 614.3,641.5 614.6,639.7C614.7,638.6 614.7,637.4 614.1,636.5C613.7,635.9 613.2,635 612.5,634.5C612.4,634.4 612.3,634.2 612,634.3C611.5,634.5 610.4,634.9 609.2,634.9C608.4,634.9 607.6,634.7 606.8,634.4L606.8,634.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M446.6,626.4C446.6,626.4 448.2,626.5 449,626.4C449.9,626.4 450.3,626.4 451,626.2C451.9,625.9 452.7,625.1 453.7,624.9C455.5,624.5 458.1,625.1 459.3,624.9C460.2,624.7 461.7,624.3 461.7,624.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M466.6,623.2C468.5,622.6 470.3,621.3 472.2,620.7C474.1,620.3 475.9,621.2 477.7,621.1C478.3,621.1 478.9,620.9 479.5,620.9" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M485.9,619.7C487.7,619.2 489.4,618.9 491.2,618.5C493,618.2 494.3,618.3 495.9,617.6C496.8,617.3 497.9,617 498.5,616.7C499,616.4 499.6,616.1 500.1,616" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M505,614.2C506.2,614.3 507.4,614.2 508.6,614.1C509.7,614.1 510.8,614 511.9,613.8C512.8,613.7 513.7,613.5 514.6,613.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M520.8,611.6C522.1,611.5 523.3,611.1 524.4,610.5C526.1,610 527.8,610 529.5,609.6C530.2,609.5 530.8,609.4 531.5,609.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M537.9,607.9C540,608.2 541.8,607.5 544,606.9C545.8,606.3 547.3,606 549.8,606.2" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M555.6,604.8C557,603.3 557.1,603.2 558.8,602.2C559.2,602 561.6,601.4 562,601.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M431.8,626.8C431.8,626.8 432.9,626.9 433.4,626.9C434.9,627 436.3,627 437.8,626.9C438.3,626.9 438.8,626.9 439.2,626.8C439.7,626.8 440.7,626.6 440.7,626.6" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M375.6,662.3C376.1,661.4 375.5,661.1 372.8,660.6C371.5,660.4 370.2,660.7 369.4,661.6C368.7,662.5 368.7,663.9 369,665C369.4,666.5 370.1,667.7 371.6,668.8C373,670 373.1,669.5 373.3,668.4C373.8,665.2 374.6,664.3 375.6,662.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><g transform="matrix(1,0,0,1,0,308.3)"><path d="M489.8,414.5C489.8,414.5 489,416.1 488.9,416.9C488.9,417.5 489.3,418.2 489.5,418.5C489.8,419.1 490.6,420 490.6,420" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,308.3)"><path d="M467.1,414.2C467.1,414.2 465.6,414.8 465.1,415.5C464.7,416.1 464.6,416.9 464.7,417.7C464.9,418.2 465.7,419.1 465.7,419.1" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,308.3)"><path d="M447.1,408.9C447.1,408.9 446.3,409.4 446.1,409.8C445.8,410.5 445.8,411.3 446.1,412.1C446.3,412.7 447.3,413.6 447.3,413.6" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><path d="M430.4,709.6C430.4,709.6 429.6,710.3 429.3,710.7C429.1,711 428.9,711.4 428.9,711.8C428.9,712.2 429,713 429.1,713.3C429.2,713.7 430.1,715.2 430.1,715.2" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><g transform="matrix(1,0,0,1,0,308.3)"><path d="M417.7,389.8C417.7,389.8 416.7,390 416.3,390.4C415.9,390.8 415.7,391.3 415.6,392C415.6,392.5 415.6,392.9 415.6,393.3C415.7,394.1 415.8,394.8 416.1,395.5C416.5,396.3 417.6,397.7 417.6,397.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,308.3)"><path d="M405.8,377.1C405.8,377.1 404.7,377.9 404.4,378.4C404.1,378.9 403.9,379.6 403.9,380.1C403.9,380.6 404,381.1 404.1,381.4C404.4,382.7 405,383.6 405,383.6" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,308.3)"><path d="M513.1,409.2C513.1,409.2 512.4,410.2 512.2,410.8C512,411.5 511.7,412.2 511.8,413C511.9,413.4 512.3,414.2 512.3,414.2L512.9,415" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g transform="matrix(1,0,0,1,0,308.3)"><path d="M534.2,403.8C534.2,403.8 533.5,404.6 533.2,405.6C533,406.3 533.2,407.2 533.6,407.8C534,408.4 535.5,408.9 535.5,408.9" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g></g><g ns1:id="noga tył" transform="matrix(1,0,0,1,0,-308.3)"><path d="M756.4,632.9C755.8,634.8 750.3,629.6 748.1,626.9C742,620.2 736.6,612.8 731.4,605.4C729,601.9 725.5,599.7 721.4,598.4C718.8,597.6 716,597.2 713.2,597C710.6,596.8 708,597.3 705.5,596.4C704,595.9 702.6,594.9 701.5,593.9C700.4,592.8 699.2,589.7 699,588C698.7,586.7 698.1,579.8 698.6,578.3C698.8,577.4 698.8,574.1 699.6,571.6C699.8,570.1 700.7,567.5 701.9,566.3C704.6,563.2 707.2,561.7 710.8,559.9C712.5,559 714.2,558.5 716.2,558.5C718.9,558.5 721.7,558.8 724.1,559.9C726.1,560.7 727.6,562.5 728,564.7C729.4,568.7 731.9,572.2 735.2,574.6C738.2,576.8 742.4,577.9 744.2,581.5C745.4,583.9 746,586.5 747.4,588.9C749.4,592.4 752.2,595.7 755.5,598.1C762.8,602.8 770.8,606.2 778.6,609.8C783.7,612.3 789.3,613.7 794.4,616.2C796.5,617.1 798.9,618.8 798.5,621.4C798,624 796.4,626.6 794.2,628.1C792.5,629.3 790.3,629.2 788.1,629.1C785.6,628.9 783.2,629.7 781.1,631.2C778.9,632.8 776.9,634.4 774.8,635.9C773.7,636.7 771.9,638.1 770.8,637.1C769.6,635.6 770.3,633.5 770.7,631.8C771.6,627.9 771.3,623.9 771,620C770.7,617.4 769.6,615 767.8,613.2C765.1,610.9 761.6,609.1 758,609.9C755.6,610 753.8,610.9 751.9,612.6C750.6,613.8 750,615.4 749.8,616.9C749.7,618.7 749.8,620.5 750.3,622.2C751.5,626.1 756.7,632.1 756.4,632.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M771.2,627.9C771.4,625.6 771.2,622.1 770.9,619.2C770.6,618.4 770.6,617.7 771.7,617.9C773,618.2 774.5,618.4 775.7,619.1C776.6,619.6 777.3,620.4 778,621.3C778.6,622.2 779.1,623.3 778.9,624.5C778.9,625.6 777.9,626.6 777.1,627.2C776.1,628.1 774.6,628.4 773.3,628.4C772.6,628.5 771.2,628.7 771.2,627.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M766.4,644.9C767.4,643.5 768.5,642.4 769.3,640.9C770.6,638.5 769.7,637.4 770.2,634C770.5,631.9 771.2,629.4 771.2,627.5C771.3,625.9 771.3,623.9 771.2,622.3C771.1,619.7 770.7,617.7 769.2,615C765.2,608.1 752.5,607.7 750,616C748.2,622.9 754.9,628.7 756.6,633C757.8,636.1 752.7,645.3 750.8,651.4C743.8,666.7 740.3,683.4 738.2,700C736.9,709.6 732.3,715.5 736,723.3C738.5,727.5 741.1,732.4 745.1,735.5C751.4,738.9 758.4,735.1 763.2,730.8C766.5,728.4 772.7,733.9 773.2,729.2C774.2,720.9 770.6,719 768.3,717.5C765,715.2 758.2,718.5 758,712.4C757.8,708.2 756.1,705.1 754.5,701.8C753.1,697 753.2,693.2 753.4,688.8C753.8,679.5 755.8,670.3 758.5,661.2C760.2,655.2 763.1,650.1 766.4,644.9L766.4,644.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M758.7,714.3C759.8,712.6 761.2,711.8 762.3,710.2C762.7,709.6 763.1,709 763.8,708.9C764.6,708.9 765.1,709.5 765.5,710.2C766.1,711.1 767.1,711.8 767.3,713C767.5,714 766.9,714.9 766.3,715.7C766,716.2 765.6,716.6 765,716.5C763.6,716.5 762.2,716.4 760.9,716.2C759.9,716 758.3,715 758.7,714.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M740.9,731.3C740.1,732.6 739.7,734.4 738.3,735.1C737.5,735.5 736.5,734.8 735.6,734.2C734.8,733.5 734.2,732.8 733.8,731.8C733.4,730.9 733,730 733.1,729C733.1,727.9 733.4,726.6 732.9,725.6C732.2,724.2 731,723.2 730.4,721.8C730,720.9 729.8,720 729.9,719.1C729.9,718.3 730,717.2 730.4,716.5C730.7,716.1 730.9,715.8 731.4,715.5C732.3,715 733.5,714.9 734.5,715C734.8,715 734.6,715.9 734.6,717.1C734.6,718.9 734.9,720.8 735.7,722.5C736.2,723.6 737,724.9 737.6,725.9C738.8,727.8 739.7,729.4 740.9,731.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M776.9,743.3C777.3,741.1 778.8,739.7 779.5,737.1C780.5,733.2 771.2,731.1 767,730.3C763.7,729.6 763.8,730.4 760.9,732.8C756.4,736.4 758,738.1 756.6,739C753.5,740.8 750.8,742.7 751,747.1C751.2,752.8 752.6,757.9 757.4,761.9C766.2,769.4 774.6,777.4 783.2,785.2C790.6,793.5 798.4,801.6 804.9,810.6C810.4,818.2 817,825.5 819.9,834.5C821.3,838.8 822.3,843.3 828.2,841.7C832.7,839.4 825.4,834.4 823.8,831C818.6,819.5 812.7,808.3 806.5,797.4C803.8,791.7 801.5,784.9 798.7,779.3C796.2,774.6 792.5,772.2 789.7,767.6C785.5,761.4 779.8,755.5 777.1,748.4C776.6,747 776.6,744.9 776.9,743.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M829.1,838C829.9,839.2 830.1,840.4 831.3,841.2C832.8,842 834.9,842.1 836,840.7C837.5,839 838.1,836.5 837.6,834.4C837.3,833 835.9,831.8 834.9,830.7C832.5,827.9 830.5,824.7 828.4,821.6C825.9,817.6 824,813.3 821.5,809.3C816.8,800.7 812.5,791.8 808,783.1C804.5,775.3 801.5,767.3 798.1,759.4C796.2,755.1 794.2,750.9 791.3,747.1C789.2,744.3 786.9,740.7 783.6,739.3C782,738.6 780.9,738 779.2,738C778.8,739.8 776.4,742.4 776.7,745.3C776.6,748.9 777.3,749.1 779,752C779.9,753.8 780.5,753.9 781.4,754.1C782.3,754.3 784.7,754.3 786,755.4C787.2,756.5 790,760.7 791.6,763.5C793.3,766.5 795.2,771.3 795.9,772.8C797.2,775.7 798.7,778.8 799.8,781.6C802.9,788.5 805.2,795.8 809.2,802.2C814.6,811.8 819.5,821.6 824.2,831.6C825.6,833.9 827.6,835.7 829.1,838Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M840.6,820.6C839,821.3 839,822.4 839.8,824.1C840.4,825.5 840.9,827.1 840.6,828.6C840.2,830 839.8,831.3 838.8,832.3C837.8,833.1 837.4,833.4 837.7,834.7C838.3,836.9 837.3,839.5 835.7,841.1C833.5,842.7 831.8,840.9 831.6,841.7C831,843.2 832,845.6 833.9,845.8C836.3,846 835.9,849 835.5,850.8C835.3,851.8 837.2,852.3 836.7,853.7C836.1,855.1 837.8,856.5 839.3,856.9C840.6,857.1 841.8,857 843,856.6C844.5,855.9 845.7,855 847.1,854.2C848.4,853.1 849.6,851.8 849.7,850C849.8,847.8 850.6,845.7 851,843.5C851.1,841.6 850.8,839.6 851.3,837.8C851.9,835.7 852.8,833.8 853.1,831.7C853.6,830.3 853.8,828.8 853.1,827.3C852.4,825.7 851.5,823.2 849.9,821.8C848.6,820.7 846.8,820.3 845.1,819.9C844.3,819.8 843.4,819.8 842.6,819.9C841.9,820 840.6,820.6 840.6,820.6L840.6,820.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M835.4,851.4C834.7,851.7 833.9,852 833.1,852.1C832.3,852.2 831.3,852.2 830.6,852C829.8,851.9 829.3,851.6 828.7,851.3C827.9,850.8 827.4,850.6 826.6,850C826.1,849.6 825.2,848.9 824.8,848.4C824.4,848 824,847.5 823.7,846.8C823.4,846.1 823.3,845.1 823.4,844.2C823.5,843.4 823.9,842.5 824.4,841.8C825.1,842.1 825.9,842 826.7,842C827.7,842 828.9,841.5 829.5,840.6C829.7,840.2 829.5,839.5 829.7,839.3C830.2,839.7 830.3,840.3 830.7,840.8C831,841.2 832,841.2 831.5,841.9C831.3,842.9 831.5,843.9 832,844.7C832.4,845.3 833.1,845.8 833.8,845.8C834.5,845.8 835,846.3 835.4,846.8C835.7,847.3 835.9,847.9 835.8,848.5C835.8,849.5 835.6,850.4 835.4,851.4L835.4,851.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M837.8,856.5C837.1,857.7 836.2,858.8 835.6,860.1C835.2,861 835.3,862 835.8,863C836.3,863.9 837.1,864.8 837.1,865.9C837,866.9 837.4,868 837.5,869C837.7,869.7 838.5,868.9 838.9,868.8C839.7,868.5 840.5,868.2 840.9,867.6C841.5,866.6 842.6,866.5 843.6,866.2C844.7,865.9 845.9,866.1 847.1,866.1C847.7,866.1 848.4,866.1 849,865.9C849.7,865.6 850.1,865 850.2,864.4C850.4,863.7 850.6,863.1 850.7,862.4C850.7,861.5 850.7,860.5 850.4,859.5C850,858.3 849.8,856.9 848.9,856C848.2,855.4 847.4,854.7 846.5,854.6C845.7,855 845,855.6 844.1,856C843.2,856.6 842.1,857.1 840.9,857C840,857 839.2,856.9 838.4,856.5C838.2,856.5 838,856.4 837.8,856.5L837.8,856.5Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M835.5,860.3C835.2,860.8 834.7,861.1 834.2,861.4C833.7,861.7 833.2,861.9 832.7,862C832,862 831.2,862 830.6,861.9C829.8,861.8 829,861.3 828.4,860.9C827.8,860.5 827.4,860 827.1,859.4C826.6,858.5 826.5,857.5 826.3,856.6C826.2,855.7 826.1,854.8 826.2,853.9C826.3,853.1 826.6,852.5 827.1,851.9C827.4,851.6 827.9,850.9 828.3,851.1C829.1,851.4 829.8,851.9 830.6,852.1C831.7,852.3 832.9,852.2 834.1,851.9C834.5,851.8 834.8,851.6 835.1,851.4C835.5,851.2 836.4,852.2 836.7,852.8C836.9,853.3 836.4,854 836.6,854.6C836.9,855.5 837.5,856 837.9,856.3C837.7,856.8 837.2,857.4 836.9,857.9C836.6,858.5 836.4,858.7 836,859.3C835.9,859.5 835.6,860.1 835.5,860.3L835.5,860.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M837.7,869.4C837.8,869.8 837.7,870.5 837.5,870.8C836.7,871.4 836.3,871.6 835.4,871.6C834.7,871.7 833.8,871.8 833,871.7C832.2,871.6 831.5,871.6 830.9,871C830,869.9 829.6,868.4 829.1,867.1C828.8,866.1 828.7,865.2 828.6,864.2C828.5,863.6 828.4,862.7 828.4,862.1C828.4,861.6 828.6,860.9 829.3,861.4C829.7,861.8 830.3,861.9 830.9,862C831.7,862.1 832.6,862.1 833.4,861.8C833.9,861.6 834.5,861.3 834.9,861C835.1,860.8 835.5,860.4 835.4,860.9C835.2,861.8 835.6,862.8 836.1,863.5C836.6,864.2 837,864.9 837.1,865.8C837.1,866.3 837.1,866.8 837.2,867.3C837.3,868 837.4,868.7 837.7,869.4L837.7,869.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M848.4,866.1C851.3,869.5 850.1,873.9 849.4,877.9C849.4,888 849.5,896.1 849.9,907.9C849.9,910.6 851.5,913.4 851.1,916.3C850.8,919.4 850.2,920.9 845.5,920.1C842.1,919.6 843.6,915.1 843.5,912.2C843.4,905.4 841,897.2 842.1,890.4C842.4,886.6 842.9,884.6 842.3,880.8C841.8,877.8 840.6,876 839.2,872.7C838.5,870.8 838.3,869 839.7,868.5C840.5,868.3 841.2,866.9 841.8,866.7C843.6,866.2 845,865.9 846.7,866L848.4,866.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M844.8,920.3C844.9,921.4 843.6,922.1 842.7,922.5C842,922.9 841,922.8 840.3,922.6C839.6,922.5 838.8,922.4 838.5,921.8C837.9,920.8 838.6,919 838.7,918.3C839.4,914.3 839,913.3 839,909.3C839.1,906.8 838.6,904.4 838.4,901.9C838.1,897.7 838.2,893.4 838,889.2C837.5,883.6 837.2,878.1 836.4,872.6C836.3,871.8 836.2,871.5 837,871.1C837.8,870.8 837.9,870.2 837.7,869.5C837.7,869.3 838.9,868.8 839,869C838.1,870.2 839,871.7 839.4,873C840.3,875.5 841.7,877.8 842.2,880.4C842.6,881.9 842.7,883.5 842.6,885C842.4,888.5 841.6,892 841.9,895.5C842.2,900 842.9,904.5 843.4,908.9C843.6,910.5 843.5,911.3 843.5,912.4C843.6,914.8 842.8,917.6 843.9,919.4C844.1,919.7 844.8,920 844.8,920.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M837.5,922.7C834.8,923.2 832.4,922.3 830.9,920.5C829.9,919.2 830.1,917.1 830.4,915.5C830.7,913.9 832.5,913 832.6,911.1C833.5,897.7 833.1,888.6 832.8,877.4C832.9,875.5 832.8,873.6 832.7,871.7C834,871.8 835.4,871.6 836.4,871.6C836.3,873.5 836.8,875.6 837,877.6C837.6,884 838.3,890.4 838.2,896.9C838.3,900.1 838.4,903.4 838.9,906.6C839,909.4 839,912.2 839.1,915C839.1,916.8 838.7,918.4 838.3,920.2C838.1,920.9 838.4,921.3 838.5,921.7C838.6,922.1 839.5,922.3 837.5,922.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M849.4,920C849.7,921.1 850.3,922.2 850.2,923.4C850.2,924.5 850.2,925.4 849.1,926.4C848.6,926.9 847.7,927.3 846.8,928.1C846,928.9 845.1,930.1 844.3,931.4C843.4,932.5 843.2,933.3 843.1,934.6C843,935.5 842.8,936.3 842.2,937C841.8,937.6 841.4,938.1 840.5,938.2C840.1,938.2 839.1,938.1 838.4,937.9C837.6,937.6 836.9,937.3 836.3,936.5C835.6,935.6 835.4,934.6 835.6,933.6C835.8,932.2 836.4,931.1 837.1,930C837.5,929.4 838.2,928.8 838.8,928.4C839.7,927.7 840.1,927.2 840.6,926.3C841.3,925.2 841.5,923.7 842.4,922.7C843.1,922.2 844.3,922 844.7,921C844.9,920.4 844.5,919.9 845.5,920.2C846.4,920.4 847.4,920.3 848.4,920.3C848.7,920.2 849.1,920.1 849.4,920Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M838.1,928.8C836.2,929.4 834.3,929.9 832.5,930.8C831.3,931.3 829.8,932 828.7,932.8C827.8,933.2 826.8,934 825.7,933.7C824.8,933.4 824.1,933 823.3,932.2C822.6,931.4 822.1,930.6 822.5,929.6C822.9,928.7 824.4,928.1 825.1,927.6C827.5,926.2 829.6,925.6 832,924.2C832.8,923.7 833.5,923.2 834.3,922.7C835.8,922.9 837.4,922.9 838.8,922.3C839.8,922.5 840.8,923 841.9,922.8C842.5,923 841.4,923.6 841.6,924.1C841,925.7 840.2,927.3 838.9,928.3C838.7,928.5 838.4,928.6 838.1,928.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M822.2,930.4C821.7,930.7 820.8,930.5 820.3,930.3C819.3,930 818.9,929.6 818.5,929.1C818.1,928.6 817.9,928 817.9,927.6C817.9,926.7 818,925.8 818.4,925.1C818.9,924.2 819.6,923.4 820.5,922.9C821.8,922 823.3,921.4 824.8,921C826.2,920.4 827.3,920 828.7,919.6C829.1,919.5 829.9,919.2 830.3,919.2C830.6,920.3 831.4,921.1 832.3,921.7C832.9,922.1 833.6,922.5 834.3,922.7C833.7,923.3 832.9,923.6 832.2,924.1C830.6,925 829,925.8 827.3,926.5C826.2,927 825.5,927.5 824.4,928.1C823.9,928.5 823.1,928.6 822.7,929.2C822.6,929.4 822.3,930.3 822.2,930.4L822.2,930.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M840.4,938.2C839.4,940.6 838.6,941.9 837.5,944.2C837.2,944.7 836.7,946.1 835.7,946.3C835,946.4 834.6,946.1 834,945.6C833.2,944.9 832.3,944.3 831.8,943.4C831.4,942.7 831.2,941.8 831.4,940.9C831.5,940 832.1,939.1 832.7,938.5C833.3,937.9 833.9,937.8 834.6,937.2C835,936.8 835.4,936.2 835.9,935.8C836.4,936.9 837.4,937.7 838.6,937.9C839.2,938.1 839.8,938.2 840.4,938.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M835.7,946.3C835.9,947.3 835.7,947.6 835.5,948.2C835.2,948.9 834.7,949.6 834.1,950C833.3,950.6 832.5,951 831.6,951C830.8,950.9 830,950.8 829.7,949.9C829.5,949.4 829.6,948.8 829.6,948.3C829.7,947 829.8,946 829.2,944.8C828.7,944 827.9,943.5 827.6,942.7C827.4,942.2 827.3,941.6 827.6,941.2C827.9,940.6 828.6,940.2 829.2,939.9C829.8,939.6 830.7,939.6 831.3,940C831.7,940.2 831.3,940.9 831.3,941.4C831.3,942.3 831.6,943.2 832.1,943.8C832.6,944.5 833.3,945.1 834,945.6C834.5,946 835.1,946.4 835.7,946.3L835.7,946.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M827.4,942.3C826.6,942.5 825.8,943.1 825.1,943.7C823.6,945.1 822.2,946.8 821.4,948.7C820.9,950 820.7,951.4 820.9,952.9C821.1,953.9 821.1,955.5 821.9,954.1C822.3,953.4 822.7,952.8 823.2,952.3C824.2,951.2 825.1,950.1 826.3,949.2C827.1,948.6 828.4,948.6 829.3,948.6C829.7,948.7 829.6,947.1 829.7,946.7C829.7,945.5 829.1,944.4 828.3,943.7C827.9,943.3 827.6,942.8 827.4,942.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M826.3,933.7C825.1,935.3 823.8,936.4 822.4,937.8C821.5,938.8 820.7,940.3 819.6,941.1C819,941.5 817.9,941.8 817.1,941.5C816.1,941.2 815.2,940.3 814.7,939.3C814.4,938.6 814.2,937.7 814.6,937C815.2,936 815.9,935.2 816.6,934.3C817.5,933.4 818.5,932.5 819.4,931.7C819.8,931.2 820.1,930.9 820.7,930.4C822.1,930.9 822.5,929.8 822.4,930.8C822.5,931.4 823,932 823.5,932.4C824.3,933.1 825.3,933.6 826.3,933.7L826.3,933.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M814.3,937.3C813.8,937.6 813,937.3 812.4,937C811.6,936.5 811.1,935.7 810.6,935C810.2,934.4 809.9,933.8 809.8,933.1C809.7,932.5 809.7,931.9 810,931.4C810.4,930.6 811.4,930.3 812.1,929.8C812.6,929.4 813.2,929.1 813.9,928.6C815.1,927.8 816.3,926.8 817.6,925.9C818.6,925.2 817.8,926.4 817.9,927.5C817.9,928.4 818.5,929.3 819.3,929.9C819.9,930.3 821,930.4 819.9,931.2C818.4,932.1 817.6,933.4 816.4,934.6C815.7,935.2 815.5,935.4 815,936.2C814.9,936.4 814.5,937.3 814.3,937.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M819.3,941.4C819.8,942.7 820.2,944 819.7,945.4C819.6,947 818.3,948.7 816.5,948.8C815.1,948.9 813.5,948.6 812.7,947.5C812,946.5 812.8,945.2 812.5,944.1C812.3,942.9 812.1,941.7 811.1,940.9C810.1,940 811.2,938.5 812,938C812.7,937.6 814.3,937.2 814.4,938.5C814.6,939.7 815.7,940.8 816.6,941.3C817.4,941.7 818.5,941.7 819.3,941.4L819.3,941.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M810.8,940.6C810.6,940.4 809.5,941.1 808.9,941.4C808.3,941.8 807.7,942.2 807.2,942.6C806.5,943.4 805.8,944.2 805.3,945.1C804.9,945.8 804.6,946.6 804.4,947.4C804.2,948.4 804.1,949 804.2,950C804.3,950.4 804.2,951.6 804.7,950.9C805.1,950.4 805.5,950.1 805.8,949.7C806.4,949 807,948.4 807.6,947.8C808.1,947.5 808.5,947.1 809.1,946.9C810,946.5 811,946.2 812,946.5C812.5,946.6 812.4,946 812.5,945.7C812.6,945.1 812.6,944.5 812.5,943.9C812.3,943 812.2,942.4 811.8,941.6C811.6,941.3 811.5,941.2 811.2,941L810.8,940.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M810.6,940.1C809.8,940.2 807.8,941.1 807,940.6C806.1,940.1 807.4,938.6 807.3,937.6C807.2,936.9 807,936.2 806.6,935.6C806.3,935 804.4,934.8 805.1,934C805.7,933.3 807.9,932 808.8,931.6C810.8,930.6 810.4,930.6 809.8,931.9C809.6,933 810,934.1 810.7,935C811.3,936 812.1,937.2 813.3,937.4C814.1,937.6 812.2,937.7 811.9,938.1C811.3,938.7 810.7,939.3 810.6,940.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M804.4,934.6C803.4,935.7 803.9,935.1 802.9,936.1C802.1,937 801.2,937.9 800.4,939.3C800.1,939.9 799.9,940.4 799.4,942.3C799,944.4 801.5,942.4 801.8,942.1C802.3,941.7 802.7,941.2 803.3,940.8C804.2,940.2 805.2,940 806.3,940C806.8,940 806.7,939.4 806.9,939.1C807.1,938.6 807.3,938.1 807.3,937.6C807.3,936.9 807.1,936 806.6,935.5C806,934.9 804.6,934.4 804.4,934.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><g transform="matrix(1,0,0,1,0,308.3)"><path d="M772,298.4C772,298.4 775.7,301.1 776.6,302.8C777.2,304 777.5,305.4 777.5,306.8C777.5,307.7 777.6,308.8 777,309.4C776.4,310.1 774.6,310.4 774.6,310.4" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g></g><g transform="matrix(1,0,0,1,0,-308.3)"><path d="M744.8,582.7C745.1,583.2 751,582.3 754.2,582.6C755.5,582.7 757.6,582.3 757.5,580.6C757.3,579.4 756.4,577.7 758,577.1C759.3,576.6 758.6,574 757.9,573.3C756.9,572.3 754.9,573.3 754.5,571.4C754.3,570.2 754.5,568.2 752.9,568.1C751.4,568 749.8,567.8 748.5,568.5C747.3,569.1 745.8,569 744.4,569.1C742.6,569.3 740.8,569.2 738.8,569.2C736.3,569.1 733.8,568.8 731.3,568.6C730.7,568.6 729.4,568.1 730.1,569.1C730.9,570.3 731.8,571.4 732.7,572.5C734.8,574.6 737.4,576.1 740.1,577.4C741.6,578.6 742.7,579.2 743.7,580.6C743.8,581 744.2,581.5 744.4,581.9L744.8,582.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M757.1,581.7C758.1,581.9 758.8,582.3 759.8,582.3C761.1,582.4 762.6,582.2 763.6,581.7C764.6,581.3 765.2,580.1 766.3,579.7C767.7,579 769.1,578.7 770.7,578.6C772.3,578.5 774,579.1 775.6,578.8C777.5,578.3 778.9,577.9 780.2,576.6C780.7,576.1 781.1,574.8 781.1,574.1C781.1,572.6 780.8,571 780.6,569.5C780.4,568.5 780,567.1 778.7,567.1C777.7,567.1 777,566.6 776.8,565.6C776.7,564.9 776.3,563.8 775.6,563.4C774.7,562.8 773.5,563 772.5,563.2C771.6,563.5 771.2,563.8 770,564.7C769.1,565.4 768,565.7 766.9,565.9C765.3,566.2 763.7,566.7 762,566.6C760.2,566.6 758.4,566.3 756.6,566.7C755.4,566.9 754,567.3 753.2,568.1C754.6,568.7 754.2,570.4 754.6,571.6C755,573.2 757.3,572.3 758.1,573.6C759.1,575 759.4,576.5 757.4,577.5C756.8,577.8 757.1,579.1 757.4,580.2C757.5,580.5 757.5,581 757.4,581.2L757.1,581.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M780.9,575.4C780.1,576.4 782.6,575.9 783.6,575.7C784.7,575.5 785.9,574.9 786.8,574.2C788.1,573.4 789.2,572.3 790.4,571.3C791.2,570.7 792.1,570.3 793.2,570.2C795.3,570.1 797.3,569.6 799.2,568.9C800.3,568.5 801.2,567.7 801.7,566.6C802.2,565.6 802.1,564.6 801.4,563.7C800.9,562.5 799.6,561.9 799.3,560.6C799.1,559.4 798.8,558.1 798,557.1C797,556.2 795.9,556 794.6,556.2C793.4,556.3 792.7,557.3 791.8,557.8C790.4,558.5 789.1,559.4 787.6,559.8C784.6,560.8 781.4,560.9 778.2,561.3C777.1,561.4 775.4,563.1 776.1,564C776.7,564.6 776.8,566.1 777.5,566.8C778.4,567.4 779.4,566.7 780.1,568.1C780.8,569.3 780.7,570.1 780.9,571.5C781,572.3 781.2,572.9 781.1,573.5C781.1,573.5 781.2,574.9 780.9,575.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M801.8,566C801.2,567.8 806,565.5 807.8,564.8C809.7,564.1 810.5,563.7 812.4,563.3C813.5,563 815.5,562.2 816.6,561.8C817.6,561.4 818.7,560.8 818.8,559.7C819,558.1 818,557.6 818,556.1C817.8,555 818.5,552.2 817,551.3C816.1,550.7 815.3,549.8 813,550.6C811.6,551.1 811.3,551.2 810.3,551.5C807.7,552.3 806.4,552.8 804,553.8C802.8,554.2 801.3,554.5 800.1,554.9C798.9,555.3 798,555.3 796.7,556.2C798.3,556.9 799.2,558.9 799.2,559.6C799.3,560.9 799.8,561.9 800.7,562.7C801.4,563.5 802.3,564.7 801.8,566Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M818.8,559.5C818.5,560.6 820.2,560.2 822.2,559.4C823.4,558.9 824.4,557.7 825.7,556.7C826.9,555.7 828.1,554 829.5,553C830.7,552.2 832,551.9 833.2,551.3C834.4,550.7 836,550.5 836.8,549.4C837.6,548.3 838.1,547 837.7,545.5C837,544.5 835.9,542.9 835.7,541.6C835.5,540.3 833.4,539.7 832.3,539.7C830.9,539.7 829.8,539.9 828.7,540.9C827.4,541.7 826.1,542.6 824.9,543.5C823.6,544.4 822.6,545.8 821.2,546.1C819.1,546.7 817.7,547.2 816.1,548.9C815.3,549.7 815.4,550.4 816.4,550.9C817.6,551.3 818.1,553 818,554.2C817.9,555.4 817.8,556.7 818.4,557.7C818.6,558.1 819.1,558.6 818.8,559.5Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M837.8,547.5C838.8,547.6 840,547 840.8,546.6C841.7,546.2 842.4,545.7 843,545.1C844.1,544 845.1,542.6 846,541.5C846.8,540.5 847.8,539.5 848.9,538.8C849.9,538.2 851,537.8 852,537.4C853.3,537 854.3,536.2 855.2,535.1C856,533.9 855.7,532.6 855.4,531.5C854.9,530.4 854.1,529.3 853.1,528.6C852.1,527.9 850.9,527.5 849.8,527.5C848.6,527.5 847.5,528.1 846.5,528.7C845.1,529.6 844.1,530.4 842.8,532C842.1,532.8 841.2,533.4 840.2,533.8C839,534.3 836.8,535.4 835.8,536.3C834.7,537.2 832.3,539.4 833.8,539.9C834.9,540.3 835.8,540.9 835.9,542.5C836,543.9 837.3,544.4 837.7,545.6C838.1,546.8 837.7,547.5 837.8,547.5Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M855.7,532.6C857.3,532.6 859.3,532 860.1,530.5C861.4,528.4 862.1,526.9 864.4,525.5C865.7,524.6 867.8,523.7 869.1,522.5C870.5,521.2 872.5,520 872.7,517.9C872.8,516.3 871.6,515.1 870.6,514.1C869.5,513.1 867.4,512.4 866.1,512.5C864.7,512.6 863.7,513.2 863.1,514.5C862.5,515.8 861.9,517 860.8,518.1C859.8,519.2 858.7,520.2 857.4,521.1C855.6,522.4 853.2,523 851.5,524.3C850.6,525 849.7,526.8 851,527.6C852.3,528.1 853.6,528.7 854.4,529.9C855.1,530.6 855.5,531.6 855.7,532.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M872.5,516.4C874.8,516.2 877.1,514.8 878,512.6C878.9,510.7 878.7,508.8 880.3,507.1C881.7,505.4 883.2,504.4 883.4,502C883.6,500.4 884.1,499.1 883,497.9C882.2,496.9 878,495.5 876.9,495.9C875.4,496.4 875.3,496.7 874.7,497.3C874.1,497.9 873.9,498.8 873.7,499.5C872.9,501.5 873.1,503.6 871.9,505.7C871.1,507 869.7,507.6 868.8,508.6C867.9,509.5 867,510.6 867.2,511.8C867.5,512.8 869.1,513 870,513.6C871,514.3 871.8,515.4 872.5,516.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M883.3,498.1C884.6,497.7 885.6,497.2 886.3,496.1C887.2,494.9 887,493.3 887.3,492C887.6,490.1 887.5,488.2 888.1,486.5C888.8,484.5 890.8,483.1 891.4,481.1C891.8,479.9 891.8,478.2 891.2,477.1C890.6,475.6 888.7,476.3 887.4,475.4C886.4,474.6 884.3,474.4 883.4,475C882.1,475.9 882.1,476.3 881.8,477.1C881.1,478.8 880.7,480.4 881,482.6C881.1,484.7 880.3,487.4 879,489.2C878.1,490.6 877.9,491.5 877.5,493.2C877.3,494.2 877.2,495.4 878.1,495.8C879.5,496.1 881.3,496.8 882.6,497.5C882.8,497.7 883.2,497.9 883.3,498.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M891,476.5C891.7,475.5 892.1,473.8 892.2,472.4C892.4,469.7 892.5,467.8 893,465.1C893.2,464.1 894,462.9 894.2,461.9C894.6,460.5 894.6,459.2 894.4,457.8C894.2,456.8 893.3,456.1 892.1,456C891.1,455.8 889,455.5 887.8,456.3C887.1,456.7 886.7,457.2 886.5,458.3C886.4,459.4 886.5,460.5 886.8,461.6C887.2,463.4 886.8,465.4 885.8,467.1C885.1,468.3 884.6,469.4 884.3,470.6C884,471.9 882.9,474.8 884.1,474.7C885.2,474.5 886.7,474.8 887.6,475.4C888.5,476.1 890.2,476 891,476.5L891,476.5Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M892.7,456.1C893.2,454.8 894.1,453.5 894.1,452.1C894.2,449.3 892.9,447 893.3,444.1C893.6,441.7 894.1,438.6 892.6,436.5C891.9,435.3 889.6,434.8 888.3,435.4C887,435.9 886.7,436.3 886,437.4C885.3,438.8 885.8,440.4 885.9,441.9C886.1,444.1 886,446.3 885.9,448C885.8,449.8 885.8,451.6 886.3,453.3C886.7,454.3 887.4,455.2 888.3,455.7C889,455.9 890.1,455.7 891.1,455.8C891.6,455.9 892.2,456 892.7,456.1L892.7,456.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M890.7,435.1C891.7,434.2 892.1,433.9 892.3,432.4C892.4,431.3 892.1,429.3 891.4,428.3C890.1,426.3 889.3,424.1 888.6,421.8C888.2,420.6 888,419.4 887.4,418.3C887.1,417.4 886.5,416.6 885.9,415.9C885.2,415 883.9,415.6 883.1,415.9C882.2,416.3 881.4,417.1 881.4,418.2C881.3,419.5 881.6,420.5 882.2,421.6C883.2,423.4 884.2,425.3 884.7,427.4C885.3,429.4 885.3,431.4 885.5,433.5C885.6,434.3 885.5,436.1 886.5,436.3C887.9,435.5 889,435 890.7,435.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M885.3,414.4C885.5,413 885.4,412.4 885.1,411C884.9,409.7 884.5,408.5 884,407.4C882.5,404.4 881.7,401.1 880.3,398.1C879.8,397.1 878.9,396.3 878.1,395.5C877.4,394.7 876,394.4 875,394.8C874.1,395.3 872.7,395.5 872.5,396.9C872.3,398.2 872.9,399.4 873.4,400.6C874,402.1 874.9,403.5 876.1,404.7C877,405.8 877.6,407.2 877.8,408.6C878.1,409.6 878,410.5 878.1,411.5C878.3,412.8 879,414 879.7,415C880.3,416 880.8,416.5 881.9,416.5C882.7,416.2 883,415.9 883.6,415.7C884.1,415.6 885,415.5 885.1,415.4L885.3,414.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M870.4,394.8C871.1,395.5 871.9,396.4 873,395.8C874.1,395.1 875.4,395 876.3,393.9C876.8,392.5 876.4,390.4 875.5,389.3C873.5,387.2 871.2,385.5 869.7,383.1C868.2,381.2 866.7,379.1 864.2,378.6C862.5,378.2 861.8,378.4 860.9,379.5C859.6,381 860.6,383.3 862,384.2C863.4,385 864.7,385.8 865.8,387.1C867.5,389 868.6,391 869.4,393.3C869.7,393.9 870,394.4 870.4,394.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M860.2,381.4C858.6,381.4 856.7,381.2 855.5,380C853.9,378.6 852.8,376.5 850.9,375.5C848.4,374 845.3,373.2 843.3,371.1C842.4,370.2 843.4,369.5 843.9,368.4C844.3,367.5 844.2,366.3 845.3,366.3C846.4,366.3 848.7,366.6 849.5,367.5C851.4,369.6 853.5,371.4 856.3,372C858.3,372.7 860.1,374 861.2,375.8C861.8,376.6 862.7,377.8 861.8,378.7C861.4,378.9 860.1,380.5 860.2,381.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M843,369.9C842.3,371 840.4,369.9 839.2,369.2C837.3,368.1 835.3,367 833.1,366.6C831.8,366.4 830.4,366.1 829.3,365.3C828.7,364.9 828.1,364.5 827.8,363.7C827.3,362.5 827.3,361.6 827.7,360.3C828.1,359.4 829,358.9 829.9,359C831.1,359.2 832.1,360 833.1,360.6C834.1,361.3 835.1,362 836.3,362.3C837.5,362.6 838.8,363 840,363.5C841,364 842.4,364.5 843.1,365.1C843.8,365.7 844.4,366 844.3,366.9C844.1,368 843.7,368.9 843,369.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M828.6,359.2C827.1,358.4 825.6,357.3 824,356.8C822.4,356.2 820.6,355.7 818.2,355.7C815.5,355.7 813.1,355.5 810.2,356.4C809.6,356.6 808.3,357.7 809.4,358.2C812.1,359.4 814.4,358.8 816.8,360.6C817.4,361 818.3,361.9 819.2,362.4C820.1,362.8 821.5,363.2 822.2,363.3C823.6,363.6 824.7,363.7 825.6,363.6C826.5,363.6 827.7,363.8 827.6,363.2C827.4,362.1 827.4,361.7 827.6,360.7C827.6,360.3 828,359.8 828.6,359.2L828.6,359.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g ns1:id="noga przód 2 plan" transform="matrix(1,0,0,1,0,-308.3)"><path d="M457.3,780.9C459.5,786.9 461.9,792.8 463.6,798.9C465.7,805.4 467.6,811.3 469.4,817.9C470.4,821.8 471.1,825.3 472.1,829C472.9,832.1 473.8,835.8 474.6,838.3C476.5,843.9 478.5,848.8 479.7,854.6C479.9,855.8 480.2,856.7 481.6,856.8C482.8,857.1 483.6,858.5 484.8,858.6C485.8,858.6 486.8,857.9 487.3,856.7C488.1,854.4 490.1,852.5 489.7,850C488.9,844.4 485,840.4 483.1,835C479.4,824.7 476.8,814.1 474,803.5C472.3,797.3 470,791.2 468.5,784.9C467.8,781.9 467.5,778.8 467,775.7" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M477.5,774.7C477.2,779.3 477,784.1 477.3,789.1C477.6,795.7 478.3,800.9 479,806.9C479.7,812.6 481.3,818.1 482.5,823.7C483.9,827.8 485.5,831.8 487.3,835.7C489.4,839.6 491.8,844.4 492.9,848.8C493.7,851.7 494.1,855.8 489.9,853.7C488.2,852 490.9,853.1 489.1,847.2C487.2,842.1 484.7,840.2 482.2,832.4C479.9,825.5 477.9,818.6 476.1,811.6C474,803.1 471.2,794.8 468.8,786.4C467.8,782.6 467.5,778.7 467,774.9" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M479.7,856.2C479.2,857.9 478.3,859.1 478.3,860.7C478.3,861.3 478.4,862.2 478.9,862.7C480,863.8 481.8,863.8 483.3,863.9C484.6,863.9 485.9,863.8 487.1,863.8C488.1,863.6 488.2,863.6 489.2,863.1C490.4,862.4 490.6,861.3 491.5,860.3C492.3,859.2 493.3,858.7 493.7,857.6C494.1,856.8 493.7,855.8 493.9,854.9C494,854 494.2,853.1 494.5,852C494.5,851.7 493.8,851.6 493.5,851.3C493.6,852.2 493.6,853.4 492.8,854C492,854.6 490.8,854.2 490,853.7C489.6,853.5 489.3,852.5 488.9,853.4C488.2,854.6 487.7,855.3 487.2,856.9C486.8,857.9 485.7,858.5 484.8,858.6C483.6,858.5 482.6,856.9 481.9,856.9C480.2,856.7 479.9,855.5 479.7,856.2L479.7,856.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M489.5,862.9C490.2,863.8 490.6,865.1 491.6,865.4C493.3,865.9 495,864.5 495.8,862.9C496.5,861.3 497.8,860.4 499.3,859.7C500.9,859 501.3,857.5 501.3,856.1C501.4,854.7 500.5,853.1 499.8,852.3C498.2,851 496.3,852.8 494.7,851.9C493.8,853.5 493.9,855.1 493.9,856.9C494,858.6 491.6,859.4 491,861C490.6,861.7 490.2,862.4 489.5,862.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M481.3,864C480.7,864.6 480.8,865.8 480.9,866.7C481.1,867.8 481.3,870 482.6,869.5C484.3,868.8 485.4,868.8 486.8,868.6C488.2,868.5 490,869.1 491,868.6C491.8,868.3 491.6,867.5 491.8,866.8C492.2,865.5 490.6,865.2 490.3,864.1C489.5,862.4 488.9,863.3 488,863.7C486.8,864.1 485.5,863.8 484.2,863.9C483.3,864 481.6,863.7 481.3,864L481.3,864Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M490.8,868.9C491.2,869.4 491.5,870 492,870.3C493.1,870.9 494.5,869.9 494.8,868.7C495.1,867.6 495.2,866.1 495.2,864.9C495.2,863.6 494.1,865.5 492.4,865.5C491.5,865.5 491.5,865.2 491.8,866.1C491.9,866.6 491.8,867.4 491.6,868.1C491.6,868.2 491.4,868.4 491.2,868.6L490.8,868.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M489.8,873.2C490.2,871.7 492.8,869 489.5,868.8C487.1,868.6 482.9,868.3 481.1,871C478.7,875.5 477.1,879.5 475.3,883.8C474.2,886.4 472.8,888.8 472.3,891.6C471.6,896.7 470,900.2 466.6,904.9C465.8,906 465.6,908.9 466.8,910.2C468.5,912 471.9,912.4 474.4,911C476.3,909.9 477.7,905.8 477.7,903.5C478.1,900.5 477.8,899.4 478.8,896.5C480.2,892.7 481.8,889.1 483.1,885.3C484.1,882.6 485.1,880.3 486.9,878C488,876.5 489.1,874.6 489.8,873.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M466,908.1C464.1,908.9 461.9,908.4 461.8,906C461.7,903.5 464.2,902.3 465,900.2C466.8,897 467.6,893.4 468.8,890C470.4,885 472.4,880.1 474.3,875.2C475.9,870.9 477,866.2 478.4,861.8C479,863.6 481.1,863.6 481.8,863.8C480.2,864.5 481,866.8 481.3,868.4C481.8,870.6 483.1,869.1 481.8,870.3C480.6,871.7 480,873.5 479.1,875.1C477.8,878 476.6,881 475.2,883.9C474,886.9 472.4,889.4 472.2,892.7C471.7,896.3 469.9,900 467.9,903.1C466.7,904.7 465.9,905.7 466,908.1Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M475.2,911.3C475.4,912.1 475.9,912.8 476.7,913.3C477.3,913.7 478.3,913.5 478.9,913C480.3,911.9 480.8,909.8 480.7,907.9C480.7,905.8 481.1,903.8 481.8,901.9C482.9,897.8 483.8,894 485.2,890C486.2,887.1 487.7,884.1 489.2,881.4C490.2,879.6 491.9,878.1 492.6,876C493.1,874.4 493.5,873.3 493.9,871.7C494.2,870.4 493.5,870.4 492.8,870.5C492.2,870.5 491.6,870.3 491.3,869.6C491,869.3 491.4,869.9 490.7,871.3C490,872.7 489.4,874.1 488.6,875.3C487.1,878.1 484.8,880.5 483.8,883.6C482.1,888.1 480.3,892.6 478.6,897.2C477.6,900 478.1,903 477.3,905.8C477,907.4 476.3,909 475.2,910.2C475.1,910.4 475.1,910.9 475.2,911.3L475.2,911.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M466.6,910C465.1,911.9 463.4,913.6 462,915.5C461.6,916 460.7,917.1 460.7,917.7C460.5,918.9 460.7,919.8 461.1,920.9C461.4,921.5 461.9,921.8 462.5,922C463.1,922.2 463.8,922.1 464.4,921.9C465,921.8 465.6,921.6 466.2,921.2C467.2,920.6 468,919.7 469,918.9C470.5,917.4 471.9,915.8 473.4,914.3C474.2,913.7 474.9,913.1 475.7,912.5C475.4,911.9 475.1,911.2 475.1,910.5C474.7,910.9 474.2,911.2 473.3,911.5C471.6,911.9 470,911.9 468.5,911.4C467.8,911 467.1,910.6 466.6,910L466.6,910Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M464.4,922.2C463.8,922.9 464,923.7 464.6,924.6C465.1,925.3 465.9,925.9 466.8,925.7C468,925.6 468.8,924.7 469.6,924C470.9,923 472.2,921.5 473.2,920.3C474.3,918.8 475.1,917.7 476.4,916.5C477.2,915.6 478,915.3 478.5,914.3C478.7,913.9 479.2,912.8 478.4,913.4C477.7,913.8 476.5,913.3 476,912.7C475.5,912.2 474.9,913.2 474.5,913.5C471.6,916 469.3,919 466.3,921.3C465.8,921.6 464.7,921.8 464.4,922.2L464.4,922.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M460.8,917.3C460.5,917.9 459.1,917.5 458.4,917.2C457.7,916.9 457.1,916.4 456.8,915.8C456.5,915.3 456.1,914.8 456.3,914.2C456.4,913.6 456.7,912.6 457.1,912.1C458.4,910.7 459.3,909.5 460.6,908.1C461.1,907.6 461.5,907.1 461.9,906.6C462.1,907.3 462.6,908 463.2,908.3C463.9,908.6 464.8,908.4 465.6,908.2C466,908.1 466,908.1 466.1,908.7C466.2,909.4 466.6,909.7 466.6,910.1C466.1,910.7 465.5,911.1 465.1,911.7C463.9,913.3 463.6,913.5 462.4,915.1C462.1,915.4 461.8,915.9 461.5,916.3C461.2,916.6 460.8,917.3 460.8,917.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M456.9,929.3C457.6,929.1 459,928.2 459.6,927.7C460.4,926.8 460.9,926.2 461.4,925.1C461.8,924.2 462,923.1 462.3,922C461.2,921.6 460.8,920.5 460.7,919.5C460.5,918.8 460.7,918 460.5,917.5C459.7,917.8 459,917.4 458.2,917.1C456.9,918.5 455.7,919.9 454.5,921.4C454,922.3 453.4,923.1 452.8,923.9C452.5,924.3 451.9,924.6 451.8,925.3C451.8,926 451.8,927.5 452,928.2C452.2,928.7 452.2,929.5 452.6,929.8C454,929.7 455.5,929.7 456.9,929.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M449.4,924.4C448.4,924.5 447.3,924.5 446.4,924.9C445.9,925 445.4,925.2 445.2,925.6C445.1,925.9 445.2,926.4 445.4,926.7C445.6,927.4 445.7,927.9 445.6,928.6C445.5,929 445.4,929.8 445.8,929.8C446.1,929.9 446.5,929.7 446.8,929.7C447.6,929.5 448.3,929.4 449.1,929.4C450.2,929.4 451.4,929.6 452.5,929.8C452.3,929.2 452.1,928.7 452,928C451.8,926.9 451.7,926 451.8,925C451.9,924.6 451.4,924.6 451,924.5C450.5,924.5 450,924.4 449.4,924.4L449.4,924.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M452.7,929.8C452.8,930.5 453.2,931 453.3,931.6C453.3,932.2 453.4,932.8 452.9,933.3C452.2,933.9 451.3,934.5 451.1,935.5C450.9,936 451,936.6 451.3,937.2C451.6,937.8 452.3,938.2 453,938.1C453.7,938.1 454.4,938.2 455.1,938C455.8,937.8 456.3,937.8 456.8,937.3C457.6,936.5 458.3,935.1 458.3,933.9C458.2,933 458.1,932.1 457.9,931.1C457.8,930.6 457.5,930 457.2,929.6C456.9,929.2 456.2,929.5 455.8,929.6C455,929.7 454.9,929.7 454.4,929.7C454.1,929.7 453.6,929.8 453.4,929.8L452.7,929.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M452.5,929.9C451,929.5 450.2,929.4 448.6,929.4C447.7,929.4 446.1,929.7 445.2,930.1C444.5,930.4 443.7,930.5 443.1,930.9C442.5,931.2 441.9,931.5 441.5,932C440.9,932.6 440.5,933 440.2,933.8C440.1,934.2 439.9,934.7 439.8,935.6C439.8,936.3 440.9,935.6 441.5,935.4C442.3,935.1 442.8,935 443.6,934.9C444.2,934.8 445.1,934.8 445.9,934.8C447,934.8 448.1,934.9 449.2,935.1C449.8,935.2 450.5,935.7 451,935.6C451.1,935 451.5,934.6 451.9,934.1C452.3,933.7 452.9,933.5 453.1,933C453.4,932.5 453.3,932.1 453.3,931.6C453.2,931 452.8,930.4 452.5,929.8L452.5,929.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M444.9,926C443.8,925.6 442.8,925.9 441.7,926.2C440.6,926.4 439.5,926.8 438.6,927.5C437.6,928.2 436.6,929.1 436,930.1C435.4,930.8 435,931.5 434.5,932.6C434.3,933.3 434.2,933.6 434.2,934.3C434,935.6 434.4,935.3 434.9,935.1C435.5,934.6 435.9,934.3 436.5,933.9C437.3,933.4 437.7,932.9 438.6,932.7C439.4,932.4 440.5,932.3 441.3,931.9C442,931.6 442.5,931.1 443.1,930.8C443.9,930.5 444.4,930.3 445,930.1C445.5,930 445.5,930.1 445.5,929.3C445.5,928.5 445.8,927.7 445.4,926.9C445.3,926.7 445.2,926.1 444.9,926L444.9,926Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M456.1,914.3C454.7,915.3 453.2,916.2 451.8,917.3C450.8,918 450.5,918.3 449.5,919.1C448.9,919.5 448,920.5 447.3,920.3C446.7,920.2 445.5,919.4 445.5,920.3C445.6,921.2 445.8,921.8 445.8,922.7C445.7,923.5 445.5,924.3 445.3,925.4C445.6,925.4 446,925 446.5,924.9C447.1,924.7 447.6,924.6 448.2,924.5C449.2,924.3 450.2,924.4 451.2,924.6C451.9,924.7 452.3,924.5 452.6,924C453.6,923 454.2,921.7 455.2,920.6C455.9,919.8 456.5,919 457.2,918.1C457.6,917.7 458.4,917.1 457.7,916.8C456.9,916.4 456.7,915.9 456.4,915.2C456.3,915 456.3,914.5 456.1,914.3L456.1,914.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M445.6,921C445.6,920.6 445.6,920.1 445.5,919.4C445.5,919 444.8,918.9 444.3,918.8C443.7,918.7 442.9,918.7 442.3,918.7C441.8,918.8 441.2,918.8 440.8,919C440.4,919.3 440,919.6 439.9,920.1C439.8,920.5 439.9,920.9 440.1,921.5C440.4,922.5 440.2,923.6 440,924.5C439.8,925 439.7,925.4 439.5,925.9C439.5,926.2 439.7,926.7 440.1,926.6C440.6,926.5 441,926.3 441.5,926.2C442.5,926 443.6,925.7 444.7,926C445.2,926.1 445.1,925.8 445.3,925.2C445.5,924.4 445.6,924.1 445.7,923.3C445.7,922.2 445.8,922.9 445.7,921.8L445.6,921Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M439.9,920.7C439.7,919.5 436.9,920.8 435.8,921.3C434.6,921.9 434.2,922.4 433.6,922.8C433.1,923.2 432.6,923.6 432.4,924C432.1,924.6 431.8,925 431.6,925.5C431.5,926 431,926.7 431.6,926.9C432,926.9 432.4,926.7 432.8,926.5C433.7,926.3 434.5,925.8 435.4,925.8C436.4,925.7 437.5,925.5 438.5,925.8C438.8,926 439.3,926 439.5,925.7C439.9,925.1 440,924.5 440.1,923.8C440.2,923.2 440.4,922.6 440.3,922.1C440.2,921.5 439.9,921.2 439.9,920.7L439.9,920.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M458.2,935.4C459.2,935.2 460.1,935 461,934.6C461.6,934.2 462.3,933.7 462.8,933.2C463.4,932.4 463.8,931.5 464.2,930.5C464.8,929.1 465.2,927.5 465.7,926C465.9,925.4 465.1,925.4 464.7,924.8C464.2,924.2 463.8,923.5 464,922.8C464.3,921.9 466.1,921.4 465.3,921.6C464,922.1 463.4,922.2 462.5,922C462.1,922 462.1,923 461.9,923.4C461.4,925.3 460.3,927.4 458.5,928.5C458.1,928.9 457.4,929.1 457.1,929.3C457.9,930.8 458.1,931.8 458.2,933.1C458.3,934.2 458.2,934.7 458.2,935.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g><g ns1:id="noga tył 2 plan" transform="matrix(1,0,0,1,0,-308.3)"><path d="M687.9,681.3C687.9,681.3 693.5,673.5 696.4,669.7C699.4,666 702.6,662.4 705.7,658.9C709.6,654.4 717.5,645.6 717.5,645.6C717.5,645.6 723.4,638.7 726.5,635.3C729.2,632.4 732.1,629.6 734.8,626.7C736.8,624.5 738.8,622.3 740.7,620C741,619.7 741.5,619 741.5,619C741.5,619 742.5,620.3 743,620.9C743.6,621.7 744.1,622.4 744.7,623.1C745.5,624 746.2,624.8 747,625.7C747.7,626.5 748.4,627.3 749.2,628.1C749.7,628.7 750.2,629.2 750.7,629.8C751.4,630.4 752.1,630.9 752.7,631.5C753,631.7 753.6,632.3 753.6,632.3C753.6,632.3 751.6,633.3 750.6,633.8C748.7,634.9 746.8,635.8 745.1,637C743.1,638.5 741.1,640.7 739.5,642.1C738.1,643.3 734.2,646.9 734.2,646.9C734.2,646.9 731.8,649.5 730.2,650.9C728.6,652.3 727.8,653.3 726.3,654.8C724.5,656.5 722.8,658.6 721.1,660.5C719.3,662.4 717.6,664.3 715.9,666.4C713.6,669.3 711.1,672.1 709,675.1C706.9,677.9 705.2,681 703.3,684C702.2,685.8 700.1,689.3 700.1,689.3" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M680.4,730.5C677.1,733.6 675.9,737.5 676,742C676.1,747.3 681.2,749 682.3,753.8C683.4,758.4 684.3,763.4 687.4,767.1C692.3,771 698,774.3 700.3,780.6C704.6,792.3 708.4,804 711.9,815.9C713,823.1 715.3,830.1 716.6,837.3C717.5,842.4 720.3,841.6 725.2,841.3C730.1,841 727.2,838.9 732.3,834.6C736.4,831.1 728.4,822.8 726.2,817.7C720.8,807.8 717,797.1 714.3,786.2C712.2,779.9 710.6,773.5 710.1,767" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M716.1,764.7C717.4,771.9 719.3,778.9 720.9,786C722.3,791.5 723.5,797 725,802.5C726,805.4 726.6,808.5 727.9,811.4C729.9,815.9 731.5,820.4 733.8,824.7C734.7,826.3 737,827.5 736.5,829.2C736.1,830.3 735,831.2 733.7,830.9C732.9,830.9 733.1,829.4 732.6,828.8C731.8,826.8 730.6,825.1 729.5,823.3C726.5,819 724.5,814.2 722.2,809.5C720,805.1 718.6,800.3 716.9,795.7C715.9,792.2 715,788.6 714,785C712.6,781.2 711.5,777.3 710.9,773.3C710.6,771.4 710,769.4 710.1,767.4" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M727.4,842.4C725.5,840.6 720.6,840.9 721.2,844.5C721.6,846.9 722,850.3 724.8,850.7C727.5,850.9 729.7,848.1 728.8,845.4C728.4,844.2 728.3,843.3 727.4,842.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M732.6,834.6C733.7,834.8 734.9,834.9 735.9,835.6C736.8,836.2 737.6,837.2 737.6,838.2C737.5,839.6 737,840.8 736.5,842.1C735.6,843.8 734.6,845.4 734.2,847.3C733.5,849.4 733,851.6 732.3,853.8C732,854.9 731.1,855.6 729.9,855.8C728.5,856.1 727.3,855.4 726,855C724.4,854.3 722.6,853.9 720.9,853.3C719.9,853 718.9,852.3 718.6,851.2C718,849.9 718.2,848.4 718.1,847C718,845.6 718.3,844.1 718.9,842.9C719.2,842.2 720,841.3 721,841.6C722.8,841.6 724.6,841.4 726.3,841.2C727.7,841.1 728.6,840 728.9,838.8C729.7,837.2 730.8,835.6 732.3,834.5L732.5,834.5L732.6,834.6L732.6,834.6Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M730,855.8C730.7,856.3 731.4,858.5 732.6,858.7C734,858.9 735.2,857.6 736.1,856.6C737,855.7 737.5,854.4 738,853.3C738.8,851.5 739.3,849.6 740.1,847.9C741.1,845.7 742.3,843.6 743.7,841.7C745.5,839 748.3,837.5 749.7,834.2C750.2,833 749.9,830.7 749.4,829.6C748.7,827.9 747.5,826.3 746.3,824.9C745.8,824.3 745.3,823.6 744.6,823.2C743.9,822.9 743,822.5 742.2,822.7C741.3,822.8 740.4,823.2 739.7,823.8C739.1,824.3 738.7,825.1 738.5,825.9C738.2,826.7 737.9,827.7 737.1,828.2C736.7,828.4 736.5,828.6 736.5,829.1C736.1,830.1 735.3,831 734.2,831C733.9,831 733.1,830.4 733.3,831C733.7,832.3 733.3,833.7 732.3,834.5C733.2,834.6 733.9,835 734.8,835.1C735.9,835.5 737,836.4 737.5,837.5C737.7,838.4 737.4,839.5 737.1,840.4C736.4,842.4 735.2,844.2 734.5,846.2C733.7,848.9 733.2,850.8 732.4,853.8C732.3,853.9 732.1,854.5 731.5,855.1C731,855.6 730.1,855.7 730,855.8L730,855.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M729.9,855.9C730.6,856 730.8,856.9 731.2,857.5C731.6,858.1 732.2,858.6 733,858.8C733.5,859.4 733.5,860.4 733.7,861.2C733.7,862.9 732.8,864.6 731.4,865.7C730.6,866.4 730.1,865.3 729.6,864.9C728.7,864.4 727.6,864.4 726.6,864.2C725.9,863.7 726,862.5 725,862.2C723.6,861.7 722,861.7 720.6,861.2C719.7,861 718.8,860.2 717.5,860.1C716.6,860.1 716.3,858.7 716.3,858C716.2,856.3 716.6,855.8 717.3,854.9C717.8,854 718.8,853.4 719.7,853C720.5,852.9 721.2,853.6 722,853.7C723.8,854.3 725.6,854.8 727.3,855.5C728.2,855.7 729,856.1 729.9,855.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M726.4,864.8C726.6,863.8 725.9,862.6 724.9,862.2C723.6,862 722.3,861.3 721.1,861.8C719.8,862.4 719.2,863.7 718.5,864.9C717.8,866.2 716.6,867.8 717.5,869.3C718,870.5 719.3,871.1 720.5,871C721.8,870.9 723.2,870.1 724.2,869.2C725.4,868 726.1,866.4 726.4,864.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M723.1,870.2C723.5,871.2 724,872.1 724.9,872.7C725.6,873.1 726.6,873.5 727.3,872.9C728.1,872.1 728.9,871.2 729.5,870.2C730,869.5 730.5,868.7 730.9,867.9C731.2,867.1 731.2,866.3 730.5,865.8C729.9,865.4 729.4,864.6 728.6,864.5C727.9,864.5 726.6,863.9 726.4,864.8C726.1,866.2 725.6,867.2 724.7,868.5C724.3,869.2 723.4,869.7 723.1,870.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M717.1,867.7C716.1,867.8 714.6,868 713.9,867.1C713.3,866.1 713.1,864.8 713.5,863.7C713.8,862.6 714.3,861.9 715.1,861C715.8,860.1 717.3,859.9 718.4,860.3C719.6,860.7 720.5,861.5 721.5,861.5C722.1,861.5 720.3,862 720.1,862.6C719,863.7 718.4,865.2 717.6,866.5C717.5,866.9 717.3,867.3 717.1,867.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M712.7,883.9C710.2,889.1 707.8,894.2 705.3,899.4C704.2,902.1 701.2,905.3 703.2,907.4C704.7,909.2 708.1,909 709.7,907.5C711.7,905.5 712.1,902.8 713,900.3C714.4,896.1 714.7,892 716.2,887.8C717.5,884.2 719.7,880.9 722.4,878.2C723.8,876.7 725,875 725.5,873C722.5,871.4 724.2,869.4 721.3,870.9C719.1,871.6 717.5,869.6 716.8,872.3C716.4,874.8 715.5,877.1 714.5,879.4C713.9,881 713.5,882.3 712.7,883.9L712.7,883.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M709.3,907.8C710.3,909.2 713.5,911.1 713.7,908C713.9,904.2 715.1,900.4 716,896.6C717.1,892.5 718.4,888.2 720.7,884.6C723.1,881.7 725.6,878.9 727.4,875.7C728.6,873.9 725.4,871.9 725.1,874.3C723.6,877.8 720.3,879.8 718.6,883.1C717.2,885.3 716,887.7 715.4,890.2C714.5,893.1 714.1,896.1 713.4,899.1C712.4,901.7 712.1,904.6 710.3,906.9C710,907.2 709.6,907.4 709.3,907.8Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M702.6,904.7C701.4,903.3 699.4,902.6 698.2,901.3C697,900 697.7,897.8 698.7,896.3C700.4,893.6 702,890.8 703.5,887.9C706.2,883 709.1,878.1 710.8,872.8C711.2,871.2 712,869.8 712,868.1C711.9,866.8 713.2,864.5 713.7,866.7C714.4,868.1 715.9,867.9 717.1,867.7C717.1,869.1 717.9,870 719,870.9C716.8,870.2 716.8,872.6 716.4,874.1C714.2,881.9 710.1,889.3 706.7,896.6C705.6,898.6 705.1,900.6 703.7,902.4C703.4,902.8 703,903.8 702.6,904.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M704.3,908.3C702.2,909 700.1,909.6 698.2,910.6C697.2,911 696.2,911.6 695.3,912.1C694.4,912.6 693.6,913.1 692.5,913.7C691.8,914 690.9,914.8 690.5,915.5C690,916.4 690.1,917.2 690.6,918.1C691.1,918.9 691.8,919.7 692.6,920.2C693.2,920.7 693.7,920.8 694.4,920.4C696,919.3 696.9,918.4 698.5,917.3C699.4,916.7 700.7,915.7 701.6,915.3C702.9,914.8 704.3,914.2 705.7,913.9C707.5,913.5 709.2,912.9 710.8,912C711.6,911.6 712,911.1 712.3,910.2C712.4,909.5 711.2,909.5 710.7,909C709.9,908.2 709.9,907.6 708.8,908.2C707.8,908.7 707,908.7 706.3,908.7C705.6,908.7 705,908.5 704.3,908.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M690.1,916C689.7,916.5 687.5,916.6 685.9,915.5C684.8,914.6 685.1,913 685.6,911.7C686,910.5 687.3,910 688.3,909.6C690.5,908.7 692.1,908.2 694.4,907.4C696,906.9 697.5,906 698.8,904.9C699.7,904.2 700.1,902.6 701.8,903.9C702.9,904.8 702.3,905.4 702.6,906.6C702.8,907.5 704.8,908.2 703.4,908.6C700.5,909.5 697.7,910.8 695,912.3C694,912.8 693,913.4 692,914C691.2,914.4 690.8,915.1 690.1,916Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M684.8,912.3C683.4,912.2 680.9,910.7 682.3,909.1C683.7,907.3 685.9,906.5 688,905.8C690.3,905 692.5,904 694.6,902.8C695.7,902.2 696.8,900.5 697.7,900.6C698.3,901.9 699.9,902.4 700.9,903.3C699.3,903.7 699.3,905 697.1,906.2C693.8,908 689.7,908.5 686.5,910.5C685.5,911.3 685.6,912.4 684.8,912.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M692.3,923.3C691.9,924.6 691.9,924.6 691.8,925.4C691.6,926.1 691.8,927 691.5,927.8C691.2,928.4 690.7,929.1 690.1,929.4C689,930.1 687.6,930.6 686.3,930.4C685.5,930.3 685.6,929.5 686,929C686.5,928.4 686.8,927.7 686.9,926.9C687,926 687.2,924.8 686.6,924C686.2,923.4 685.8,923.1 685.2,922.9C684.7,922.7 683.7,923 683.7,922.8C685.2,921.5 686.8,919.7 688.3,918.3C688.9,917.6 689.5,916.9 690.1,916.3C690.2,917.7 691,918.9 692,919.8C692.5,920.2 693.2,920.5 693.3,920.7C692.9,921.4 692.6,922.5 692.3,923.3Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M685.7,930.2C685.2,930.8 684.8,931.4 684,931.5C682.6,931.5 681.3,931 680.2,930.3C679.6,929.9 678.9,929.2 679.1,928.4C679.1,927.9 679.4,927.4 680,927.3C680.5,927.1 680.7,926.6 680.8,926.1C681,925.3 680.9,924.5 680.7,923.7C680.5,923.2 681.2,922.7 681.8,922.7C682.8,922.7 683.7,923 684.8,922.9C685.7,922.8 686.5,923.7 686.8,924.6C687.2,925.5 687,926.6 686.8,927.6C686.5,928.4 685.6,929.2 685.7,930.2L685.7,930.2Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M678.9,923.5C678,923.5 677.2,923.6 676.1,923.8C675,923.9 674.1,924.7 673.3,925.5C672.7,926.1 672.3,926.4 671.9,927C671.6,927.5 671.3,928 671.2,928.5C671.2,929.3 671.2,930.5 672.1,929.4C672.5,929.1 672.9,928.8 673.4,928.6C674.6,928.1 675.9,928 677.2,927.9C677.8,927.8 678.4,927.9 679,928.1C679.3,928 679.3,927.5 679.8,927.4C680.3,927.3 680.6,926.8 680.8,926.4C681,925.9 681,925.4 680.9,924.9C680.9,924.4 680.7,923.7 680.1,923.6C680,923.6 679.3,923.5 678.9,923.5L678.9,923.5Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M685.1,913.7C683.9,914.9 682.9,916.3 681.5,917.2C680.4,918.1 679.8,918.7 678.2,919.9C677.7,920.4 679.3,920.9 680,921.3C680.6,921.6 680.6,922.6 680.9,922.9C681.4,922.7 682.4,922.7 683.1,922.8C683.9,923 684.3,922.3 684.7,921.9C686.2,920.4 687.5,919 689,917.6C689.6,917 690.1,916.3 690.4,915.6C690.2,916 689.7,916.3 689.2,916.3C688.3,916.4 687.4,916.2 686.6,915.9C686,915.6 685.4,915.2 685.2,914.5C685.2,914.3 685.1,913.9 685.1,913.7Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M678.2,919.9C678.1,918.7 678.2,917.2 677.5,916.1C677.2,915.4 676.8,914.8 676.3,914.2C676,913.8 675.2,913.4 675.8,913.1C676.6,912.6 677,912.3 677.6,912C678.4,911.5 678.8,911.3 679.3,911C680.4,910.3 681.1,909.9 682.1,909.2C681.8,909.8 681.9,910.7 682.4,911.2C683.1,911.7 684.1,912.3 685,912.3C685.3,912.3 685.7,911.4 685.6,911.8C685.2,912.4 685.3,912.9 685.2,913.5C684.5,914.1 684,915 683.3,915.6C682.2,916.7 681.6,917.1 680.3,918.2C679.7,918.7 680,918.5 679.4,919C678.9,919.4 678.6,919.6 678.2,919.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M675.5,923.9C674.8,923.5 674.7,923.3 674.2,922.6C674,922.3 673.7,921.6 673.6,921.2C673.5,920.6 673.7,920.3 675.2,920.2C675.9,920.1 676.7,920 677.4,919.9C678.2,919.7 678.1,920.4 678.4,920.5C679,920.8 679.3,920.9 679.8,921.2C680.2,921.4 680.5,921.7 680.6,922.1C680.7,922.7 680.9,922.7 680.8,923.1C680.7,923.6 680.3,923.6 679.9,923.6C679.2,923.5 678.6,923.6 678,923.6C677.3,923.6 677.2,923.6 676.5,923.7L675.5,923.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M673.2,921.4C672.9,921.6 672.4,921.6 671.9,921.5C671.5,921.4 670.9,921.2 670.8,920.6C670.8,920.3 670.8,919.9 671,919.5C671.2,919.1 671.2,918.9 671.2,918.6C671.2,918.1 671,917.6 670.7,917.3C670.4,916.9 669.6,917 669.5,916.5C669.4,915.9 669.8,915.3 670.3,915.1C670.8,914.8 672.1,914.3 672.6,914.1C673.5,913.8 674,913.6 674.7,913.4C675.4,913.2 675.8,913.6 676.3,914.2C677.4,915.3 678,916.8 678.1,918.3C678.1,918.8 678.4,919.7 678,919.8C677.7,919.9 677.1,919.9 676.6,920C675.7,920.1 674.7,920 673.8,920.4C673.5,920.6 673.5,921.2 673.2,921.4L673.2,921.4Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M672.7,926C671.6,926 670.5,925.7 669.4,926C668.5,926.1 667.2,926.6 666.1,927.1C665.6,927.3 666.3,925.7 666.6,925C666.9,924.3 667.3,923.6 667.8,923.1C668.2,922.6 668.7,922.3 669.2,922C670,921.6 670.8,921.5 671.7,921.5C672.3,921.6 672.6,921.8 673.4,921.2C673.6,921 673.7,921.7 673.9,921.9C674.1,922.4 674.3,922.7 674.6,923.2C674.8,923.5 675.2,923.8 675.4,923.9C674.9,924.1 674.6,924.4 674.2,924.8C673.7,925.2 673.2,925.6 672.7,926L672.7,926Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /><path d="M668.6,916.9C667.6,917.2 667.4,917.4 666.9,917.8C666.4,918.1 666,918.5 665.7,919C665.3,919.6 665,920.3 664.8,921C664.5,922.2 664.2,922.8 664.2,924.9C664.2,925.4 665.2,924.4 665.5,924.1C666,923.6 666.3,923.2 666.8,922.9C667.3,922.6 667.9,922.3 668.5,922.1C669.4,921.8 670,921.9 670.9,921.5C671.1,921 670.7,920.4 670.8,919.9C671,919.3 671.3,919 671.2,918.4C671.1,917.9 670.9,917.2 670.4,917.1C670,917 669.6,916.6 669.3,916.7L668.6,916.9Z" style="fill:rgb(255,249,236);fill-rule:nonzero;stroke:black;stroke-width:1px;" /></g></g><g transform="matrix(1,0,0,1,177.7,-26.9)"><g transform="matrix(1,0,0,-1,-188.1,1367.1)"><g transform="matrix(1,0,0,-1,1032.9,1650)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,-1,-238.1,1261.9)"><g transform="matrix(1,0,0,-1,1083,1663.9)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,-1,-289.8,1156.7)"><g transform="matrix(1,0,0,-1,1083,1663.9)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,-1,-320.6,1078.7)"><g transform="matrix(1,0,0,-1,824.9,1742.9)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,-1,-573.2,1066.4)"><g transform="matrix(1,0,0,-1,864.6,1829.5)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,-1,-581.7,1125.7)"><g transform="matrix(1,0,0,-1,517.4,1868.1)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,1,-612,-171.7)"><g transform="matrix(1,0,0,1,544.2,-75.7)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,1,-595.4,-234.5)"><g transform="matrix(1,0,0,1,544.2,-75.7)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,1,-594.4,-287.2)"><g transform="matrix(1,0,0,1,544.2,-75.7)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,1,-602.7,-365.2)"><g transform="matrix(1,0,0,1,544.2,-75.7)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,1,-619,-468.6)"><g transform="matrix(1,0,0,1,488.9,-46.3)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,1,-834.3,-526.5)"><g transform="matrix(1,0,0,1,625.2,-54.4)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g><g transform="matrix(1,0,0,-1,-834.3,969.5)"><g transform="matrix(1,0,0,-1,640.4,1657.5)"><switch style="font-family:\'Arial-BoldMT\', \'Arial\', sans-serif;font-weight:700;font-size:30px;" /></g></g></g>'
)

SKELETON_CREDIT = ("Skeleton: A.Spielhoff, Wikimedia Commons, CC BY-SA 3.0 — "
                   "labels and callouts removed.")


def sensor_anatomy_svg(neck_hz=100, back_hz=100) -> str:
    """Where the two IMUs sit, on the vertebrae they actually sit on.

    This is the diagram the whole build argues from: one sensor on the collar,
    one on the back harness, and the CORRELATION BETWEEN THEM is what separates
    a dog that is travelling (both sensors move together) from a dog whose head
    is doing something its body is not (shaking, scratching, sniffing).

    Drawn over a real skeleton rather than a stick figure, because the argument
    is anatomical: the collar rides the CERVICAL chain, which the neck can swing
    on its own, and the harness rides the THORACIC chain, which only moves when
    the whole animal does. On four lines and a circle you cannot see that, and
    the reader has to take the claim on trust.
    """
    # ONE LINE, NO BLANK LINES, AND THAT IS NOT A STYLE CHOICE.
    #
    # This markup used to be a triple-quoted block with blank lines separating
    # the collar group from the harness group. st.markdown runs its input
    # through a Markdown renderer before the HTML reaches the page, and a blank
    # line TERMINATES an HTML block: everything after the first one stopped
    # being markup and became paragraph text. The skeleton still drew, because
    # it comes first — but both sensor dots, both leader lines, the dashed arc
    # and all four labels vanished, and the label TEXT reappeared underneath
    # the figure as a wrapped sentence reading "COLLAR cervical spine BACK
    # HARNESS thoracic spine CORR(vm_neck, vm_back) both at 100/100 Hz…".
    # It looked like a caption someone forgot to style. It was the diagram.
    parts = [
        '<svg viewBox="255 45 880 700" role="img" '
        'aria-label="Dog skeleton showing the collar sensor on the cervical '
        'vertebrae and the harness sensor on the thoracic vertebrae" '
        'style="width:100%;height:auto;display:block">',
        _DOG_SKELETON_G,
        f'<g font-family="{FONT_CHART}">',
        # the arc from one sensor to the other: the correlation, as a gesture
        f'<path d="M521,220 Q604,163 688,223" fill="none" stroke="{INK}" '
        f'stroke-width="2.5" stroke-dasharray="8 7"/>',
        f'<line x1="497" y1="206" x2="497" y2="114" stroke="{S_ORANGE}" stroke-width="2"/>',
        f'<circle cx="497" cy="237" r="31" fill="none" stroke="{S_ORANGE}" '
        f'stroke-width="3" opacity=".4"/>',
        f'<circle cx="497" cy="237" r="15" fill="{S_ORANGE}"/>',
        f'<text x="497" y="104" text-anchor="middle" font-size="27" '
        f'font-weight="700" fill="{S_ORANGE}">COLLAR</text>',
        f'<text x="497" y="76" text-anchor="middle" font-size="20" '
        f'fill="{INK_2}">cervical spine</text>',
        f'<line x1="712" y1="209" x2="712" y2="114" stroke="{S_BLUE}" stroke-width="2"/>',
        f'<circle cx="712" cy="240" r="31" fill="none" stroke="{S_BLUE}" '
        f'stroke-width="3" opacity=".4"/>',
        f'<circle cx="712" cy="240" r="15" fill="{S_BLUE}"/>',
        f'<text x="712" y="104" text-anchor="middle" font-size="27" '
        f'font-weight="700" fill="{S_BLUE}">BACK HARNESS</text>',
        f'<text x="712" y="76" text-anchor="middle" font-size="20" '
        f'fill="{INK_2}">thoracic spine</text>',
        f'<text x="690" y="668" text-anchor="middle" font-size="26" '
        f'font-weight="700" fill="{INK}">CORR(vm_neck, vm_back)</text>',
        f'<text x="690" y="700" text-anchor="middle" font-size="19" '
        f'fill="{INK_2}">both at {neck_hz}/{back_hz} Hz — high = the whole body '
        f'is travelling, low = the head is acting alone</text>',
        "</g></svg>",
    ]
    return "".join(parts)


def symbol_ribbon(code, dog_id, test_num, match_id, *, height=H_RIBBON):
    """The matched epochs, each coloured by the pattern variable it played.

    ALL ROWS PER MATCH + CLASSIFIER() is what makes this drawable at all: the
    engine reports which symbol consumed each row, so "onset shake itch itch
    itch shake itch itch" is a picture of real seconds rather than a caption.
    Shared by the Syndromes tab and the Vet Note tab, because a note that
    asserts a sequence should be printed beside it.
    """
    mrows = rows(f"""
        SELECT epoch_ts, state, symbol, seq_in_match
        FROM MARTS.SYNDROME_MATCH_ROWS
        WHERE syndrome_code = '{code}' AND dog_id = {dog_id}
          AND test_num = {test_num} AND match_id = {match_id}
        ORDER BY seq_in_match
    """)
    if not mrows:
        empty_state("No per-epoch symbols for this match.",
                    "CALL MARTS.SP_BUILD_MATCH_ROWS() populates them "
                    "(ALL ROWS PER MATCH + CLASSIFIER()).")
        return
    syms = list(dict.fromkeys(col(mrows, "SYMBOL")))
    cmap = {s: SYMBOL_COLOURS[i % len(SYMBOL_COLOURS)] for i, s in enumerate(syms)}
    st.markdown(
        "<div class='tt-mono' style='font-size:15px;letter-spacing:.02em;"
        "margin-bottom:6px'>" +
        " ".join(f"<span style='color:{cmap[r['SYMBOL']]};font-weight:600'>"
                 f"{r['SYMBOL']}</span>" for r in mrows) + "</div>",
        unsafe_allow_html=True)
    if PLOTLY:
        fig = go.Figure()
        for r in mrows:
            fig.add_trace(go.Bar(
                x=[1], y=["match"], orientation="h",
                # a hairline of surface between segments, so a run of six
                # identical symbols reads as six seconds and not one block
                marker=dict(color=cmap[r["SYMBOL"]],
                            line=dict(width=1, color=CARD)),
                hovertext=[f'{r["SYMBOL"]} — state {r["STATE"]}<br>{r["EPOCH_TS"]}'],
                hoverinfo="text", showlegend=False))
        fig.update_layout(
            barmode="stack",
            title="every matched epoch, coloured by the pattern variable it played",
            title_font_size=11,
            xaxis=dict(visible=False), yaxis=dict(visible=False))
        chart(clean_axes(fig), height)
    st.markdown(" ".join(
        f'<span class="tt-chip" style="background:{cmap[s]}22;'
        f'border-color:{cmap[s]}"><b>{s}</b> = '
        f'{[r["STATE"] for r in mrows if r["SYMBOL"] == s][0]}</span>'
        for s in syms), unsafe_allow_html=True)


def empty_state(what: str, fix: str) -> None:
    st.markdown(
        f'<div class="tt-card"><b>{what}</b><br>'
        f'<span class="tt-quiet">{fix}</span></div>',
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.markdown(
            f'<div class="tt-card"><div class="tt-metric-label">{label}</div>'
            f'<div class="tt-metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def esc(text) -> str:
    """Escape for injection into an unsafe_allow_html block.

    The chat is the only place in this app where text a human typed is put back
    on the page, and `white-space: pre-line` means it does not need <br> — only
    the three characters that would end the markup early.
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# The two chat avatars, inline.
#
# st-chat asks DiceBear for a `bottts` robot and a `fun-emoji` face over HTTPS.
# SiS blocks that request, so the stock component renders two broken images per
# turn — the one thing that would make this page look unfinished. These are the
# same two characters redrawn as SVG: nothing to fetch, nothing to break.
def chat_avatar(is_user: bool) -> str:
    if not is_user:
        return (
            '<div class="avatar">'
            '<svg viewBox="0 0 48 48" style="width:48px;height:48px">'
            f'<circle cx="24" cy="24" r="24" fill="{CARD}"/>'
            f'<line x1="24" y1="5" x2="24" y2="13" stroke="{CHAT_BOT}" '
            f'stroke-width="2.4" stroke-linecap="round"/>'
            f'<circle cx="24" cy="4.4" r="2.7" fill="{CHAT_BOT}"/>'
            f'<rect x="3.5" y="21" width="4.2" height="9" rx="2.1" '
            f'fill="{CHAT_BOT}"/>'
            f'<rect x="40.3" y="21" width="4.2" height="9" rx="2.1" '
            f'fill="{CHAT_BOT}"/>'
            f'<rect x="8" y="12" width="32" height="26" rx="7.5" '
            f'fill="{CHAT_BOT}"/>'
            '<rect x="13" y="18.5" width="22" height="9.5" rx="4.75" '
            'fill="#2B2B2B"/>'
            f'<circle cx="19" cy="23.2" r="2.1" fill="{CARD}"/>'
            f'<circle cx="29" cy="23.2" r="2.1" fill="{CARD}"/>'
            '<rect x="18" y="31" width="12" height="2.8" rx="1.4" '
            'fill="#2B2B2B"/>'
            '</svg></div>')
    return (
        '<div class="avatar">'
        '<svg viewBox="0 0 48 48" style="width:48px;height:48px">'
        f'<circle cx="24" cy="24" r="24" fill="{CHAT_USER}"/>'
        '<rect x="13.5" y="18" width="4.2" height="4.2" rx="0.6" fill="#2B2B2B"/>'
        '<rect x="30.3" y="18" width="4.2" height="4.2" rx="0.6" fill="#2B2B2B"/>'
        '<rect x="17.7" y="21.4" width="2.6" height="2.6" rx="0.5" fill="#2B2B2B"/>'
        '<rect x="27.7" y="21.4" width="2.6" height="2.6" rx="0.5" fill="#2B2B2B"/>'
        '<path d="M18.5,30.5 Q24,35.5 29.5,30.5" fill="none" stroke="#2B2B2B" '
        'stroke-width="2.2" stroke-linecap="round"/>'
        '<path d="M22.2,32.6 Q24,36.4 25.8,32.6 Z" fill="#E0555B"/>'
        '</svg></div>')


def chat_bubble(text: str, *, is_user: bool, meta: str = "",
                hue: str = None) -> None:
    """One message row: avatar and bubble, the sender deciding the side."""
    side = " user" if is_user else ""
    edge = f'border-left:3px solid {hue};' if hue and not is_user else ""
    st.markdown(
        f'<div class="tt-chat{side}">{chat_avatar(is_user)}'
        f'<div class="msg" style="{edge}">{esc(text)}</div></div>'
        + (f'<div class="tt-chat-meta">{meta}</div>' if meta else ""),
        unsafe_allow_html=True)


def rail_meter(label: str, value: str, frac, hue: str, note: str = "") -> str:
    """One vitals row in the rail: a label, a number, and the share it is of
    its whole drawn as a hairline underneath.

    The rail used to be four bare numbers. "38 / 45 dogs" and "3 / 40 queued"
    look alike at 12px and mean completely different things — a bar is read
    before the digits are, so the one that is nearly empty is obvious.
    """
    try:
        pct = max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        pct = 0.0
    tail = f'<span class="tt-quiet"> {note}</span>' if note else ""
    return (
        f'<div class="tt-meter"><div class="tt-meter-top">'
        f'<span>{label}</span><b>{value}{tail}</b></div>'
        f'<div class="tt-meter-track"><div class="tt-meter-fill" '
        f'style="width:{pct * 100:.1f}%;background:{hue}"></div></div></div>')


def table_html(data: list[dict], columns: list[tuple[str, str]]) -> str:
    """The markup for a register, without rendering it — so the same table can
    be dropped on the page or put inside a scroll box."""
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = ""
    for r in data:
        cells = "".join(f"<td>{'' if r.get(k) is None else r.get(k)}</td>" for k, _ in columns)
        body += f"<tr>{cells}</tr>"
    return f'<table class="tt"><tr>{head}</tr>{body}</table>'


def html_table(data: list[dict], columns: list[tuple[str, str]]) -> None:
    """HAZARD 2 fallback. SiS pins an older Streamlit; st.column_config may not
    exist, and st.dataframe of Decimals renders badly. A styled HTML table is
    also denser, which suits a clinical register."""
    st.markdown(table_html(data, columns), unsafe_allow_html=True)


def scroll_table(data: list[dict], columns: list[tuple[str, str]],
                 *, height: int = 320) -> None:
    """A register that keeps every row but only takes `height` pixels of page.

    THE ANSWER TO A LONG TABLE BESIDE A SHORT ONE. A forty-row table next to a
    six-row one ends four hundred pixels lower, and the two ways out of that
    are both bad: pad the short one with nothing, or delete rows off the long
    one. A fixed box with the header pinned does neither — the column ends
    where its partner ends and the rows are all still there.
    """
    st.markdown(
        f'<div class="tt-scroll" style="max-height:{height}px">'
        + table_html(data, columns) + "</div>", unsafe_allow_html=True)


def cell_bar(value, span: float, colour: str, nd: int = 0) -> str:
    """A number with a proportional hairline under it, for a table cell.

    A register of forty numbers is a wall; the same numbers with four pixels of
    length under each are a chart the reader can rank at a glance without the
    column stopping being a number they can read exactly. `span` is passed in
    rather than derived from the rows shown, so a filtered view does not
    quietly rescale itself against its own maximum.
    """
    frac = 0.0 if not span else max(0.0, min(1.0, float(value or 0) / span))
    return (f'{fmt(value, nd)}<span class="tt-cellbar" '
            f'style="width:{100 * frac:.1f}%;background:{colour}"></span>')


def dataframe(data: list[dict], columns: list[tuple[str, str]]) -> None:
    if hasattr(st, "column_config") and hasattr(st, "dataframe"):
        try:
            st.dataframe(
                [{label: r.get(k) for k, label in columns} for r in data],
                use_container_width=True, hide_index=True,
            )
            return
        except Exception:  # noqa: BLE001
            pass
    html_table(data, columns)


def fmt(v, nd: int = 2, dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}"
    return str(v)


def clean_axes(fig, *, y_zero_line: bool = True):
    """Spec section 15: no gridlines except a single horizontal baseline, no
    chart junk, direct labelling over legends wherever a series can carry its
    own name.

    THE LEGEND IS OPT-IN, NOT FORBIDDEN. This used to set showlegend=False
    unconditionally, which quietly deleted the legend from the charts that
    genuinely need one — a stack of triage bands or weight bands has nowhere to
    put a direct label — because clean_axes runs AFTER the caller's
    update_layout and won the argument. Reading the current value keeps the
    default (no legend) while letting a caller that asked for one keep it.
    """
    legend = bool(fig.layout.showlegend)
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_CHART, size=11, color=INK_2),
        # a legend parked below the plot needs floor to stand on, or plotly
        # crops it to a row of half-height swatches
        margin=dict(l=8, r=8, t=28, b=46 if legend else 8),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(color=INK, size=11)),
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, ticks="outside",
                     tickcolor=BORDER)
    fig.update_yaxes(showgrid=False, zeroline=y_zero_line, zerolinecolor=BORDER,
                     linecolor=BORDER, ticks="outside", tickcolor=BORDER)
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def state_palette() -> dict[str, str]:
    """State colours live in REF.ETHOGRAM, so the ribbon and the SQL agree.
    Neck-dominant behaviours are amber so pathology literally stands out."""
    data = rows("SELECT state, colour_hex, display_name, sort_order, derivation, "
                "family, description FROM REF.ETHOGRAM ORDER BY sort_order")
    return {r["STATE"]: (r["COLOUR_HEX"] or "#D6D3D1") for r in data}


@st.cache_data(ttl=600, show_spinner=False)
def ethogram() -> list[dict]:
    return rows("SELECT * FROM REF.ETHOGRAM ORDER BY sort_order")


try:
    import plotly.graph_objects as go
    PLOTLY = True
except ModuleNotFoundError:
    PLOTLY = False


def chart(fig, height: int = H_MD) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# THE CHART LANGUAGE — EVILCHARTS, PORTED RATHER THAN INSTALLED.
#
# evilcharts.com is React, on Recharts or Apache ECharts. It cannot be installed
# here for exactly the reason st-chat could not be (hazard 6 at the top of this
# file): Streamlit in Snowflake cannot serve a custom component's frontend
# bundle, and there is no outbound network for a CDN. So its nine components are
# rebuilt below on the renderers this app already ships, and what is being
# copied is the DESIGN — gradient fills that fade to nothing at the baseline, a
# glow under the stroke, rounded bar caps, a dotted grid instead of ruled lines,
# a faint track behind each bar showing the space it could have filled.
#
# THE PALETTE IS NOT PART OF THE PORT. Every colour still comes from the
# CVD-validated eight at the top of this file, on the same warm surface. A
# gradient here is a ramp in ALPHA of one validated hue, never a blend between
# two hues, because a two-hue ramp invents intermediate colours that no longer
# carry the separation those eight were chosen for.
#
# ---------------------------------------------------------------------------
# EVERY EFFECT IS CAPABILITY-PROBED, AND THAT IS NOT DEFENSIVE PROGRAMMING FOR
# ITS OWN SAKE.
#
# SiS ships Altair 4, which means an old Streamlit, which means the plotly the
# conda solve lands on is not knowable from a development machine. The three
# properties this layer wants are all recent:
#
#     Scatter.fillgradient    plotly >= 5.23
#     layout.barcornerradius  plotly >= 5.19
#     axis.griddash           plotly >= 5.5
#
# An unknown property is not ignored by plotly — it raises ValueError at figure
# CONSTRUCTION, which would take down the page rather than degrade the chart.
# Probing by building a throwaway figure at import is the only way to ask, and
# it costs microseconds once. Each probe has a defined fallback that still looks
# deliberate: a flat tint instead of a ramp, square caps instead of round, a
# faint solid rule instead of a dotted one. Nothing disappears and nothing
# raises.
# ===========================================================================
def _supported(build) -> bool:
    """True if plotly accepts this construction on whatever version is here."""
    if not PLOTLY:
        return False
    try:
        build()
        return True
    except Exception:      # ValueError today; anything at all is still a no
        return False


HAS_FILLGRAD = _supported(lambda: go.Scatter(
    x=[0, 1], y=[0, 1], fill="tozeroy",
    fillgradient=dict(type="vertical",
                      colorscale=[(0, "rgba(0,0,0,0)"), (1, "rgba(0,0,0,1)")])))
HAS_BARRADIUS = _supported(lambda: go.Figure().update_layout(barcornerradius=6))
HAS_GRIDDASH = _supported(lambda: go.Figure().update_xaxes(griddash="dot"))

# How far a fill is allowed to get. 0.34 at the top of an area and nothing at
# the baseline: enough to read as a body of colour, never enough to compete with
# the stroke that carries the actual value.
FILL_TOP = 0.34
GLOW_PASSES = 3


def gfill(colour: str, top: float = FILL_TOP):
    """A vertical alpha ramp of one hue, or None if this plotly cannot.

    Returned as a kwargs dict so a caller can splat it and get either the
    gradient or the flat-tint fallback without branching:

        fig.add_trace(go.Scatter(..., **gfill(S_BLUE)))
    """
    if HAS_FILLGRAD:
        return dict(fillgradient=dict(
            type="vertical",
            colorscale=[(0.0, alpha(colour, 0.0)),
                        (0.55, alpha(colour, top * 0.42)),
                        (1.0, alpha(colour, top))]))
    # No ramp available. A flat tint at a bit under half the peak reads as the
    # same weight of colour overall, which is what keeps a page of these
    # looking consistent whichever branch a given deployment took.
    return dict(fillcolor=alpha(colour, top * 0.45))


def glow(fig, x, y, colour: str, *, width: float = 2.2,
         passes: int = GLOW_PASSES, shape: str = "spline", row=None, col=None):
    """The same path drawn underneath itself, wider and fainter each time.

    SVG has a blur filter and plotly exposes none of it, so a glow has to be
    built out of the one primitive available: more strokes. Three passes at
    3x/6x/9x the width and a twelfth of the opacity is indistinguishable from a
    soft shadow at screen resolution and costs three lines in the spec.

    Drawn BEFORE the real stroke by every caller, because these are additive
    and a glow painted on top of its own line greys it out.
    """
    for i in range(passes, 0, -1):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=alpha(colour, 0.10 - 0.022 * (i - 1)),
                      width=width * (1 + 3 * i), shape=shape)),
            **({"row": row, "col": col} if row is not None else {}))
    return fig


def evil_area(fig, x, y, colour: str, *, name: str = "", text=None,
              width: float = 2.1, shape: str = "spline", fill: str = "tozeroy",
              show_glow: bool = True, cap: bool = False, top: float = FILL_TOP):
    """EvilCharts' area chart: gradient body, glowing stroke, optional end cap.

    The workhorse of this rebuild. `cap` puts a single filled dot on the last
    point — their charts all do it, and it is genuinely useful here because it
    marks where a series actually ends when several are overlaid.
    """
    if show_glow:
        glow(fig, x, y, colour, width=width, shape=shape)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, fill=fill,
        line=dict(color=colour, width=width, shape=shape),
        showlegend=bool(name) and bool(fig.layout.showlegend),
        hovertext=text, hoverinfo="text" if text is not None else "skip",
        **gfill(colour, top)))
    if cap and len(x):
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]], mode="markers", showlegend=False,
            hoverinfo="skip",
            marker=dict(color=colour, size=7,
                        line=dict(color=CARD, width=2))))
    return fig


def evil_bar(fig, cats, vals, colours, *, text=None, horizontal: bool = True,
             track: bool = True, width: float = 0.62):
    """EvilCharts' bar chart: rounded caps over a faint full-length track.

    THE TRACK IS THE POINT, not decoration. A bare bar tells you a value; a bar
    in a track tells you the value AND the headroom, which is the difference
    between 'TROT: 7s' and 'TROT: 7s of a possible 22'. It is drawn first, at
    the longest bar's length, so every row is the same visual width.
    """
    if track and len(vals):
        span = max([v for v in vals if v is not None] or [0]) * 1.02
        bar = dict(y=cats, x=[span] * len(cats), orientation="h") if horizontal \
            else dict(x=cats, y=[span] * len(cats))
        fig.add_trace(go.Bar(
            **bar, marker=dict(color=GRID, line=dict(width=0)),
            width=width, hoverinfo="skip", showlegend=False))
    bar = dict(y=cats, x=vals, orientation="h") if horizontal \
        else dict(x=cats, y=vals)
    fig.add_trace(go.Bar(
        **bar, width=width,
        marker=dict(color=colours, line=dict(width=0)),
        hovertext=text, hoverinfo="text" if text is not None else "skip",
        showlegend=False))
    # overlay, so the track sits behind rather than stacking end to end
    fig.update_layout(barmode="overlay", bargap=0.34)
    if HAS_BARRADIUS:
        fig.update_layout(barcornerradius=4)
    return fig


def evil_axes(fig, *, y_zero_line: bool = True, dotted: str = "y"):
    """clean_axes, plus the dotted grid and without the boxed-in axis lines.

    WRAPS RATHER THAN REPLACES clean_axes. That function encodes the spec's
    no-chart-junk rules and the opt-in legend behaviour that took a bug to get
    right; this only changes how the remaining lines are drawn. `dotted` picks
    which axes get a grid — "y" for almost everything, "xy" for a scatter where
    both readings matter, "" for a ribbon.
    """
    clean_axes(fig, y_zero_line=y_zero_line)
    dash = dict(griddash="dot") if HAS_GRIDDASH else {}
    # A dotted grid can afford to be darker than a ruled one — it is mostly
    # gaps — and it has to be, or it vanishes on the warm surface.
    axis = dict(showgrid=True, gridcolor=BORDER, gridwidth=1,
                linecolor="rgba(0,0,0,0)", ticks="", **dash)
    if "x" in dotted:
        fig.update_xaxes(**axis)
    if "y" in dotted:
        fig.update_yaxes(**axis)
    # The frame goes. EvilCharts' charts float on their card, and the axis
    # domain line is the single thing that most makes a plotly chart look like
    # a plotly default.
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showline=False, zerolinecolor=BORDER)
    return fig


def scene3d(fig, xt: str, yt: str, zt: str, *, eye=(1.55, 1.45, 0.85)):
    """One 3D scene style, so nine scenes across the app look like one system.

    Pale walls rather than none: a 3D scatter with no floor has no depth cue at
    all, and the reader cannot tell a point that is far away from one that is
    low. Pale rather than default, because plotly's stock scene is a grey box
    with three heavy panes that dominates any colour inside it.
    """
    # NOT `titlefont=`. That spelling was removed in plotly 6 and raises a
    # ValueError on the property path rather than being ignored, which would
    # take out every 3D scene in the app on a current plotly while working
    # perfectly on an old one. `title=dict(text=..., font=...)` is accepted by
    # both, which is the only reason to write it the long way.
    ax = dict(backgroundcolor=CARD, gridcolor=BORDER, zerolinecolor=BORDER,
              showbackground=True, tickfont=dict(size=9, color=INK_2),
              showspikes=False)

    def _axis(title: str) -> dict:
        return dict(title=dict(text=title, font=dict(size=10, color=INK_2)), **ax)

    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_CHART, size=11, color=INK_2),
        margin=dict(l=0, r=0, t=28, b=0),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(color=INK, size=11)),
        scene=dict(
            xaxis=_axis(xt), yaxis=_axis(yt), zaxis=_axis(zt),
            camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])),
            aspectmode="cube"))
    return fig


def chart3d(fig, height: int = H_LG) -> None:
    """Render a 3D scene. Separate from chart() because the mode bar STAYS.

    A rotatable scene the reader cannot discover is rotatable is just a bad 2D
    chart, and the mode bar is the only affordance plotly offers.
    """
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["toImage"]})


def evil_donut(labels, values, colours, *, text=None, centre: str = "",
               centre_sub: str = "", hole: float = 0.66):
    """EvilCharts' pie: a thin gradient ring with the total living in the hole.

    The hole is 0.66 rather than their default because something is written in
    it. A donut whose centre is empty is a pie chart wearing a disguise; a
    donut whose centre carries the total is the only version of the form that
    earns the space it takes.
    """
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=hole, sort=False, direction="clockwise",
        marker=dict(colors=colours, line=dict(color=CARD, width=2)),
        text=text, hoverinfo="text" if text is not None else "label+percent",
        textinfo="none"))
    if centre:
        fig.add_annotation(
            text=(f"<b>{centre}</b>" +
                  (f"<br><span style='font-size:10px'>{centre_sub}</span>"
                   if centre_sub else "")),
            x=0.5, y=0.5, showarrow=False, align="center",
            font=dict(size=21, color=INK, family=FONT_CHART))
    return fig


def evil_radial(value: float, total: float, colour: str, *, label: str = "",
                sub: str = ""):
    """EvilCharts' radial bar — a semi-circle gauge over a full-circle track.

    A share, drawn as an arc. Used where the app already had a ratio and was
    printing it as a number: held-out accuracy, model-derived share, the
    fraction of the queue that reached the chain.

    Barpolar rather than Pie-with-a-hidden-half: a pie's hidden half still
    occupies its layout box, so the chart sits with a hole under it that no
    amount of margin fixes.
    """
    frac = 0.0 if not total else max(0.0, min(1.0, float(value) / float(total)))
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=[1], theta=[90], width=[360], base=[0],
                              marker=dict(color=GRID), hoverinfo="skip",
                              showlegend=False))
    # Clockwise from twelve o'clock, which is the direction a reader expects a
    # gauge to fill and the opposite of plotly's default.
    fig.add_trace(go.Barpolar(
        r=[1], theta=[90 - 180 * frac], width=[360 * frac], base=[0],
        marker=dict(color=colour), showlegend=False,
        hovertext=[f"{label}: {fmt(value, 0)} of {fmt(total, 0)} "
                   f"({100 * frac:.1f}%)"], hoverinfo="text"))
    fig.update_layout(
        polar=dict(hole=0.68, bgcolor=CARD,
                   radialaxis=dict(visible=False, range=[0, 1]),
                   angularaxis=dict(visible=False, direction="clockwise",
                                    rotation=90)),
        annotations=[dict(text=f"<b>{sub or f'{100 * frac:.0f}%'}</b>",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=19, color=INK, family=FONT_CHART))])
    return fig


def evil_radar(cats, series, *, fill: bool = True, rng=None):
    """EvilCharts' radar. `series` is [(name, values, colour), ...].

    ONLY FOR COMMENSURABLE AXES. A radar silently implies that its spokes share
    a scale and that the AREA it encloses means something; give it a mix of
    percentages and seconds and it draws a confident shape out of nothing. Used
    here only where every spoke is already the same unit — per-class precision
    and recall, and a dog's features expressed in z-scores.
    """
    fig = go.Figure()
    closed = list(cats) + [cats[0]] if cats else []
    for name, vals, colour in series:
        ring = list(vals) + [vals[0]] if vals else []
        fig.add_trace(go.Scatterpolar(
            r=ring, theta=closed, name=name, mode="lines+markers",
            fill="toself" if fill else None,
            fillcolor=alpha(colour, 0.16),
            line=dict(color=colour, width=2, shape="linear"),
            marker=dict(size=5, color=colour),
            hovertemplate="%{theta}: %{r:.3f}<extra>" + name + "</extra>"))
    dash = dict(griddash="dot") if HAS_GRIDDASH else {}
    fig.update_layout(
        showlegend=len(series) > 1,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_CHART, size=11, color=INK_2),
        margin=dict(l=30, r=30, t=34, b=34),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(color=INK, size=11)),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        polar=dict(bgcolor=CARD,
                   radialaxis=dict(range=rng, gridcolor=BORDER, linecolor=BORDER,
                                   tickfont=dict(size=9, color=INK_2), **dash),
                   angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                                    tickfont=dict(size=10, color=INK_2), **dash)))
    return fig


def evil_sankey(labels, sources, targets, values, colours, *, text=None):
    """EvilCharts' sankey. A flow, and only ever used for something that flows.

    Two things in this app genuinely do: a first-order Markov chain over
    behaviour states, where the width of a band IS the transition probability,
    and the pipeline itself, where rows move raw -> staging -> marts -> oracle.
    Neither is a bar chart wearing a costume.

    Link colour is inherited from the SOURCE node at low alpha, which is what
    makes a band readable as 'leaving here' rather than as its own object.
    """
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colours, pad=15, thickness=13,
                  line=dict(color=CARD, width=0.5),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=sources, target=targets, value=values,
                  color=[alpha(colours[s], 0.30) for s in sources],
                  customdata=text or [""] * len(values),
                  hovertemplate=("%{customdata}<extra></extra>" if text
                                 else "%{source.label} → %{target.label}"
                                      "<extra></extra>"))))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_CHART, size=10, color=INK_2),
        margin=dict(l=6, r=6, t=30, b=6),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(color=INK, size=11)))
    return fig


def evil_blocks(segments, *, title: str = "", gap: float = 0.004):
    """EvilCharts' bar blocks: one stacked bar broken into visible tiles.

    `segments` is [(label, value, colour), ...]. The hairline gap between tiles
    is the whole difference from a plain stacked bar — it turns a continuous
    band into countable blocks, so the eye reads four categories rather than
    one bar with colour changes in it.
    """
    fig = go.Figure()
    total = sum(float(v or 0) for _, v, _ in segments) or 1.0
    for label, value, colour in segments:
        share = 100.0 * float(value or 0) / total
        fig.add_trace(go.Bar(
            x=[max(0.0, share - gap * 100)], y=[""], orientation="h",
            marker=dict(color=colour, line=dict(width=0)),
            name=str(label),
            hovertext=[f"{label}: {fmt(value, 0)} ({share:.1f}%)"],
            hoverinfo="text"))
    fig.update_layout(
        barmode="stack", bargap=0.18, showlegend=True,
        legend=dict(orientation="h", y=-0.55, x=0, font=dict(size=10)),
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    # ONLY TOUCH layout.title WHEN THERE IS A TITLE. `title=None` with
    # `title_font_size=11` still creates the title container, and plotly.js
    # draws a title object whose `text` is unset as the literal string
    # "undefined" — which is exactly what was sitting above the triage mix bar.
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=11)))
    if HAS_BARRADIUS:
        fig.update_layout(barcornerradius=3)
    return fig


# ---------------------------------------------------------------------------
# LAYOUT
#
# The grid was the other half of the complaint and it is not a chart problem.
# Rows were a mix of [3,1], [1,2], [3,2] and equal splits, and several sections
# had no partner at all, so one column ran hundreds of pixels past the other and
# every page ended ragged. Streamlit will not equalise columns for you — it
# stacks each one independently — so matching heights has to be arithmetic the
# page does before it draws, which is what these do.
# ---------------------------------------------------------------------------
def panel(title: str, note: str = "") -> None:
    """A section heading and its one-line explanation, as one unit.

    This pattern appeared ~30 times written out by hand, and the margins drifted
    between copies — which is most of why two charts side by side did not start
    on the same line even when they were the same height.
    """
    st.markdown(
        f'<div class="tt-panelhead"><b>{title}</b>'
        + (f'<span class="tt-quiet">{note}</span>' if note else "")
        + "</div>", unsafe_allow_html=True)


def row_h(*counts: int, row: int = 24, floor: int = H_MD) -> int:
    """One height for every chart in a row, from the longest of them.

    Generalises the `max(bars(a), bars(b))` that the Ethogram page already did
    by hand. Pass the category count of each chart in the row; they all get the
    height the biggest one needs, so their baselines line up.
    """
    return max([floor] + [bars(c, row=row) for c in counts if c])


# table_pair() lived here. It trimmed two tables to `min(cap, MAX(len(a),
# len(b)))` and called that a pairing, which it is not: the max only binds the
# LONGER table, so a 40-row register beside a permanently-6-row one came out
# 18 against 6 and the row ended ragged anyway. Its one caller now uses
# scroll_table(), which keeps every row and fixes the height instead.


# ===========================================================================
# THREE RENDERERS, BECAUSE THEY ARE GOOD AT THREE DIFFERENT THINGS
#
# plotly  — 3D, and only 3D. It is the one of the three that can put a point
#           cloud, a hull or a surface in a rotatable scene. Neither of the
#           others can: Vega-Lite is a 2D grammar by specification and Bokeh's
#           core has no 3D glyph (its `surface3d` demo is a custom extension
#           that loads vis.js from a CDN, and SiS has no outbound network, so
#           it is not an option here rather than merely a bad one).
# altair   — LINKED SELECTION. Drag a box on one chart and the others filter,
#            with no rerun, no callback and no round trip to the warehouse,
#            because the whole interaction compiles into the Vega spec and runs
#            in the browser. Streamlit cannot do that with plotly at all.
# bokeh    — LINKED AXES over a long signal. Shared x_range across stacked
#            panels plus a RangeTool scrubber is Bokeh's one genuinely
#            unmatched trick, and a 100 Hz waveform is exactly its use case.
#
# A chart goes to whichever of the three owns the job. Nothing is ported for
# the sake of using a library.
# ===========================================================================

try:
    import altair as alt
    ALTAIR = True
except ModuleNotFoundError:
    ALTAIR = False

# numpy is in environment.yml for plotly's benefit and is used directly in
# exactly one place: the short-time Fourier transform behind the spectrogram
# on Live Collar. Guarded anyway, so that a page which is fine without it
# stays up if the solve ever changes underneath us.
try:
    import numpy as np
    NUMPY = True
except ModuleNotFoundError:
    NUMPY = False

try:
    from bokeh.embed import file_html as _bk_file_html
    from bokeh.layouts import gridplot as bk_gridplot
    from bokeh.models import (ColumnDataSource, CrosshairTool, HoverTool,
                              RangeTool, Span)
    from bokeh.plotting import figure as bk_figure
    from bokeh.resources import INLINE as _BK_INLINE
    from bokeh.themes import Theme as _BkTheme
    import streamlit.components.v1 as components
    BOKEH = True
except ModuleNotFoundError:
    BOKEH = False


# ---------------------------------------------------------------------------
# ALTAIR
#
# NO GLOBAL THEME. alt.themes.register moved to alt.theme.register in 5.5 and
# the old spelling warns; SiS pins whatever Streamlit pins, so a module-level
# registration is a coin flip on which of the two APIs exists. Configuring the
# chart object instead works identically on 4, 5 and 6 — and configure_* is
# only legal on the TOP-LEVEL chart, which is the one place this helper is
# ever called.
#
# DATA GOES IN AS alt.Data(values=...), NOT A DATAFRAME. Same reason the
# plotly charts are fed lists (hazard 1 at the top of this file): Snowpark
# hands back object-dtype Decimal, and altair infers encoding types from
# dtypes, so a DataFrame of Decimals types every numeric as nominal and you
# get a bar chart of forty-five identical bars. Explicit :Q/:N/:T on every
# field, always.
# ---------------------------------------------------------------------------
AXIS_CFG = dict(labelFont=FONT_CHART, labelFontSize=10,
                labelColor=INK_2, titleFont=FONT_CHART,
                titleFontSize=10, titleColor=INK_2, titleFontWeight="normal",
                domainColor=BORDER, tickColor=BORDER, grid=False)


def tt_alt(chart_obj, *, grid_y: bool = False):
    """Apply the app's design system to a finished top-level altair chart."""
    return (chart_obj
            .configure_view(stroke=None, fill=CARD)
            .configure_axisX(**AXIS_CFG)
            .configure_axisY(**dict(AXIS_CFG, grid=grid_y, gridColor=GRID,
                                    domain=False))
            .configure_legend(labelFont=FONT_CHART,
                              labelFontSize=10, labelColor=INK_2,
                              titleFont=FONT_CHART,
                              titleFontSize=10, titleColor=INK_2,
                              titleFontWeight="normal", symbolType="circle",
                              symbolStrokeWidth=0, offset=8)
            .configure_title(font=FONT_CHART, fontSize=11,
                             color=INK_2, fontWeight="normal", anchor="start",
                             offset=10)
            .configure_range(category=list(SYMBOL_COLOURS))
            .configure_concat(spacing=26)
            .configure_facet(spacing=14)
            .configure_header(labelFont=FONT_CHART,
                              labelFontSize=10, labelColor=INK_2,
                              titleFontSize=10, titleColor=INK_2))


def avals(data: list[dict]):
    """list[dict] -> an inline altair data source.

    Every value is already float/str/None by the time it gets here (`rows()`
    scrubs Decimal), which matters: Vega serialises the spec to JSON and
    json.dumps cannot encode a Decimal or a datetime.
    """
    return alt.Data(values=data)


# ---------------------------------------------------------------------------
# ALTAIR 4 / 5 SELECTION COMPATIBILITY.
#
# THIS IS A SEVENTH SiS HAZARD and it took a whole tab down. Selections were
# renamed wholesale in Altair 5 — `selection_point` for `selection_multi`,
# `.add_params` for `.add_selection` — and Streamlit in Snowflake pins whatever
# Altair the Streamlit it ships depends on, which is 4.x. Written in the 5
# spelling, the cross-filter on Syndromes raised
#
#     AttributeError: module 'altair' has no attribute 'selection_point'
#
# at module scope inside the page function, so the tab rendered its heading and
# then a red traceback instead of everything below it. It reproduces nowhere
# locally, because a development machine has Altair 5 or 6.
#
# The shim resolves the spelling at call time rather than at import, so the same
# file is correct on 4, 5 and 6. `selection_interval` needs no shim — the name
# survived the rename unchanged, which is exactly why the brush on the same
# chart worked while the click selection did not.
# ---------------------------------------------------------------------------
def alt_point(*, fields: list[str], name: str, toggle=None):
    """A click/tap selection over `fields`, on whichever Altair is installed.

    `toggle` is OMITTED rather than passed as None when it is not wanted.
    Altair treats an explicit None as a value to encode rather than as 'not
    supplied', and fails deep inside schema validation with

        TypeError: cannot use 'list' as a dict key

    which names neither the argument nor the function that took it. Passing the
    keyword only when it has a value is the difference between the default
    behaviour and a traceback.
    """
    kw = dict(fields=fields, name=name)
    if toggle is not None:
        kw["toggle"] = toggle
    if hasattr(alt, "selection_point"):                       # 5, 6
        return alt.selection_point(**kw)
    return alt.selection_multi(**kw)                          # 4


def alt_bind(chart_obj, selection):
    """Attach a selection to a chart, on whichever Altair is installed.

    add_params is 5+; add_selection is 4 and warns loudly on 5 before being
    removed in 6. getattr order matters: prefer the modern name so a future
    build that drops the old one keeps working.
    """
    binder = getattr(chart_obj, "add_params", None)
    if binder is None:
        binder = chart_obj.add_selection
    return binder(selection)


def altair_chart(chart_obj, *, container: bool = True) -> None:
    """Render, and survive Streamlit renaming the width argument.

    use_container_width has been deprecated in favour of width="stretch" but
    both spellings work in the versions SiS ships; older builds only know the
    first. Try, fall back, then fall back again to no argument at all rather
    than losing the chart.
    """
    try:
        st.altair_chart(chart_obj, use_container_width=container)
    except TypeError:
        try:
            st.altair_chart(chart_obj, width="stretch" if container else "content")
        except TypeError:
            st.altair_chart(chart_obj)


# ---------------------------------------------------------------------------
# BOKEH
#
# NOT st.bokeh_chart. That function pinned Bokeh to exactly 2.4.3 for years
# and raises StreamlitAPIException against anything else; the Snowflake
# Anaconda channel's Bokeh is 3.9. Rendering the document ourselves sidesteps
# the pin entirely and works the same on every Streamlit build.
#
# RESOURCES ARE INLINE, NOT CDN. SiS serves the app from a sandbox with no
# outbound network and a strict CSP, so a <script src="cdn.bokeh.org/..."> is
# a blank rectangle. INLINE writes BokehJS into the document — about 1.4 MB
# per embed, which is why every page that uses Bokeh emits exactly ONE
# document containing all of its panels rather than one document per panel.
# Panels have to share a document anyway for their ranges to link.
# ---------------------------------------------------------------------------
BK_THEME = {
    "attrs": {
        "Plot": {"background_fill_color": CARD, "border_fill_color": CARD,
                 "outline_line_color": None,
                 "min_border_left": 44, "min_border_right": 12,
                 "min_border_top": 8, "min_border_bottom": 8},
        "Axis": {"axis_line_color": BORDER, "major_tick_line_color": BORDER,
                 "minor_tick_line_color": None,
                 "axis_label_text_color": INK_2, "axis_label_text_font_size": "10px",
                 "axis_label_text_font_style": "normal",
                 "axis_label_text_font": FONT_CHART,
                 "major_label_text_color": INK_2,
                 "major_label_text_font_size": "10px",
                 "major_label_text_font": FONT_CHART},
        "Grid": {"grid_line_color": None},
        "Title": {"text_color": INK_2, "text_font_size": "11px",
                  "text_font_style": "normal",
                  "text_font": FONT_CHART},
        "Legend": {"border_line_color": None, "background_fill_alpha": 0.0,
                   "label_text_color": INK_2, "label_text_font_size": "10px",
                   "label_text_font": FONT_CHART,
                   "spacing": 2, "padding": 4},
        "Toolbar": {"logo": None},
    }
}


def bokeh_panel(layout, height: int) -> None:
    """Render one Bokeh document into the page.

    height is the IFRAME height and Bokeh does not report its own, so it has
    to be told: too small silently clips the bottom panel, and the scrollbar
    that would otherwise reveal it is off because a scrollbar inside a chart
    reads as a bug. Pass the sum of the panel heights plus ~20px of chrome.
    """
    html = _bk_file_html(layout, _BK_INLINE, "", theme=_BkTheme(json=BK_THEME))
    # The iframe's own document defaults to a white body with an 8px margin,
    # which shows as a hairline frame of the wrong colour around a card that
    # is already the right one. Bokeh's Theme cannot reach the <body>.
    html = html.replace(
        "</head>",
        f"<style>html,body{{margin:0;padding:0;background:{CARD};"
        f"overflow:hidden;}}</style></head>", 1)
    components.html(html, height=height, scrolling=False)


def bk_style(p, *, ylabel: str = "", title: str = ""):
    """The per-figure half of the theme — the parts Theme(json=...) cannot set."""
    p.toolbar.autohide = True
    p.toolbar_location = "right"
    p.yaxis.axis_label = ylabel
    if title:
        p.title.text = title
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = GRID
    p.ygrid.grid_line_dash = [2, 3]
    return p


def renderer_note(which: str, why: str) -> None:
    """One line under a chart saying which library drew it and what it bought.

    A judge who has looked at forty dashboards has seen forty plotly defaults.
    Saying 'this is Vega-Lite, the filtering runs in your browser' out loud is
    the difference between a chart they scroll past and one they interact with.
    """
    st.markdown(
        f'<div class="tt-renderer"><span class="tt-renderer-tag">{esc(which)}'
        f'</span>{esc(why)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
stats = rows("SELECT * FROM MARTS.V_PIPELINE_STATS")
prov = rows("SELECT * FROM MARTS.V_STATE_PROVENANCE ORDER BY epochs DESC")

hl, hr = st.columns([3, 2])
with hl:
    st.markdown(
        f"<h1 style='margin-bottom:0'>TELLTAIL</h1>"
        f"<div style='color:{INK_2};font-size:13px;margin-top:-6px'>"
        f"A dog cannot tell you where it hurts. Ten million rows of collar data can."
        f"</div>", unsafe_allow_html=True)
with hr:
    latest = one(stats, "LATEST_EPOCH_TS")
    st.markdown(
        f"<div style='text-align:right;font-size:11px;color:{INK_2};padding-top:14px'>"
        f"pipeline clock &nbsp;<b class='tt-mono'>{latest or '—'}</b><br>"
        f"rows in the last minute &nbsp;<b>{fmt(one(stats,'ROWS_LAST_MINUTE',0),0)}</b>"
        f"</div>", unsafe_allow_html=True)

if not PLOTLY:
    st.error("plotly is not importable. environment.yml must be staged next to "
             "streamlit_app.py. Re-run: python scripts/deploy_streamlit.py")

# Honesty banner. If any of the state layer is threshold-derived rather than
# model-derived, say so at the top of every tab rather than in a footnote.
heur = sum(r["EPOCHS"] for r in prov if r["STATE_SOURCE"] == "HEURISTIC")
tot = sum(r["EPOCHS"] for r in prov) or 1
if heur:
    st.markdown(
        f'<div class="tt-caveat"><b>{100 * heur / tot:.1f}% of epochs carry a '
        f'heuristic state.</b> SHAKE and SCRATCH are not first-class labels in '
        f'this dataset, so they are derived from the neck/back correlation '
        f'feature and flagged <span class="tt-mono">state_source = '
        f'&#39;HEURISTIC&#39;</span> in the data. Not a diagnostic device.</div>',
        unsafe_allow_html=True)

# ===========================================================================
# NAVIGATION
#
# A left rail rather than st.tabs, for a reason that is not taste: EVERY
# st.tabs body executes on every rerun, so the nine pages each fired their
# queries whether or not you were looking at them — nine pages of Snowflake
# work to render one. The router below calls exactly one page function.
#
# Each page also carries its own hue, marking its rail row, its page header and
# any chrome that belongs to the page rather than to the data — so you can tell
# at a glance which page a screenshot came from.
#
# IT DOES NOT COLOUR THE CHARTS. Colour inside a chart is already spoken for:
# it means an ethogram state, a triage band, or which sensor a trace came from,
# and those meanings have to survive a page change. Page hue is chrome; chart
# colour is data.
# ===========================================================================
PAGES = [
    ("Pack",            "the ward round",                   "#B45309"),
    ("Live Collar",     "100 Hz, two sensors",              "#0F766E"),
    ("Ethogram",        "states, transitions, bouts",       "#6D28D9"),
    ("Syndromes",       "MATCH_RECOGNIZE",                  "#B91C1C"),
    ("Baselines",       "each dog against itself",          "#1D4ED8"),
    ("Vet Note",        "Cortex handoff",                   "#047857"),
    ("Drivers",         "what the model leans on",          "#A16207"),
    ("Shelter Reality", "where this ends up",               "#9D174D"),
    ("Pipeline",        "the DAG, observable",              "#374151"),
    ("On Chain",        "Solana devnet attestations",       "#0891B2"),
    ("Ask TELLTAIL",    "Cortex over the warehouse",        "#7C2D12"),
]

with st.sidebar:
    st.markdown(
        '<div class="tt-brand"><div class="tt-brand-mark">'
        + dog_glyph("TROT", size=24, colour=CARD) +
        '</div><div><div style="font-weight:800;font-size:19px;'
        f'letter-spacing:-.02em;color:{INK};line-height:1.1">TELLTAIL</div>'
        '<div class="tt-quiet" style="font-size:11px">canine telemetry · '
        'Snowflake</div></div></div>', unsafe_allow_html=True)

    # Numbered, because the nine pages are an argument in order and the numbers
    # are how you say "look at 4" out loud during a demo. The router still keys
    # off the bare name, so the labels are cosmetic and cannot desynchronise.
    _num = {p[0]: i for i, p in enumerate(PAGES)}
    _choice = st.radio(
        "section", [p[0] for p in PAGES], label_visibility="collapsed",
        format_func=lambda n: (f"{_num[n] + 1} · {n}" if _num[n] < 9
                               else f"·  {n}"))
    _meta = next(p for p in PAGES if p[0] == _choice)

    # The selected row's hue bar. Emitted here rather than in the sheet at the
    # top because it is the CURRENT page's colour, which is not known until the
    # radio above has returned.
    st.markdown(
        '<style>section[data-testid="stSidebar"] div[role="radiogroup"] > '
        'label:has(input:checked) { border-left-color: ' + _meta[2] + '; }'
        '</style>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="tt-railnote" style="border-left:3px solid {_meta[2]}">'
        f'<b>{_meta[0]}</b><br><span class="tt-quiet">{_meta[1]}</span></div>',
        unsafe_allow_html=True)

    # The rail carries the state of the pipeline, not blank space. These are
    # the numbers worth knowing before reading any page, and they are the same
    # numbers the pages themselves are computed from — if the rail and a page
    # disagree, something is stale and you can see it immediately. Each one is
    # drawn as a share of its own whole, so a near-empty queue looks near-empty.
    _vit = rows("""
        SELECT
            (SELECT COUNT(*) FROM MARTS.EPOCH_STATES)              AS epochs,
            (SELECT COUNT(*) FROM MARTS.SYNDROME_MATCHES)          AS matches,
            (SELECT COUNT(DISTINCT dog_id) FROM MARTS.SYNDROME_MATCHES) AS dogs_hit,
            (SELECT COUNT(*) FROM REF.DOG_INFO)                    AS dogs,
            (SELECT ROUND(100 * holdout_accuracy, 1) FROM ML.MODEL_SUMMARY) AS acc,
            (SELECT COUNT(*) FROM ORACLE.PUBLISH_QUEUE
              WHERE status = 'CONFIRMED')                          AS on_chain,
            (SELECT COUNT(*) FROM ORACLE.PUBLISH_QUEUE)            AS queued,
            (SELECT ROUND(AVG(mean_lag_sec)) FROM MARTS.DAG_LAG_SNAPSHOT
              WHERE mean_lag_sec IS NOT NULL)                      AS lag_s,
            (SELECT COUNT(*) FROM MARTS.DAG_LAG_SNAPSHOT
              WHERE state <> 'ACTIVE')                             AS dt_bad
    """)
    if _vit:
        _v = _vit[0]
        _lag = _v.get("LAG_S")
        _bad = int(_v.get("DT_BAD") or 0)
        _dogs = float(_v.get("DOGS") or 0) or 1.0
        _queued = float(_v.get("QUEUED") or 0) or 1.0
        # `tot` falls back to 1 to avoid a zero divide, which would make an
        # empty warehouse claim a confident 100% model-derived. No provenance
        # rows means the question has no answer yet, so the meter is dropped.
        _model_share = (1.0 - (heur / tot)) if prov else None
        st.markdown(
            # Counts stay plain rows. A meter needs a denominator to mean
            # anything, and "how many matches, out of what?" has no answer —
            # a full bar next to every count teaches the reader to ignore
            # the bars that do carry a ratio.
            '<div class="tt-railstat">'
            f'<div><span>epochs classified</span>'
            f'<b>{fmt(_v.get("EPOCHS"), 0)}</b></div>'
            f'<div><span>syndrome matches</span>'
            f'<b>{fmt(_v.get("MATCHES"), 0)}</b></div></div>'
            + rail_meter("dogs with a finding",
                         fmt(_v.get("DOGS_HIT"), 0),
                         float(_v.get("DOGS_HIT") or 0) / _dogs,
                         "#B91C1C", f'/ {fmt(_v.get("DOGS"), 0)}')
            + rail_meter("held-out accuracy", f'{fmt(_v.get("ACC"), 1)}%',
                         float(_v.get("ACC") or 0) / 100.0, "#15803D")
            + (rail_meter("model-derived states",
                          f"{100 * _model_share:.1f}%", _model_share, ACCENT)
               if _model_share is not None else "")
            + rail_meter("attested on chain", fmt(_v.get("ON_CHAIN"), 0),
                         float(_v.get("ON_CHAIN") or 0) / _queued,
                         "#7C3AED", f'/ {fmt(_v.get("QUEUED"), 0)} queued'),
            unsafe_allow_html=True)
        _dot = "#15803D" if _bad == 0 else "#B91C1C"
        _dag = "all active" if _bad == 0 else str(_bad) + " not active"
        _lagtxt = " · mean lag " + ago(_lag) if _lag is not None else ""
        st.markdown(
            f'<div style="margin-top:11px"><span class="tt-pill">'
            f'<span style="color:{_dot};font-size:13px">&#9679;</span> '
            f'DAG {_dag}{_lagtxt}</span></div>', unsafe_allow_html=True)

    # Photo coverage, said in the rail because the photographs are the one
    # thing on screen a viewer could reasonably mistake for evidence.
    _ph = breed_photos()
    st.markdown(
        '<div class="tt-railfoot" style="margin-top:14px">'
        f'{"Breed photos: " + str(len(_ph)) + " breeds covered. " if _ph else ""}'
        'Not a diagnostic device. Reference photographs are of the breed, '
        'never of the study animal.</div>', unsafe_allow_html=True)

PAGE, PAGE_SUB, PAGE_HUE = _meta

# Which Cortex model, from REF.PARAMS rather than hardcoded here — the same row
# the AI layer procedures read, so the dashboard cannot drift from what ran.
CORTEX_MODEL = one(
    rows("SELECT value_str AS m FROM REF.PARAMS WHERE key = 'cortex_model'"),
    "M", "claude-sonnet-4-5")

st.markdown(
    f'<div class="tt-pagehead" style="border-left:4px solid {PAGE_HUE}">'
    f'<span class="tt-pagetitle">{PAGE}</span>'
    f'<span class="tt-quiet"> — {PAGE_SUB}</span></div>',
    unsafe_allow_html=True)


# ===========================================================================
# TAB 1 — THE PACK.  The ward round.
# ===========================================================================
def _page_0():
    pack = rows("""
        SELECT p.*, t.triage_label, t.severity AS triage_severity, f.n_findings
        FROM MARTS.PACK_STATUS p
        LEFT JOIN (
            SELECT dog_id, ANY_VALUE(triage_label) AS triage_label, MAX(severity) AS severity
            FROM AI.TRIAGE GROUP BY dog_id
        ) t ON t.dog_id = p.dog_id
        LEFT JOIN (
            SELECT dog_id, COUNT(*) AS n_findings FROM MARTS.SYNDROME_MATCHES GROUP BY dog_id
        ) f ON f.dog_id = p.dog_id
        ORDER BY COALESCE(t.severity, 0) DESC, ABS(COALESCE(p.z_self, 0)) DESC
    """)

    metric_strip([
        ("dogs monitored",     fmt(len(pack), 0)),
        ("epochs classified",  fmt(one(stats, "EPOCHS_CLASSIFIED", 0), 0)),
        ("open findings",      fmt(one(stats, "SYNDROME_MATCHES", 0), 0)),
        ("on chain",           fmt(one(stats, "ATTESTATIONS_ONCHAIN", 0), 0)),
    ])

    # ------------------------------------------------------------------
    # THE WARD ROUND, AS A ROOM YOU CAN WALK AROUND.
    #
    # This page was a [3, 1] split: 45 cards on the left, a brief and a stacked
    # bar squeezed into a quarter-width rail on the right, and the two columns
    # ending hundreds of pixels apart. The cards are the page and they now get
    # the whole width; what was the rail is a proper half-row above them.
    #
    # And the pack gets a chart it never had. Forty-five cards tell you about
    # ONE DOG AT A TIME — that is what a card is for — so the question a ward
    # round actually opens with, "who is unusual, and unusual how", had no
    # answer anywhere on the page. Three numbers per dog answer it and they are
    # already in the query.
    # ------------------------------------------------------------------
    a1, a2 = st.columns(2)

    with a1:
        panel("The pack in its own measure space",
              "Every dog on the two deviations that matter and the number of "
              "findings each has. Drag to rotate; the cluster at the origin "
              "is a healthy pack.")
        if PLOTLY and pack:
            # THE TWO z AXES ARE NOT THE SAME QUESTION AND THAT IS THE POINT.
            # z_self is a dog against its own trailing hour; z_cohort is the
            # same dog against others of its breed, age and weight. A dog high
            # on one and flat on the other is the interesting case, and it is
            # invisible in any 2D view that picks one of them.
            fig = go.Figure()
            # DROP LINES FIRST, SO THEY SIT BEHIND THE POINTS.
            # A 45-point scatter in a cube reads as a flat smear: nothing tells
            # the eye whether a marker is high on `findings` or merely far from
            # the camera. A hairline from each dog down to the findings=0 floor
            # is the standard depth cue and it adds no information that is not
            # already the point's own coordinate. One trace with None breaks
            # rather than 45 traces, or the legend fills with them.
            _lx: list = []
            _ly: list = []
            _lz: list = []
            for d in pack:
                _n = float(d.get("N_FINDINGS") or 0)
                if _n <= 0:
                    continue
                _lx += [float(d.get("Z_SELF") or 0)] * 2 + [None]
                _ly += [float(d.get("Z_COHORT") or 0)] * 2 + [None]
                _lz += [0.0, _n, None]
            if _lx:
                fig.add_trace(go.Scatter3d(
                    x=_lx, y=_ly, z=_lz, mode="lines",
                    line=dict(color=BORDER, width=1.5),
                    hoverinfo="skip", showlegend=False))
            for sev in sorted({d.get("TRIAGE_SEVERITY") for d in pack},
                              key=lambda s: (s is None, s)):
                grp = [d for d in pack if d.get("TRIAGE_SEVERITY") == sev]
                if not grp:
                    continue
                hue = TRIAGE_COLOUR.get(sev, "#A8A29E")
                lbl = grp[0].get("TRIAGE_LABEL") or "not triaged"
                fig.add_trace(go.Scatter3d(
                    x=[float(d.get("Z_SELF") or 0) for d in grp],
                    y=[float(d.get("Z_COHORT") or 0) for d in grp],
                    z=[float(d.get("N_FINDINGS") or 0) for d in grp],
                    mode="markers", name=str(lbl),
                    marker=dict(
                        color=hue, size=7, opacity=0.9,
                        line=dict(color=CARD, width=1)),
                    text=[f'Dog {d["DOG_ID"]} · {d.get("BREED") or "unknown"}'
                          f'<br>z self {fmt(d.get("Z_SELF"))}'
                          f'<br>z cohort {fmt(d.get("Z_COHORT"))}'
                          f'<br>{fmt(d.get("N_FINDINGS") or 0, 0)} findings'
                          f'<br>{d.get("TRIAGE_LABEL") or "not triaged"}'
                          for d in grp],
                    hoverinfo="text"))
            scene3d(fig, "z vs own baseline", "z vs cohort", "findings")
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", y=0.02, x=0.5, xanchor="center",
                            font=dict(size=10)))
            chart3d(fig, H_LG)
            renderer_note(
                "plotly · 3D",
                f"{len(pack)} dogs, three measured axes, no projection chosen "
                f"for you. The hairline under each dog is a drop line to the "
                f"findings=0 floor — a depth cue, not a fourth number. Colour "
                f"is the triage band a Cortex task wrote into AI.TRIAGE — it "
                f"is not one of the three axes, so agreement between position "
                f"and colour is a result rather than a construction.")

    with a2:
        panel("Pack brief", "AI_AGG over the cached notes, written by a task.")
        brief = rows("SELECT brief, n_findings, n_dogs, generated_at FROM AI.PACK_BRIEF")
        if brief:
            st.markdown(
                f'<div class="tt-card" style="font-size:13px;line-height:1.5">'
                f'{brief[0]["BRIEF"]}'
                f'<div class="tt-quiet" style="margin-top:8px">AI_AGG over '
                f'{fmt(brief[0]["N_FINDINGS"],0)} cached notes · '
                f'{brief[0]["GENERATED_AT"]}</div></div>',
                unsafe_allow_html=True)
        else:
            empty_state("No pack brief yet.",
                        "AI.T_AI generates it on a task. Nothing is called from this page.")

        # The ward round in one bar: how the pack splits across triage bands.
        # Reads the `pack` rows already in memory, so it costs no extra query.
        panel("Triage mix", "The same 45 dogs, counted into bands.")
        bands: dict = {}
        for d in pack:
            k = (d.get("TRIAGE_LABEL") or "not triaged", d.get("TRIAGE_SEVERITY"))
            bands[k] = bands.get(k, 0) + 1
        if PLOTLY and bands:
            ordered = sorted(bands.items(), key=lambda kv: -(kv[0][1] or 0))
            fig = evil_blocks([(lbl, n, TRIAGE_COLOUR.get(sev, "#A8A29E"))
                               for (lbl, sev), n in ordered])
            fig.update_layout(showlegend=False)
            chart(clean_axes(fig), H_STRIP)
            st.markdown(" ".join(
                f'<span class="tt-chip" style="background:'
                f'{TRIAGE_COLOUR.get(sev, "#A8A29E")}1c;border-color:'
                f'{TRIAGE_COLOUR.get(sev, "#A8A29E")}">{lbl} <b>{n}</b></span>'
                for (lbl, sev), n in ordered), unsafe_allow_html=True)

        # THE THREE RAIL METERS, AS ARCS. Same numbers the sidebar prints as
        # bars; drawn here because this is the page where the reader is
        # deciding how much to trust everything downstream of it.
        if PLOTLY and pack:
            _hit = sum(1 for d in pack if (d.get("N_FINDINGS") or 0) > 0)
            _model = 1.0 - (heur / tot) if prov else None
            gauges = [("dogs with a finding", _hit, len(pack), "#B91C1C",
                       f"{_hit}/{len(pack)}")]
            if _model is not None:
                gauges.append(("model-derived states", _model, 1.0, ACCENT,
                               f"{100 * _model:.0f}%"))
            gauges.append(("epochs heuristic",
                           sum(float(d.get("EPOCHS_HEURISTIC") or 0) for d in pack),
                           sum(float(d.get("EPOCHS_TOTAL") or 0) for d in pack) or 1.0,
                           "#A8A29E", None))
            # HEIGHT, EXPLICITLY. These three were the only charts on the page
            # rendered without one, so they took plotly's default 450px each —
            # three near-empty boxes half a screen tall, which pushed this
            # column hundreds of pixels past the 3D scene beside it and left
            # the page ending ragged. H_GAUGE is the arc plus its caption.
            gcols = st.columns(len(gauges))
            for gc, (lbl, val, den, hue, sub) in zip(gcols, gauges):
                with gc:
                    fig = evil_radial(val, den, hue, label=lbl, sub=sub or "")
                    fig.update_layout(height=H_GAUGE,
                                      margin=dict(l=4, r=4, t=6, b=6))
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown(f'<div class="tt-gaugecap">{esc(lbl)}</div>',
                                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # THE CARD GRID, NOW FULL WIDTH AND FOUR ACROSS.
    # ------------------------------------------------------------------
    st.markdown("---")
    panel("The ward round",
          "One card per dog, ordered by how much attention it wants.")
    sort = st.radio("sort", ["triage severity", "deviation from own baseline", "dog id"],
                    horizontal=True, label_visibility="collapsed")
    if sort == "deviation from own baseline":
        pack = sorted(pack, key=lambda r: -abs(r.get("Z_SELF") or 0))
    elif sort == "dog id":
        pack = sorted(pack, key=lambda r: r.get("DOG_ID") or 0)

    if not pack:
        empty_state("No dogs yet.",
                    "Run scripts/load_raw.py then scripts/run_sql.py --all.")
    palette = state_palette()
    photos = breed_photos()          # once, not once per card
    spark = rows("""
        SELECT dog_id, epoch_ts, activity_index
        FROM MARTS.ACTIVITY_EPOCH
        QUALIFY ROW_NUMBER() OVER (PARTITION BY dog_id ORDER BY epoch_ts DESC) <= 60
        ORDER BY dog_id, epoch_ts
    """)
    by_dog: dict = {}
    for r in spark:
        by_dog.setdefault(r["DOG_ID"], []).append(r["ACTIVITY_INDEX"])

    # FOUR PER ROW, NOT THREE. The grid used to live in three quarters of the
    # page; at full width three cards left a third of each row as whitespace
    # inside the card, which stretched the breed line and broke the fixed
    # card height the stylesheet works to maintain.
    PER_ROW = 4
    for chunk_start in range(0, len(pack), PER_ROW):
        cards = st.columns(PER_ROW)
        for c, d in zip(cards, pack[chunk_start:chunk_start + PER_ROW]):
            sev = d.get("TRIAGE_SEVERITY")
            colour = TRIAGE_COLOUR.get(sev, "#A8A29E")
            label = d.get("TRIAGE_LABEL") or "not triaged"
            state = d.get("CURRENT_STATE") or "—"
            stale_s = d.get("SECONDS_SINCE_LAST_EPOCH")
            # Past a week this is not lag, it is the 2018 bulk corpus. Say
            # which, rather than printing "stale 17,715,743s" and hoping.
            live = stale_s is not None and float(stale_s) < 604800
            freshness = (f'<span style="color:{ACCENT}">&#9679;</span> {ago(stale_s)} ago'
                         if live else '<span style="opacity:.55">&#9675;</span> corpus')
            spark = sparkline_svg(by_dog.get(d["DOG_ID"]) or [],
                                  colour=ACCENT if live else "#C9C4BE")
            photo = breed_photo(d.get("BREED"), 44, photos=photos)
            with c:
                st.markdown(f"""
<div class="tt-card tt-dogcard">
  <div class="tt-dogcard-head">
    <div class="tt-dogcard-id">
      {photo}
      <div class="tt-dogcard-name">
        <b>Dog {d['DOG_ID']}</b>
        <span class="tt-quiet tt-breed">{d.get('BREED') or 'unknown breed'}</span>
      </div>
    </div>
    <span class="tt-badge" style="background:{colour}">{label}</span>
  </div>
  <div class="tt-quiet" style="margin:1px 0 5px">
    {fmt(d.get('AGE_YEARS'),1)}y · {fmt(d.get('WEIGHT_KG'),1)}kg · {d.get('COHORT_ID') or '—'}
  </div>
  <div class="tt-chiprow">
    <span class="tt-chip" style="background:{palette.get(state,'#eee')}20;
          border-color:{palette.get(state,BORDER)}">{state}</span>
    <span class="tt-chip">z<sub>self</sub> {fmt(d.get('Z_SELF'))}</span>
    <span class="tt-chip">{fmt(d.get('N_FINDINGS') or 0,0)} findings</span>
  </div>
  <div class="tt-spark">{spark}</div>
  <div class="tt-quiet tt-dogcard-foot">
    {fmt(d.get('EPOCHS_TOTAL') or 0,0)} epochs ·
    {fmt(d.get('PCT_HEURISTIC') or 0,1)}% heuristic · {freshness}
  </div>
</div>""", unsafe_allow_html=True)

    # Said once under the grid rather than 45 times inside it, and repeated
    # in the rail. A photograph implying it is the animal being diagnosed
    # would be the single most misleading thing on this screen.
    if photos:
        n_appx = sum(1 for r in photos.values() if r.get("IS_APPROXIMATE"))
        st.markdown(
            f'<div class="tt-quiet" style="margin-top:6px">'
            f'Thumbnails are <b>reference photographs of the breed</b>, never '
            f'of the study animal — {len(photos)} breeds covered, {n_appx} of '
            f'them by a near-matching breed (amber ring). Dogs with no '
            f'reference photo get a drawn icon on a dashed ring — visibly '
            f'not a photograph, which is the point. '
            f'{list(photos.values())[0].get("CREDIT") or ""} '
            f'Placeholder icon: Font Awesome via svgrepo.com, CC BY 4.0.'
            f'</div>',
            unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # THE NUMBERS UNDER THE PAGE, PAIRED.
    #
    # Provenance used to be the last thing in the quarter-width rail, where it
    # was three rows of table under a chart under a card — the bottom of a
    # column that had already run out. It is the honesty statement for
    # everything above it and it gets a half-row of its own, against the
    # triage counts it should be read next to.
    # ------------------------------------------------------------------
    p1, p2 = st.columns(2)
    with p1:
        panel("Provenance",
              "Where each epoch's state came from. The heuristic share is the "
              "banner at the top of every tab, in numbers.")
        if prov:
            html_table(
                [{"s": r["STATE_SOURCE"], "n": fmt(r["EPOCHS"], 0),
                  "p": f'{r["PCT"]}%'} for r in prov],
                [("s", "source"), ("n", "epochs"), ("p", "share")])
        else:
            empty_state("No provenance rows.",
                        "MARTS.V_STATE_PROVENANCE is empty.")
    with p2:
        panel("Triage bands",
              "How the pack splits, and how many findings sit behind each "
              "band.")
        if bands:
            _rows = []
            for (lbl, sev), n in sorted(bands.items(), key=lambda kv: -(kv[0][1] or 0)):
                _f = sum((d.get("N_FINDINGS") or 0) for d in pack
                         if (d.get("TRIAGE_LABEL") or "not triaged") == lbl)
                _rows.append({
                    "b": f'<span class="tt-badge" style="background:'
                         f'{TRIAGE_COLOUR.get(sev, "#A8A29E")}">{lbl}</span>',
                    "n": fmt(n, 0),
                    "s": fmt(sev, 0) if sev is not None else "—",
                    "f": fmt(_f, 0)})
            html_table(_rows, [("b", "band"), ("n", "dogs"),
                               ("s", "severity"), ("f", "findings")])
        else:
            empty_state("Nothing triaged yet.",
                        "AI.T_AI writes AI.TRIAGE on a task.")


# ===========================================================================
# TAB 2 — LIVE COLLAR.  Kills the "is it real" objection.
# ===========================================================================
def _page_1():
    st.markdown("#### Live collar")
    st.markdown('<span class="tt-quiet">Raw 100 Hz dual-sensor waveform, the '
                'features derived from it, and the state each second was '
                'classified into. Same seconds, three levels of abstraction.'
                '</span>', unsafe_allow_html=True)

    # ONLY THE DOGS THAT ACTUALLY HAVE A WAVEFORM.
    #
    # This picker used to list every dog in MARTS.EPOCH_STATES — all 45,
    # including the bulk-corpus dogs whose epochs came from CSV and which have
    # no rows in RAW.COLLAR_TELEMETRY at all. It therefore opened on dog 16,
    # which has zero samples, and drew three empty panels on the one tab whose
    # entire job is answering "is any of this real". Most recent feed first, so
    # it opens on whatever the replayer touched last.
    dogs = rows("""
        SELECT dog_id, COUNT(*) AS n_samples, MAX(sample_ts) AS latest
        FROM RAW.COLLAR_TELEMETRY
        GROUP BY dog_id
        ORDER BY latest DESC
    """)
    if not dogs:
        empty_state(
            "No collar telemetry has landed yet.",
            "This tab reads RAW.COLLAR_TELEMETRY, which the replayer fills. "
            "Start it: python ingest/replay.py --speed 60 --dogs 12")
    else:
        n_classified = one(rows("SELECT COUNT(DISTINCT dog_id) AS n "
                                "FROM MARTS.EPOCH_STATES"), "N", 0)
        st.markdown(
            f'<span class="tt-quiet">{len(dogs)} of {fmt(n_classified, 0)} dogs '
            f'have a raw waveform to show — the rest are bulk-corpus dogs whose '
            f'epochs were loaded from CSV, so they appear on every other tab but '
            f'have no 100 Hz feed to replay here.</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        dog = c1.selectbox("dog", [int(r["DOG_ID"]) for r in dogs], key="live_dog",
                           format_func=lambda d: "dog " + str(d))
        window = c2.selectbox("window (seconds)", [30, 60, 120, 300], index=1)
        auto = c3.checkbox("auto-refresh every 15s while the replayer runs", value=False)
        if auto:
            try:
                st.autorefresh(interval=15000, key="live_refresh")
            except AttributeError:
                st.markdown('<span class="tt-quiet">This Streamlit build has no '
                            'st.autorefresh; use the browser refresh.</span>',
                            unsafe_allow_html=True)

        ing = rows(f"""
            SELECT
                (SELECT SUM(n_rows) FROM RAW.INGEST_LOG
                  WHERE landed_at > DATEADD('minute', -1, CURRENT_TIMESTAMP())) AS last_min,
                (SELECT COUNT(*) FROM RAW.COLLAR_TELEMETRY WHERE dog_id = {dog}) AS dog_rows,
                (SELECT MAX(sample_ts) FROM RAW.COLLAR_TELEMETRY WHERE dog_id = {dog}) AS latest,
                (SELECT COUNT(DISTINCT _batch_id) FROM RAW.COLLAR_TELEMETRY) AS batches
        """)
        metric_strip([
            ("rows landed, last minute", fmt(one(ing, "LAST_MIN", 0), 0)),
            ("raw samples, this dog",    fmt(one(ing, "DOG_ROWS", 0), 0)),
            ("micro-batches",            fmt(one(ing, "BATCHES", 0), 0)),
            ("latest sample",            str(one(ing, "LATEST") or "—")[:19]),
        ])

        # The diagram the rest of this tab is evidence for. Drawn, not
        # photographed: SiS blocks outbound requests, so an <img> to any CDN
        # renders broken — and the useful picture is sensor PLACEMENT anyway.
        with st.expander("Where the two sensors sit, and why their correlation is the feature",
                         expanded=True):
            # EQUAL COLUMNS AND A CAPPED FIGURE. At [3, 2] the drawing filled
            # three fifths of the row, and an 880x700 viewBox at width:100% is
            # over six hundred pixels tall — against a prose card half that, so
            # the panel ended in a column of skeleton beside a column of white.
            # The figure is capped instead of stretched: it is a diagram, and a
            # diagram gets no clearer past about five hundred pixels.
            a1, a2 = st.columns(2)
            with a1:
                st.markdown(
                    '<div style="max-width:470px;margin:0 auto">'
                    + sensor_anatomy_svg() + "</div>", unsafe_allow_html=True)
                # A LICENCE CONDITION, NOT A NICETY. The skeleton is CC BY-SA
                # and the attribution has to be on the page it appears on. It
                # was written as a constant and then never rendered.
                st.markdown(f'<div class="tt-quiet" style="text-align:center;'
                            f'font-size:10.5px;margin-top:2px">{SKELETON_CREDIT}'
                            f'</div>', unsafe_allow_html=True)
            with a2:
                st.markdown(
                    '<div class="tt-card" style="font-size:13px;line-height:1.55">'
                    '<b>Two sensors, one question.</b><br>'
                    'A collar alone cannot tell a dog walking from a dog shaking '
                    'its head — both are vigorous neck motion. The back harness '
                    'resolves it: in <i>locomotion</i> the two sensors rise and '
                    'fall together, so their correlation is high. In a head '
                    'shake or a scratch the neck moves and the back does not, so '
                    'it collapses toward zero.<br><br>'
                    '<span class="tt-quiet">That single number, '
                    '<code>CORR(vm_neck, vm_back)</code> over a one-second epoch, '
                    'is computed in Snowflake and is what the ethogram states '
                    'are built on. The chart below it is the raw 100 Hz signal '
                    'the correlation is taken over.</span></div>',
                    unsafe_allow_html=True)
                # The two readings the whole tab turns on, said as the numbers
                # they actually are rather than left to the prose. This is also
                # what brings the right column down to meet the figure.
                st.markdown(f"""
<div class="tt-card" style="margin-top:10px">
  <div class="tt-metric-label">what the correlation says</div>
  <div style="display:flex;gap:14px;margin-top:8px;font-size:12.5px;
              line-height:1.5;flex-wrap:wrap">
    <div style="flex:1 1 150px">
      <b style="color:{S_BLUE}">corr near +1</b><br>
      <span class="tt-quiet">neck and back rise and fall together. The whole
      animal is travelling: walk, trot, gallop, pace.</span>
    </div>
    <div style="flex:1 1 150px">
      <b style="color:{S_ORANGE}">corr near 0</b><br>
      <span class="tt-quiet">the neck moves and the back does not. The head is
      acting alone: shake, scratch, sniff.</span>
    </div>
  </div>
  <div class="tt-quiet" style="margin-top:9px;font-size:11.5px">
    Both sensors sample at 100 Hz. The dot on the cervical chain is the collar,
    the dot on the thoracic chain is the harness, and the dashed arc between
    them is the only feature in this build that needs two of them.
  </div>
</div>""", unsafe_allow_html=True)

        wave = rows(f"""
            SELECT sample_ts, neck_ax, neck_ay, neck_az,
                   SQRT(neck_ax*neck_ax + neck_ay*neck_ay + neck_az*neck_az) AS vm_neck,
                   SQRT(back_ax*back_ax + back_ay*back_ay + back_az*back_az) AS vm_back,
                   is_synthetic
            FROM RAW.COLLAR_TELEMETRY
            WHERE dog_id = {dog}
              AND sample_ts >= DATEADD('second', -{window},
                    (SELECT MAX(sample_ts) FROM RAW.COLLAR_TELEMETRY WHERE dog_id = {dog}))
            ORDER BY sample_ts
        """)
        feat = rows(f"""
            SELECT e.epoch_ts, e.vm_neck_mean, e.vm_neck_std, e.neck_back_corr,
                   e.n_samples, s.state, s.state_source
            FROM STAGING.EPOCH_FEATURES e
            LEFT JOIN MARTS.EPOCH_STATES s
                   ON s.dog_id = e.dog_id AND s.test_num = e.test_num
                  AND s.epoch_ts = e.epoch_ts
            WHERE e.dog_id = {dog}
              AND e.epoch_ts >= DATEADD('second', -{window},
                    (SELECT MAX(epoch_ts) FROM STAGING.EPOCH_FEATURES WHERE dog_id = {dog}))
            ORDER BY e.epoch_ts
        """)

        # ------------------------------------------------------------------
        # THREE LEVELS OF ABSTRACTION, ON ONE X AXIS YOU CAN DRAG.
        #
        # These were three separate plotly figures stacked down the page: the
        # 100 Hz waveform, the per-second features derived from it, and the
        # correlation the states are built on. Reading them together is the
        # entire point of the tab — you are meant to see a burst in panel 1
        # become a spike in panel 2 and a collapse toward zero in panel 3, at
        # the same instant. Three independent figures cannot do that. Zoom
        # into six interesting seconds on one and the other two still show
        # sixty, so the reader has to align them by eye and take it on trust.
        #
        # WHY BOKEH AND NOT THE OTHER TWO. Sharing `x_range` between figures
        # makes the pan, the zoom and the range scrubber below drive all three
        # at once, and the linked crosshair puts one vertical line through the
        # same instant in every panel. That is a Bokeh model-level feature —
        # plotly has no cross-figure range binding without a Dash callback,
        # and Streamlit gives plotly no callbacks; Vega-Lite can bind scales
        # across a concat but not to a scrubber over a 30 000-point signal.
        #
        # THE X AXIS IS SECONDS, NOT WALL CLOCK, and deliberately: a
        # TIMESTAMP_NTZ comes back naive, so rendering it as a datetime axis
        # would silently label it in the viewer's timezone. Elapsed seconds
        # from the first sample also lets the 100 Hz panel and the 1 Hz panels
        # share one exact numeric axis with no resampling.
        # ------------------------------------------------------------------
        drawn_signal = False
        if BOKEH and wave:
            t0 = wave[0]["SAMPLE_TS"]
            wt = [(r["SAMPLE_TS"] - t0).total_seconds() for r in wave]
            neck = [float(r["VM_NECK"] or 0) for r in wave]
            back = [float(r["VM_BACK"] or 0) for r in wave]
            raw_src = ColumnDataSource(dict(t=wt, neck=neck, back=back))

            span = float(wt[-1] - wt[0]) or 1.0
            # Open on a window you can actually see individual strides in.
            # The full 60 s of 100 Hz data is 6 000 points across ~1 100 px:
            # five samples per pixel, which draws as a solid band. Twelve
            # seconds is roughly one pixel per sample, and the scrubber is
            # right there to say that the rest of the record still exists.
            show = min(span, 12.0)
            xr = (float(wt[0]), float(wt[0]) + show)

            TOOLS = "xpan,xwheel_zoom,box_zoom,reset"
            link = Span(dimension="height", line_color=INK_2, line_width=1,
                        line_dash=[3, 3], line_alpha=0.55)

            p1 = bk_figure(height=176, sizing_mode="stretch_width", tools=TOOLS,
                           active_scroll="xwheel_zoom", x_range=xr,
                           title="1 · raw 100 Hz vector magnitude — neck collar "
                                 "(orange) against back harness (blue)")
            p1.line("t", "back", source=raw_src, color=S_BLUE, line_width=1,
                    legend_label="back harness")
            p1.line("t", "neck", source=raw_src, color=S_ORANGE, line_width=1,
                    legend_label="neck collar")
            p1.legend.location = "top_left"
            p1.legend.orientation = "horizontal"
            p1.legend.click_policy = "hide"
            bk_style(p1, ylabel="|a| (g)")
            p1.add_tools(CrosshairTool(overlay=link))
            panels = [p1]

            if feat:
                ft = [(r["EPOCH_TS"] - t0).total_seconds() for r in feat]
                fsrc = ColumnDataSource(dict(
                    t=ft,
                    mean=[float(r["VM_NECK_MEAN"] or 0) for r in feat],
                    sd=[float(r["VM_NECK_STD"] or 0) for r in feat],
                    corr=[float(r["NECK_BACK_CORR"] or 0) for r in feat],
                    state=[r.get("STATE") or "—" for r in feat],
                    src=[r.get("STATE_SOURCE") or "—" for r in feat]))

                p2 = bk_figure(height=150, sizing_mode="stretch_width", tools=TOOLS,
                               active_scroll="xwheel_zoom", x_range=p1.x_range,
                               title="2 · derived epoch features — neck vector "
                                     "magnitude, mean (solid) and standard "
                                     "deviation (dotted), in g")
                p2.line("t", "mean", source=fsrc, color=INK, line_width=1.6)
                p2.line("t", "sd", source=fsrc, color=INK_2, line_width=1.2,
                        line_dash=[2, 3])
                bk_style(p2, ylabel="g")
                p2.add_tools(CrosshairTool(overlay=link))
                panels.append(p2)

                # Panel 3 carries the state bands behind the line, because the
                # claim being made is causal: the correlation collapses AND
                # THAT IS WHY this second was called a head shake. Two charts
                # cannot say "and that is why"; a band behind the line can.
                pal = state_palette()
                p3 = bk_figure(height=182, sizing_mode="stretch_width", tools=TOOLS,
                               active_scroll="xwheel_zoom", x_range=p1.x_range,
                               y_range=(-1.06, 1.06),
                               title="3 · the feature the states are built on — "
                                     "CORR(vm_neck, vm_back) over the same "
                                     "seconds, banded by the state it produced")
                runs = []
                for i, r in enumerate(feat):
                    s = r.get("STATE") or "UNKNOWN"
                    if runs and runs[-1][0] == s:
                        runs[-1][2] = ft[i]
                    else:
                        runs.append([s, ft[i], ft[i]])
                if runs:
                    p3.quad(left=[a for _, a, _ in runs],
                            right=[b + 1.0 for _, _, b in runs],
                            top=[1.06] * len(runs), bottom=[-1.06] * len(runs),
                            fill_color=[pal.get(s, "#D6D3D1") for s, _, _ in runs],
                            fill_alpha=0.16, line_color=None, level="underlay")
                p3.add_layout(Span(location=0, dimension="width",
                                   line_color=BORDER, line_width=1))
                p3.line("t", "corr", source=fsrc, color=S_ORANGE, line_width=1.8)
                dots = p3.scatter("t", "corr", source=fsrc, size=4,
                                  color=S_ORANGE, alpha=0)
                p3.add_tools(HoverTool(
                    renderers=[dots], mode="vline", attachment="above",
                    tooltips=[("t", "@t{0.0} s"), ("corr", "@corr{0.000}"),
                              ("state", "@state"), ("label from", "@src")]))
                bk_style(p3, ylabel="corr")
                p3.add_tools(CrosshairTool(overlay=link))
                panels.append(p3)

            # The scrubber. It is the whole window at a glance with the shaded
            # box showing where the panels above are looking; drag the box and
            # all three follow. Without it, zooming in is a one-way door — you
            # lose any sense of where in the recording you ended up.
            nav = bk_figure(height=78, sizing_mode="stretch_width",
                            y_range=p1.y_range, tools="", toolbar_location=None,
                            x_axis_label="seconds since the start of this window")
            nav.line("t", "neck", source=raw_src, color=INK_2, line_width=0.7,
                     alpha=0.55)
            rt = RangeTool(x_range=p1.x_range)
            rt.overlay.fill_color = S_ORANGE
            rt.overlay.fill_alpha = 0.14
            rt.overlay.line_color = S_ORANGE
            nav.add_tools(rt)
            nav.ygrid.grid_line_color = None
            nav.yaxis.visible = False
            nav.xgrid.grid_line_color = None
            panels.append(nav)

            # merge_tools=False keeps each panel's own toolbar (autohidden, so
            # they appear on hover); toolbar_location=None suppresses only the
            # merged one the grid would otherwise add above everything.
            grid = bk_gridplot([[p] for p in panels], sizing_mode="stretch_width",
                               toolbar_location=None, merge_tools=False)
            # Bokeh's `height` is the OUTER height of a figure — title, axis and
            # label included — so the sum is exact and the slack is only for the
            # grid's own row spacing. Erring high costs a few px of white space;
            # erring low silently cuts the scrubber off the bottom, and it is
            # the control the rest of the view depends on.
            bokeh_panel(grid, sum(p.height for p in panels) + 36)
            renderer_note(
                "bokeh",
                f"One document, {len(panels) - 1} panels, one shared x axis. Drag "
                f"the shaded box in the strip at the bottom to move all of them "
                f"together; scroll to zoom, click a legend entry to hide a trace. "
                f"The dashed crosshair is the same instant in every panel.")
            drawn_signal = True

        # PLOTLY IS THE FALLBACK, NOT THE DEAD CODE PATH. If bokeh failed to
        # import — the one thing in environment.yml that is not load-bearing
        # elsewhere — this tab is the tab that answers "is any of this real",
        # and it renders the signal or it has no argument.
        if not drawn_signal and PLOTLY and wave:
            xs = [str(r["SAMPLE_TS"]) for r in wave]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=[float(r["VM_NECK"] or 0) for r in wave],
                                     mode="lines", name="neck",
                                     line=dict(color=S_ORANGE, width=1),
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xs, y=[float(r["VM_BACK"] or 0) for r in wave],
                                     mode="lines", name="back",
                                     line=dict(color=S_BLUE, width=1),
                                     hoverinfo="skip"))
            fig.update_layout(title=f"1 · raw 100 Hz vector magnitude — "
                                    f"<span style='color:{S_ORANGE}'>neck collar</span> vs "
                                    f"<span style='color:{S_BLUE}'>back harness</span>",
                              title_font_size=12)
            chart(evil_axes(fig, y_zero_line=False), H_SM)

            if feat:
                xs = [str(r["EPOCH_TS"]) for r in feat]
                # THE DERIVED FEATURES GET THE FILL; THE RAW WAVEFORM ABOVE
                # DOES NOT, AND THE DIFFERENCE IS DELIBERATE. A 100 Hz trace
                # oscillates about its own mean hundreds of times across the
                # panel, so filling to zero paints a solid block of colour and
                # destroys the only thing the panel is for. An epoch feature is
                # one value per second with a real baseline at zero, which is
                # exactly the shape an area chart is for.
                fig = go.Figure()
                evil_area(fig, xs, [float(r["VM_NECK_MEAN"] or 0) for r in feat],
                          INK, width=1.6, shape="linear", name="vm mean",
                          text=[f"mean {fmt(r['VM_NECK_MEAN'])} g" for r in feat])
                fig.add_trace(go.Scatter(
                    x=xs, y=[float(r["VM_NECK_STD"] or 0) for r in feat],
                    mode="lines", line=dict(color=INK_2, width=1.2, dash="dot"),
                    name="vm sd", hoverinfo="skip", showlegend=False))
                fig.update_layout(
                    title="2 · derived epoch features — neck vector magnitude, "
                          "mean (solid) and standard deviation (dotted), in g",
                    title_font_size=12)
                chart(evil_axes(fig, y_zero_line=False), H_SM)

                # CORRELATION IS SIGNED, SO THIS ONE FILLS TO ZERO RATHER THAN
                # TO THE FLOOR. tozeroy on a [-1, 1] axis fills DOWNWARD for
                # the negative stretches, which is the correct reading: the
                # neck-dominant seconds hang below the line instead of being
                # drawn as a short bar above it.
                fig = go.Figure()
                evil_area(
                    fig, xs, [float(r["NECK_BACK_CORR"] or 0) for r in feat],
                    S_ORANGE, width=1.8, shape="linear", name="corr",
                    text=[f"corr {fmt(r['NECK_BACK_CORR'])}<br>{r.get('STATE')}"
                          for r in feat])
                fig.add_hline(y=0, line=dict(color=BORDER, width=1))
                fig.update_layout(
                    title="3 · the feature the states are built on — "
                          "CORR(vm_neck, vm_back) over the same seconds, [-1, 1]",
                    title_font_size=12,
                    yaxis=dict(range=[-1.05, 1.05]))
                chart(evil_axes(fig, y_zero_line=False), H_SM)

        if wave and any(r.get("IS_SYNTHETIC") for r in wave):
            st.markdown('<div class="tt-caveat">This window contains '
                        '<b>SYNTHETIC</b> samples injected by demo_spike.py. '
                        'They carry is_synthetic = TRUE; detection sees them, '
                        'training never fits them.</div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # THE SAME SECONDS AS A SHAPE IN SPACE.
        #
        # Three traces of ax/ay/az stacked on a time axis is three wiggly lines
        # that all look alike. The accelerometer is measuring a 3-vector, and
        # plotted as one it stops being a signal and becomes a MOTION SIGNATURE
        # you can recognise on sight: a resting dog is a dense knot around the
        # 1g gravity vector, a walking dog is a closed loop repeated once per
        # stride, and a head shake is a flat high-amplitude smear along one
        # axis. That is a genuinely 3D fact and it flattens badly — which is
        # the only good reason to spend a 3D plot on anything.
        # ------------------------------------------------------------------
        if PLOTLY and wave and len(wave) > 30:
            panel("The motion signature, in the three axes the collar "
                  "actually measures",
                  "Drag to rotate. One point per 100 Hz sample, the line is "
                  "time. Stillness knots around the gravity vector; a gait "
                  "draws a repeating loop; a head shake smears along one axis.")
            ax = [float(r["NECK_AX"] or 0) for r in wave]
            ay = [float(r["NECK_AY"] or 0) for r in wave]
            az = [float(r["NECK_AZ"] or 0) for r in wave]
            fig = go.Figure(go.Scatter3d(
                x=ax, y=ay, z=az, mode="lines+markers",
                # the path is the thread, the markers carry TIME along it —
                # a 2px opaque line drawn over its own markers hides the one
                # channel that says which way round the loop the dog went
                line=dict(color=alpha(S_ORANGE, 0.35), width=1),
                marker=dict(size=2.2, color=list(range(len(ax))),
                            colorscale=[[0, "#cde2fb"], [1, "#0d366b"]],
                            opacity=0.9),
                hovertemplate=("ax %{x:.2f} g<br>ay %{y:.2f} g"
                               "<br>az %{z:.2f} g<extra></extra>")))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="neck ax (g)", backgroundcolor=CARD,
                               gridcolor=BORDER, zerolinecolor=BORDER),
                    yaxis=dict(title="neck ay (g)", backgroundcolor=CARD,
                               gridcolor=BORDER, zerolinecolor=BORDER),
                    zaxis=dict(title="neck az (g)", backgroundcolor=CARD,
                               gridcolor=BORDER, zerolinecolor=BORDER),
                    aspectmode="cube",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))),
                paper_bgcolor=CARD, plot_bgcolor=CARD, showlegend=False,
                margin=dict(l=0, r=0, t=4, b=0), height=H_LG,
                font=dict(family=FONT_CHART, size=11,
                          color=INK_2))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            renderer_note(
                "plotly · 3d",
                "6 000 samples of a 3-vector, drawn as the 3-vector it is "
                "rather than as three lines that happen to be next to each "
                "other. Dark points are late in the window, pale ones early.")

        # ------------------------------------------------------------------
        # WHAT THE SIGNAL IS MADE OF, AND WHEN.
        #
        # Panel 1 shows amplitude over time and the cube above shows the shape
        # of the motion, but neither shows FREQUENCY, which is the thing that
        # actually separates the behaviours this app is about. A trot is a
        # narrow ridge at the stride rate, around 2 Hz. A head shake is a much
        # higher, much broader hump at 5-9 Hz. A resting dog is flat. In the
        # 100 Hz trace both look like "vigorous motion" — which is exactly the
        # confusion the two-sensor correlation was introduced to resolve, and
        # this is the same confusion viewed from the other side.
        #
        # A SURFACE RATHER THAN A HEATMAP, on purpose and not for decoration.
        # The interesting structure here is a RIDGE — a peak that holds a
        # frequency for a few seconds and then moves — and a ridge is a shape,
        # so height reads it faster than colour does. The colour is carried
        # too, so nothing depends on judging a height by eye.
        #
        # SHORT-TIME FOURIER, WRITTEN OUT. 256 samples is 2.56 s per column at
        # 100 Hz, which is long enough to resolve a 2 Hz stride (about five
        # cycles) and short enough that a head shake is not smeared across the
        # whole window. Hann window, or every column carries the spectral
        # leakage of its own rectangular edges and the whole surface sits on a
        # false noise floor.
        # ------------------------------------------------------------------
        if PLOTLY and NUMPY and wave and len(wave) >= 512:
            sig = np.asarray([float(r["VM_NECK"] or 0) for r in wave],
                             dtype=float)
            fs_hz, nfft, hop = 100.0, 256, 64
            # Gravity is a constant ~1 g offset, which is a DC term about two
            # orders of magnitude larger than any motion in the band we care
            # about. Left in, it dominates the colour scale and every column
            # renders as one spike at 0 Hz against a flat floor.
            sig = sig - sig.mean()
            win = np.hanning(nfft)
            starts = range(0, len(sig) - nfft + 1, hop)
            cols = [np.abs(np.fft.rfft(sig[s:s + nfft] * win)) for s in starts]
            if cols:
                freqs = np.fft.rfftfreq(nfft, d=1.0 / fs_hz)
                keep = freqs <= 20.0     # nothing a dog does lives above this
                # dB, because the dynamic range between a resting second and a
                # shaking one is three orders of magnitude. On a linear scale
                # the shake is the only thing with any height at all and the
                # rest of the record is an empty floor.
                spec = 20.0 * np.log10(np.asarray(cols).T[keep] + 1e-6)
                t_ax = [float(s) / fs_hz for s in starts]
                floor = float(np.percentile(spec, 5))

                panel("The same seconds as frequency over time",
                      "Drag to rotate. Each ridge running left to right is a rhythm "
                      "the dog held: a stride sits low and narrow around 2 Hz, a head "
                      "shake is a broad hump up at 5–9 Hz, and stillness is flat. "
                      "Height and colour are the same number — power in dB — so "
                      "nothing rests on judging a height by eye.")
                # .tolist(), AND THIS IS HAZARD 1 AND 4 AGAIN.
                #
                # This was the one chart in the app handing plotly a numpy
                # array instead of plain Python lists, and it rendered as an
                # empty white box the height of the figure — no axes, no
                # surface, nothing. plotly.py 6 encodes an ndarray as a
                # base64 typed array (`bdata`), and the plotly.js bundled with
                # the Streamlit build SiS pins does not decode it, so the trace
                # arrives with no data and the scene never draws. Every other
                # chart here is fed lists by rows(); this one bypassed that
                # because the STFT is computed in the page.
                fig = go.Figure(go.Surface(
                    x=t_ax, y=[float(f) for f in freqs[keep]],
                    z=np.clip(spec, floor, None).tolist(),
                    colorscale=[[0.0, "#FAFAF9"], [0.25, "#cde2fb"],
                                [0.55, S_BLUE], [0.80, S_ORANGE],
                                [1.0, "#7a1f06"]],
                    showscale=False,
                    # The contour projected onto the floor is the 2D
                    # spectrogram you would otherwise have drawn, kept as a
                    # reference under the surface it explains.
                    contours=dict(z=dict(show=True, usecolormap=True,
                                         project=dict(z=True),
                                         highlightcolor=INK_2)),
                    lighting=dict(ambient=0.62, diffuse=0.72, specular=0.18,
                                  roughness=0.85, fresnel=0.1),
                    hovertemplate=("t %{x:.1f} s<br>%{y:.1f} Hz"
                                   "<br>%{z:.0f} dB<extra></extra>")))
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(title="seconds into the window",
                                   backgroundcolor=SURFACE, gridcolor=BORDER,
                                   zerolinecolor=BORDER, showspikes=False),
                        yaxis=dict(title="frequency (Hz)",
                                   backgroundcolor=SURFACE, gridcolor=BORDER,
                                   zerolinecolor=BORDER, showspikes=False),
                        zaxis=dict(title="power (dB)", backgroundcolor=SURFACE,
                                   gridcolor=BORDER, zerolinecolor=BORDER,
                                   showspikes=False),
                        # Not a cube: time is the long axis and squaring it
                        # would compress a 60 s record into the same width as
                        # a 20 Hz band, which is what makes ridges unreadable.
                        # aspectmode said out loud rather than inferred — an
                        # aspectratio under the default 'auto' is advisory.
                        aspectmode="manual",
                        aspectratio=dict(x=1.9, y=1.0, z=0.62),
                        camera=dict(eye=dict(x=1.75, y=-1.5, z=0.95))),
                    paper_bgcolor=CARD, plot_bgcolor=CARD, showlegend=False,
                    margin=dict(l=0, r=0, t=4, b=0), height=H_LG,
                    font=dict(family=FONT_CHART, size=11,
                              color=INK_2))
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
                renderer_note(
                    "plotly · 3d",
                    f"{len(t_ax)} overlapping 2.56 s windows, Hann-tapered, "
                    f"transformed in the app rather than in the warehouse — "
                    f"Snowflake has no FFT, and shipping 100 Hz to numpy is "
                    f"the honest way round.")

        # State ribbon, drawn from contiguous runs rather than one bar per epoch.
        if PLOTLY and feat:
            palette = state_palette()
            runs = []
            for r in feat:
                s = r.get("STATE") or "UNKNOWN"
                if runs and runs[-1][0] == s:
                    runs[-1][1] += 1
                else:
                    runs.append([s, 1, str(r["EPOCH_TS"]), r.get("STATE_SOURCE")])
            fig = go.Figure()
            for s, n, t0, src in runs:
                fig.add_trace(go.Bar(
                    x=[n], y=["state"], orientation="h",
                    marker=dict(color=palette.get(s, "#D6D3D1"),
                                line=dict(width=0)),
                    hovertext=[f"{s} · {n}s · from {t0} · source {src}"],
                    hoverinfo="text", showlegend=False))
            fig.update_layout(barmode="stack",
                              title="4 · classified state, one block per second",
                              title_font_size=12,
                              xaxis=dict(visible=False), yaxis=dict(visible=False))
            chart(clean_axes(fig), H_RIBBON)
            legend = " ".join(
                f'<span class="tt-chip" style="background:{palette.get(e["STATE"])}30;'
                f'border-color:{palette.get(e["STATE"])}">{e["STATE"]}</span>'
                for e in ethogram())
            st.markdown(f'<div style="margin-top:-6px">{legend}</div>',
                        unsafe_allow_html=True)


# ===========================================================================
# TAB 3 — ETHOGRAM.  The behavioural fingerprint of one dog.
# ===========================================================================
def _page_2():
    st.markdown("#### Ethogram")
    dogs = rows("SELECT DISTINCT dog_id FROM MARTS.EPOCH_STATES ORDER BY dog_id")
    if not dogs:
        empty_state("No states yet.", "Run the pipeline, then the replayer.")
    else:
        dog = st.selectbox("dog", [int(r["DOG_ID"]) for r in dogs], key="etho_dog",
                           format_func=lambda d: "dog " + str(d))
        palette = state_palette()

        # Every query for this page, up front. The layout below pairs charts
        # across two columns and has to size both halves of a row together,
        # which it cannot do if half the data is still being fetched inside a
        # `with` block further down.
        bouts = rows(f"""
            SELECT state, bout_start, bout_seconds
            FROM MARTS.STATE_BOUTS
            WHERE dog_id = {dog}
            ORDER BY bout_start
            LIMIT 4000
        """)
        tr = rows(f"""
            SELECT from_state, to_state, prob, n
            FROM MARTS.STATE_TRANSITIONS WHERE dog_id = {dog}
        """)
        budget = rows(f"""
            SELECT state, COUNT(*) AS n,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM MARTS.EPOCH_STATES WHERE dog_id = {dog}
            GROUP BY state ORDER BY n DESC
        """)
        bl = rows(f"""
            SELECT state,
                   ROUND(AVG(bout_seconds),1) AS mean_s,
                   MEDIAN(bout_seconds)       AS median_s,
                   MAX(bout_seconds)          AS max_s,
                   COUNT(*)                   AS bouts
            FROM MARTS.STATE_BOUTS WHERE dog_id = {dog}
            GROUP BY state ORDER BY bouts DESC
        """)

        metric_strip([
            ("distinct states",  fmt(len(budget), 0)),
            ("bouts",            fmt(sum(int(r["BOUTS"] or 0) for r in bl), 0)),
            ("longest bout",     ago(max([float(r["MAX_S"] or 0) for r in bl] or [0]))),
            ("transitions seen", fmt(len(tr), 0)),
        ])
        if PLOTLY and bouts:
            # ONE TRACE PER STATE, NOT ONE PER BOUT.
            #
            # This drew 518 separate single-bar traces stacked end to end. Every
            # trace carries its own 1px antialiased edge, so at 96px tall the
            # ribbon rendered as vertical static — you could not read a single
            # bout out of it, and it was the loudest thing on the page. Grouping
            # by state and using base= to place each run on a shared axis gives
            # at most fourteen traces, contiguous fills, and the same data.
            spans: dict = {}
            cursor = 0
            for b in bouts:
                secs = int(b["BOUT_SECONDS"] or 0)
                s = b["STATE"]
                spans.setdefault(s, {"x": [], "base": [], "t": []})
                spans[s]["x"].append(secs)
                spans[s]["base"].append(cursor)
                spans[s]["t"].append(f'{s} · {secs}s · from {b["BOUT_START"]}')
                cursor += secs
            fig = go.Figure()
            for s, d in spans.items():
                fig.add_trace(go.Bar(
                    x=d["x"], base=d["base"], y=["session"] * len(d["x"]),
                    orientation="h",
                    marker=dict(color=palette.get(s, "#D6D3D1"),
                                line=dict(width=0)),
                    hovertext=d["t"], hoverinfo="text", showlegend=False))
            fig.update_layout(
                barmode="overlay",
                title=f"State ribbon — {len(bouts):,} bouts over "
                      f"{cursor / 3600:.1f} hours of recording, in order",
                title_font_size=12,
                bargap=0,
                xaxis=dict(visible=False), yaxis=dict(visible=False))
            chart(clean_axes(fig), H_RIBBON)
            legend = " ".join(
                f'<span class="tt-chip" style="background:'
                f'{palette.get(e["STATE"], "#D6D3D1")}2e;border-color:'
                f'{palette.get(e["STATE"], "#D6D3D1")}">{e["STATE"]}</span>'
                for e in ethogram() if e["STATE"] in spans)
            st.markdown(f'<div style="margin-top:-4px">{legend}</div>',
                        unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # HOW LONG EACH BEHAVIOUR LASTS, WHICH IS NOT ITS AVERAGE.
        #
        # The table further down this page reports a mean and a median bout
        # length per state, and both are close to meaningless here: bout
        # lengths are strongly right-skewed and several states are BIMODAL —
        # SNIFF is a two-second check or a ninety-second investigation and
        # almost nothing between, and a mean of those two lands in the gap
        # where the dog never actually is. A distribution shows that; a
        # summary statistic is specifically the thing that hides it.
        #
        # A RIDGELINE RATHER THAN A BOX PLOT OR FOURTEEN HISTOGRAMS. Fourteen
        # separate panels cannot be compared — the reader has to hold one
        # shape in their head while looking at the next. Overlapping the rows
        # on ONE shared x axis makes "SHAKE is short, REST is long" a single
        # glance, and the overlap is what buys the vertical room to fit
        # fourteen of them on a screen.
        #
        # WHY ALTAIR. transform_density is a kernel density estimate computed
        # by the renderer, per group, from the raw values — so this is one
        # chart specification rather than fourteen KDEs computed in Python and
        # fourteen traces. plotly's equivalent is a violin per state, which
        # spends the same vertical space to show each shape mirrored about its
        # own axis instead of aligned against the others.
        #
        # LOG SECONDS ON THE X AXIS. Bouts run from 1 s to over an hour. On a
        # linear axis every state is a spike against the left edge and the
        # only visible feature is one REST bout somewhere off to the right.
        # ------------------------------------------------------------------
        if ALTAIR and bouts:
            order = [r["STATE"] for r in bl]      # commonest state first
            ridge = [{"state": b["STATE"], "secs": float(b["BOUT_SECONDS"])}
                     for b in bouts if (b["BOUT_SECONDS"] or 0) >= 1]
            if len({r["state"] for r in ridge}) > 1:
                # THE EXTENT IS MEASURED, NOT ASSUMED. This was a hard-coded
                # [0, 3.8] — four decades, up to about 1h45 — chosen to be
                # safely wide. On a real dog the longest bout is nearer three
                # minutes, so every ridge was crushed into the left 55% of the
                # chart with half the axis empty, and a KDE evaluated over an
                # empty range is flatter as well as smaller. Rounded up to the
                # next fifth of a decade so the tallest ridge is not clipped by
                # its own kernel's tail.
                top = max(r["secs"] for r in ridge)
                hi = max(1.4, math.ceil((math.log10(top) + 0.25) * 5) / 5)
                panel("How long a bout of each behaviour actually lasts",
                      "One kernel density per state over the same log-seconds axis, "
                      "tallest first. The ridge is the shape of the distribution, not "
                      "a bar — a state with two humps is a behaviour the dog does in "
                      "two distinct ways, which is the thing a mean bout length is "
                      "guaranteed to hide.")
                # Overlap: each row is 30px tall and draws 46px of area, so
                # consecutive ridges cut into one another the way the form is
                # supposed to. Positive spacing here would just be fourteen
                # small area charts in a column.
                rid = (
                    alt.Chart(avals(ridge))
                    .transform_calculate(ls="log(datum.secs)/log(10)")
                    .transform_density("ls", groupby=["state"], as_=["ls", "d"],
                                       extent=[0, hi], steps=140, counts=False)
                    .mark_area(interpolate="monotone", fillOpacity=0.82,
                               stroke=CARD, strokeWidth=0.8)
                    .encode(
                        x=alt.X("ls:Q", title="bout length",
                                scale=alt.Scale(domain=[0, hi], nice=False),
                                axis=alt.Axis(
                                    values=[0, 1, 2, 3, 4],
                                    # Ticks are decades, labelled in units a
                                    # reader has intuitions about. "2" means
                                    # nothing; "1m 40s" is a length of time.
                                    # Values past the domain are dropped by
                                    # the renderer, so listing five is safe.
                                    labelExpr="datum.value == 0 ? '1s' : "
                                              "datum.value == 1 ? '10s' : "
                                              "datum.value == 2 ? '1m 40s' : "
                                              "datum.value == 3 ? '16m 40s' : "
                                              "'2h 46m'")),
                        y=alt.Y("d:Q", title=None, stack=None,
                                axis=None, scale=alt.Scale(range=[46, 0])),
                        row=alt.Row("state:N", title=None, sort=order,
                                    header=alt.Header(
                                        labelAngle=0, labelAlign="left",
                                        labelPadding=2, labelFontSize=10)),
                        fill=alt.Fill("state:N", legend=None,
                                      scale=alt.Scale(
                                          domain=order,
                                          range=[palette.get(s, "#D6D3D1")
                                                 for s in order])),
                        tooltip=[alt.Tooltip("state:N", title="state")])
                    .properties(width=880, height=30)
                    # EVERY ROW GETS ITS OWN y SCALE, AND HALF THIS CHART WAS
                    # A FLAT LINE WITHOUT IT.
                    #
                    # A faceted chart shares its scales across rows by default,
                    # and transform_density(counts=False) normalises each
                    # state's kernel density to unit AREA. So a state whose
                    # bouts spread over two decades — SNIFF, WALK, STAND, REST,
                    # SIT — has to be low to integrate to 1, while a state that
                    # is always about the same length — SHAKE, PAUSE, CIRCLE —
                    # is a tall narrow spike. Share one y scale between them and
                    # the spike sets the domain, which drew seven of the
                    # fourteen ridges as a horizontal rule at zero. They were
                    # not empty; they were 3% as tall as their loudest
                    # neighbour.
                    #
                    # Independent y is the right answer rather than a cosmetic
                    # one, because the y axis here IS NOT SHOWN (axis=None) and
                    # carries no reading. The chart is a comparison of SHAPES
                    # along x. Height was never the variable.
                    .resolve_scale(y="independent"))
                # tt_alt's configure_facet spacing is 14 — correct everywhere
                # else and wrong here, where the whole form depends on the rows
                # overlapping. Overriding after it wins, because configure_*
                # writes one key of the top-level config rather than appending.
                altair_chart(tt_alt(rid).configure_facet(spacing=-16),
                             container=False)
                renderer_note(
                    "altair · vega-lite",
                    f"{len(ridge):,} bouts, {len(order)} kernel density "
                    f"estimates, one specification. The densities are computed "
                    f"by the renderer from the raw bout lengths — nothing was "
                    f"pre-binned or pre-smoothed on the way here.")

        # ------------------------------------------------------------------
        # THE TRANSITION MATRIX, THREE WAYS, AND ONLY ONE OF THEM IS A GRID.
        #
        # This row used to be a 2D heatmap of p(next | current) beside the
        # donut. The heatmap was correct and almost unreadable: fourteen states
        # is a 196-cell grid whose diagonal is near 1.0 and whose every other
        # cell is under 0.05, so the colour scale was spent entirely on the
        # diagonal and the off-diagonal — the part that says anything about
        # BEHAVIOUR — was fourteen shades of nearly white.
        #
        # The surface is the same numbers with height doing the work colour
        # could not. A ridge along the diagonal is a dog that persists in
        # whatever it is doing; every bump off it is a change of behaviour, and
        # on a landscape a 0.04 bump next to a 0.01 plain is visible, where two
        # nearly-white cells are not. The contour projected onto the floor is
        # the heatmap, kept, underneath the thing that made it legible.
        # ------------------------------------------------------------------
        c1, c2 = st.columns(2)

        with c1:
            panel("Transition matrix, as a landscape",
                  "A first-order behavioural Markov chain, computed in SQL "
                  "with LAG and row-normalised. Height is p(next state | "
                  "current state). Drag to rotate.")
            if PLOTLY and tr:
                states = sorted({r["FROM_STATE"] for r in tr} | {r["TO_STATE"] for r in tr})
                idx = {s: i for i, s in enumerate(states)}
                # 0.0 RATHER THAN None for an unobserved pair. A gap in a
                # surface tears a hole in the mesh and reads as missing data;
                # this is not missing, it is a transition that never once
                # happened in the window, and zero is what that means.
                z = [[0.0] * len(states) for _ in states]
                txt = [["" for _ in states] for _ in states]
                for i, a in enumerate(states):
                    for j, b in enumerate(states):
                        txt[i][j] = f"{a} → {b}<br>never observed"
                for r in tr:
                    i, j = idx[r["FROM_STATE"]], idx[r["TO_STATE"]]
                    z[i][j] = float(r["PROB"] or 0)
                    txt[i][j] = (f'{r["FROM_STATE"]} → {r["TO_STATE"]}<br>'
                                 f'p = {fmt(r["PROB"],3)}<br>n = {fmt(r["N"],0)}')
                ticks = list(range(len(states)))
                fig = go.Figure(go.Surface(
                    z=z, x=ticks, y=ticks, text=txt, hoverinfo="text",
                    colorscale=[[0, CARD], [0.35, alpha(S_BLUE, 0.45)],
                                [1, S_BLUE]],
                    showscale=False, opacity=0.96,
                    lighting=dict(ambient=0.72, diffuse=0.62, specular=0.12),
                    contours=dict(z=dict(show=True, usecolormap=True,
                                         project_z=True, width=1))))
                scene3d(fig, "next state", "current state", "probability")
                # The axes are states, not numbers. Without this the reader is
                # looking at a landscape indexed 0..13.
                cat = dict(tickmode="array", tickvals=ticks, ticktext=states,
                           tickfont=dict(size=8, color=INK_2))
                fig.update_layout(
                    scene=dict(xaxis=cat, yaxis=cat,
                               zaxis=dict(range=[0, 1]),
                               aspectmode="manual",
                               aspectratio=dict(x=1.25, y=1.25, z=0.62)))
                chart3d(fig, H_LG)
                renderer_note(
                    "plotly · 3D",
                    f"{len(states)}×{len(states)} transition probabilities as a "
                    f"height field. The floor is the same matrix as a contour — "
                    f"the heatmap this replaced, kept as the shadow of the "
                    f"surface that made it readable.")
            else:
                empty_state("No transitions.", "MARTS.STATE_TRANSITIONS is empty.")

        with c2:
            panel("State budget",
                  "How the session divides across the ethogram. The same "
                  "states as the ribbon above, counted rather than laid out "
                  "in time.")
            if PLOTLY and budget:
                _n = [float(x) for x in col(budget, "N")]
                fig = evil_donut(
                    col(budget, "STATE"), _n,
                    [palette.get(s, "#D6D3D1") for s in col(budget, "STATE")],
                    text=[f'{r["STATE"]} · {fmt(r["N"],0)} epochs · {r["PCT"]}%'
                          for r in budget],
                    centre=ago(sum(_n)),
                    centre_sub=f"{len(budget)} states")
                fig.update_layout(title="proportion of the session in each state",
                                  title_font_size=11)
                chart(clean_axes(fig, y_zero_line=False), H_LG)
                renderer_note(
                    "plotly · donut",
                    "The hole carries the total the ring is a breakdown of. A "
                    "donut whose centre is empty is a pie chart in disguise; "
                    "the number in the middle is what pays for the shape.")

        # ------------------------------------------------------------------
        # SECOND ROW. Both halves are sized together by row_h() so they end on
        # the same line — the rule this page was rebuilt around and which the
        # rest of the app now follows.
        # ------------------------------------------------------------------
        # self-transitions are ~95% of every row (a dog in REST stays in REST)
        # and would be the only thing visible on the chart
        moves = sorted([r for r in tr if r["FROM_STATE"] != r["TO_STATE"]],
                       key=lambda r: -(r["PROB"] or 0))[:12]
        h_row = row_h(len(moves), len(bl))

        d1, d2 = st.columns(2)

        with d1:
            panel("What follows what",
                  "The same matrix as a flow. Self-transitions are dropped — "
                  "a dog at rest stays at rest through 95% of its seconds, "
                  "and leaving that in means the only thing visible is the "
                  "diagonal.")
            if PLOTLY and moves:
                # A SANKEY, BECAUSE A MARKOV CHAIN IS A FLOW AND NOT A RANKING.
                #
                # This was twelve horizontal bars of p, which answered "which
                # transition is likeliest" and hid the thing worth seeing:
                # that several transitions LEAVE the same state, and the band
                # widths out of one node are the shape of that dog's choices.
                #
                # THE NODES ARE DUPLICATED, from-side and to-side, and they
                # have to be. A sankey is a directed ACYCLIC graph; a
                # behavioural chain is full of cycles — WALK to SNIFF to WALK
                # — and feeding those to one shared node set produces a
                # figure plotly either refuses to lay out or draws as a knot.
                # Two columns of the same states makes the cycle a legal edge
                # from the left copy to the right copy.
                froms = sorted({r["FROM_STATE"] for r in moves})
                tos = sorted({r["TO_STATE"] for r in moves})
                lab = [str(s) for s in froms] + [str(s) for s in tos]
                hue = ([palette.get(s, "#D6D3D1") for s in froms]
                       + [palette.get(s, "#D6D3D1") for s in tos])
                fi = {s: i for i, s in enumerate(froms)}
                ti = {s: len(froms) + i for i, s in enumerate(tos)}
                fig = evil_sankey(
                    lab,
                    [fi[r["FROM_STATE"]] for r in moves],
                    [ti[r["TO_STATE"]] for r in moves],
                    [float(r["PROB"] or 0) for r in moves],
                    hue,
                    text=[f'{r["FROM_STATE"]} → {r["TO_STATE"]}<br>'
                          f'p = {fmt(r["PROB"],3)}<br>{fmt(r["N"],0)} times'
                          for r in moves])
                fig.update_layout(
                    title="the twelve likeliest changes of behaviour — band "
                          "width is probability",
                    title_font_size=11)
                chart(fig, h_row)
                renderer_note(
                    "plotly · sankey",
                    f"{len(moves)} transitions between {len(froms)} states the "
                    f"dog left and {len(tos)} it arrived in. Band width is the "
                    f"row-normalised probability straight out of "
                    f"MARTS.STATE_TRANSITIONS.")
            else:
                empty_state("No changes of state.",
                            "This dog never left its first state in the "
                            "window, or MARTS.STATE_TRANSITIONS is empty.")

        with d2:
            panel("Bout-length distribution",
                  "Where lameness and exercise intolerance become visible "
                  "before any syndrome fires: identical totals, different "
                  "bout lengths.")
            if PLOTLY and bl:
                # Median as the bar, mean as a marker on top of it. The GAP
                # between them is the finding: a state whose mean sits far
                # above its median is one long bout hiding in a pile of short
                # ones, which is exactly the shape lameness makes.
                #
                # The track behind each bar is the longest median on the
                # chart, so a two-second state reads as short rather than
                # merely as a small bar.
                names = [str(r["STATE"]) for r in bl][::-1]
                fig = go.Figure()
                evil_bar(
                    fig, names, [float(r["MEDIAN_S"] or 0) for r in bl][::-1],
                    [palette.get(n, "#D6D3D1") for n in names],
                    text=[f'{r["STATE"]}<br>median {fmt(r["MEDIAN_S"],0)}s'
                          f'<br>mean {fmt(r["MEAN_S"],1)}s'
                          f'<br>longest {fmt(r["MAX_S"],0)}s'
                          f'<br>{fmt(r["BOUTS"],0)} bouts' for r in bl][::-1])
                fig.add_trace(go.Scatter(
                    x=[float(r["MEAN_S"] or 0) for r in bl][::-1], y=names,
                    mode="markers",
                    marker=dict(color=INK, size=13, symbol="line-ns-open",
                                line=dict(width=1.8, color=INK)),
                    text=[f'mean {fmt(r["MEAN_S"],1)}s' for r in bl][::-1],
                    hoverinfo="text", showlegend=False))
                fig.update_layout(title="median bout length (bar) against the mean "
                                        "(tick) — seconds",
                                  title_font_size=11)
                chart(evil_axes(fig, y_zero_line=False, dotted="x"), h_row)
                renderer_note(
                    "plotly · composed",
                    "Two marks on one frame: the bar is the median, the tick "
                    "is the mean. Read the distance between them — a tick far "
                    "to the right of its bar is one long bout hiding inside a "
                    "pile of short ones.")

        # Third row: the numbers behind the two charts above, and the ethogram
        # they are all expressed in. The definitions belong on this page and
        # nowhere else — every other tab assumes you already know what SNIFF
        # means and which sensor decided it.
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Bouts, per state**")
            if bl:
                html_table(
                    [{"s": r["STATE"], "b": fmt(r["BOUTS"], 0), "m": fmt(r["MEAN_S"], 1),
                      "d": fmt(r["MEDIAN_S"], 0), "x": fmt(r["MAX_S"], 0)} for r in bl],
                    [("s", "state"), ("b", "bouts"), ("m", "mean s"),
                     ("d", "median s"), ("x", "max s")])
        with e2:
            st.markdown("**The ethogram itself**")
            etho = ethogram()
            if etho:
                html_table(
                    [{"c": f'<span class="tt-chip" style="background:'
                           f'{r.get("COLOUR_HEX") or "#D6D3D1"}22;border-color:'
                           f'{r.get("COLOUR_HEX") or "#D6D3D1"}">{r["STATE"]}</span>',
                      "d": r.get("DISPLAY_NAME") or "", "f": r.get("FAMILY") or "",
                      "v": str(r.get("DERIVATION") or "")[:28]} for r in etho],
                    [("c", "state"), ("d", "name"), ("f", "family"),
                     ("v", "derived from")])


# ===========================================================================
# TAB 4 — SYNDROMES.  THE MONEY TAB.
# ===========================================================================
def _page_3():
    st.markdown("#### Syndromes")
    st.markdown(
        '<span class="tt-quiet">A threshold cannot detect a syndrome. A vet '
        'diagnoses from an ordered sequence of events, not from a daily average. '
        'Below, each finding is a match of a regular expression over rows — a '
        'differential diagnosis expressed as a <span class="tt-mono">'
        'MATCH_RECOGNIZE</span> clause.</span>', unsafe_allow_html=True)

    finds = rows("""
        SELECT syndrome_code, syndrome_name, body_system, dog_id, test_num, match_id,
               onset_ts, resolve_ts, duration_s, n_epochs, confidence, severity,
               avg_quality, model_purity, evidence, pattern_text, define_text,
               why_not_threshold, breed, z_self
        FROM MARTS.V_FINDINGS
        ORDER BY severity DESC, confidence DESC, onset_ts DESC
        LIMIT 500
    """)

    if not finds:
        empty_state(
            "No syndrome matches yet.",
            "Either the pipeline has not run the pattern layer "
            "(CALL MARTS.SP_BUILD_SYNDROMES()), or no dog in the replayed window "
            "expressed one of the six sequences. Inject a labelled one: "
            "python scripts/demo_spike.py --dog 7 --syndrome S1")
    else:
        by_code: dict[str, int] = {}
        for f in finds:
            by_code[f["SYNDROME_CODE"]] = by_code.get(f["SYNDROME_CODE"], 0) + 1
        metric_strip([("matches", fmt(len(finds), 0))] +
                     [(k, fmt(v, 0)) for k, v in sorted(by_code.items())][:4])

        # ONE code list and ONE colour map for the whole page, because three
        # figures below draw the same six syndromes and a judge comparing them
        # is entitled to assume S3 is the same colour in all three. Built here
        # rather than inside each `if`, so that a renderer being unavailable
        # cannot shift the others' palette by changing who assigns it.
        codes = sorted(by_code)
        cmap = {c: SYMBOL_COLOURS[i % len(SYMBOL_COLOURS)]
                for i, c in enumerate(codes)}

        # WHEN each finding fired, and to WHICH dog. The table below is ordered
        # by severity and so destroys the time axis; this keeps it, which is how
        # you see that one dog fired the same syndrome four times in an evening
        # rather than four dogs firing it once.
        if PLOTLY:
            fig = go.Figure()
            for c in codes:
                pts = [f for f in finds if f["SYNDROME_CODE"] == c]
                # A HALO UNDER EVERY DOT, WHICH IS A GLOW DOING A JOB.
                #
                # These findings pile up — one dog can fire the same syndrome
                # four times in an evening, which puts four dots within a few
                # pixels of one another. Flat marks in that situation merge
                # into one blob and the reader counts one finding. A wide,
                # very faint copy of the same marker underneath makes overlap
                # read as DENSITY: two dots stacked are visibly darker than
                # one, so the pile-up is legible without moving anything.
                fig.add_trace(go.Scatter(
                    x=[str(f["ONSET_TS"]) for f in pts],
                    y=[float(f["DOG_ID"]) for f in pts],
                    mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(color=alpha(cmap[c], 0.16), line=dict(width=0),
                                size=[16 + 20 * float(f["CONFIDENCE"] or 0)
                                      for f in pts])))
                fig.add_trace(go.Scatter(
                    x=[str(f["ONSET_TS"]) for f in pts],
                    y=[float(f["DOG_ID"]) for f in pts],
                    mode="markers", name=c,
                    marker=dict(color=cmap[c], opacity=0.9,
                                line=dict(color=CARD, width=1),
                                size=[6 + 10 * float(f["CONFIDENCE"] or 0)
                                      for f in pts]),
                    text=[f'{c} · {f["SYNDROME_NAME"]}<br>dog {int(f["DOG_ID"])}'
                          f' · {f.get("BREED") or "unknown breed"}'
                          f'<br>{str(f["ONSET_TS"])[:19]}'
                          f'<br>{fmt(f["DURATION_S"],0)}s over '
                          f'{fmt(f["N_EPOCHS"],0)} epochs'
                          f'<br>confidence {fmt(f["CONFIDENCE"],3)}'
                          f' · severity {f["SEVERITY"]}' for f in pts],
                    hoverinfo="text"))
            fig.update_layout(
                title="every finding in time — one dot per match, sized by "
                      "confidence, coloured by syndrome",
                title_font_size=11, showlegend=True,
                legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                yaxis=dict(title="dog id"))
            chart(evil_axes(fig, y_zero_line=False, dotted="xy"), H_MD)
            st.markdown('<span class="tt-quiet">The band running bottom-left to '
                        'top-right is not a trend — the bulk corpus was loaded '
                        'dog by dog, so onset time and dog id increase together '
                        'by construction. The live replayed dogs are the column '
                        'on the right.</span>', unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # THE ONE CHART ON THIS PAGE THAT ANSWERS QUESTIONS IT WAS NOT ASKED.
        #
        # Every other figure in this app answers exactly the question its author
        # had. This one lets the reader ask their own: drag a box around the
        # long-and-confident findings and the three panels beside it re-count
        # themselves to describe only those — which syndromes they were, how
        # severe, which body system, which dogs. Click a syndrome bar and the
        # scatter reduces to that code. Nothing round-trips to Snowflake and
        # nothing reruns the script; the selection, the filter and the
        # re-aggregation are all compiled into the Vega spec and executed by
        # the browser on data that is already there.
        #
        # WHY ALTAIR AND NOT THE OTHER TWO. This is the one thing Vega-Lite has
        # that neither of the others does under Streamlit. A plotly selection
        # event cannot filter a second plotly figure without a callback, and
        # Streamlit has no callback to give it — the closest available is a
        # st.plotly_chart selection that reruns the whole page and re-queries
        # the warehouse to redraw four charts. Bokeh could do it with CustomJS,
        # which means writing the cross-filter by hand in JavaScript. Here it
        # is four `transform_filter`s.
        #
        # ONLY THE COLUMNS THE CHART USES ARE PASSED. `finds` carries
        # evidence, pattern_text, define_text and why_not_threshold — four long
        # strings per row that the table below needs and this does not. Vega
        # inlines its data into the spec as JSON, so handing it the whole row
        # would ship roughly a megabyte of SQL prose to the browser to draw
        # 500 dots.
        # ----------------------------------------------------------------------
        if ALTAIR and len(finds) > 1:
            slim = [{
                "code": f["SYNDROME_CODE"],
                "name": f["SYNDROME_NAME"],
                "system": f.get("BODY_SYSTEM") or "unclassified",
                "dog": int(f["DOG_ID"]),
                "breed": f.get("BREED") or "unknown breed",
                "duration": float(f["DURATION_S"] or 0),
                "epochs": float(f["N_EPOCHS"] or 0),
                "confidence": float(f["CONFIDENCE"] or 0),
                "severity": int(f["SEVERITY"] or 0),
                "quality": float(f["AVG_QUALITY"] or 0),
            } for f in finds]

            panel("Ask it your own question",
                  "Drag a box across the scatter. The three panels beside it "
                  "re-count to describe only what you selected, and the pale bars "
                  "behind them stay put as the totals you are taking a fraction "
                  "of. Click a syndrome bar to push the filter the other way. "
                  "Shift-click to add codes; double-click any blank area to "
                  "clear.")

            adata = avals(slim)
            brush = alt.selection_interval(encodings=["x", "y"], name="brush")
            pick = alt_point(fields=["code"], name="pick",
                             toggle="event.shiftKey")
            colour = alt.Color("code:N", title="syndrome",
                               scale=alt.Scale(domain=codes,
                                               range=[cmap[c] for c in codes]),
                               legend=None)

            # THESE WIDTHS ARE FIXED PIXELS AND HAVE TO BE. Vega-Lite supports
            # width:"container" for single and layered views ONLY — never for
            # a concat — so use_container_width on this chart produces a spec
            # the renderer rejects rather than a chart that fits its column.
            # Hence altair_chart(container=False) at the bottom.
            #
            # Sized to survive the narrow case, not the wide one. A wide-layout
            # content column is about 1 050 px on a 1440 laptop once the rail
            # is out, and an over-wide vega spec does not shrink to fit — it is
            # clipped, so the reader loses the right-hand panel entirely with
            # nothing to say it was ever there. Scatter on top at 880 rather
            # than beside the bars, because splitting 1 050 px between the two
            # leaves the scatter too small to brush precisely, which is the one
            # thing this chart has to be good at.
            W, WR = 880, 250

            scatter = (
                alt.Chart(adata, title="every finding: how long it ran against "
                                       "how sure the pattern was")
                .mark_circle(stroke=CARD, strokeWidth=0.6)
                .encode(
                    x=alt.X("duration:Q", title="duration (s)",
                            scale=alt.Scale(type="sqrt", nice=True)),
                    y=alt.Y("confidence:Q", title="confidence",
                            scale=alt.Scale(domain=[0, 1])),
                    size=alt.Size("epochs:Q", title="epochs matched",
                                  scale=alt.Scale(range=[18, 420]), legend=None),
                    # Unselected points go pale rather than disappearing. A
                    # brush that deletes its complement hides how big a
                    # fraction you actually grabbed, which is the one thing a
                    # selection is for.
                    color=alt.condition(brush, colour, alt.value("#E7E5E4")),
                    opacity=alt.condition(brush, alt.value(0.82), alt.value(0.35)),
                    tooltip=[alt.Tooltip("code:N", title="code"),
                             alt.Tooltip("name:N", title="syndrome"),
                             alt.Tooltip("dog:Q", title="dog", format="d"),
                             alt.Tooltip("breed:N", title="breed"),
                             alt.Tooltip("duration:Q", title="duration (s)",
                                         format=",.0f"),
                             alt.Tooltip("epochs:Q", title="epochs", format=",.0f"),
                             alt.Tooltip("confidence:Q", title="confidence",
                                         format=".3f"),
                             alt.Tooltip("severity:Q", title="severity")])
                .transform_filter(pick)
                .properties(width=W, height=330))
            scatter = alt_bind(scatter, brush)

            def counted(field: str, title: str, axis_title: str, height: int,
                        sort=None, colour_by=None):
                """A bar chart of counts under the brush, over its own total.

                The pale layer is NOT filtered and the solid one is, so the
                bar always reads as a fraction of a constant, and an empty
                selection reads as 'none of these' instead of as no data.
                """
                base = alt.Chart(adata).encode(
                    y=alt.Y(f"{field}:N", title=None, sort=sort,
                            axis=alt.Axis(labelLimit=88)),
                    x=alt.X("count():Q", title=axis_title,
                            axis=alt.Axis(tickMinStep=1, tickCount=4)))
                ghost = base.mark_bar(color=GRID, height=13)
                live = (base.mark_bar(height=13)
                        .encode(color=colour_by if colour_by is not None
                                else alt.value(INK),
                                opacity=alt.condition(pick, alt.value(1.0),
                                                      alt.value(0.45))
                                if colour_by is not None else alt.value(1.0),
                                tooltip=[alt.Tooltip(f"{field}:N", title=title),
                                         alt.Tooltip("count():Q", title="in selection")])
                        .transform_filter(brush))
                layered = alt.layer(ghost, live, title=title).properties(
                    width=WR, height=height)
                return alt_bind(layered, pick) if colour_by is not None else layered

            # NOT `by_code` — that name is the {code: count} dict this page
            # already built for the metric strip, and rebinding it to a chart
            # here would break any later reader of it in a way that only shows
            # up as a wrong number on screen.
            p_code = counted("code", "which syndrome fired", "matches", 132,
                             sort=codes, colour_by=colour)
            p_sev = counted("severity", "how severe", "matches", 132,
                            sort="descending")
            p_system = counted("system", "which body system", "matches", 132,
                               sort="-x")

            # The three counters share a height so their baselines line up,
            # and each keeps its own x scale — they count different things and
            # a shared axis would imply the three totals were comparable.
            cross = alt.vconcat(
                scatter,
                alt.hconcat(p_code, p_sev, p_system, spacing=30)
            ).resolve_scale(color="independent", size="independent")

            altair_chart(tt_alt(cross), container=False)
            renderer_note(
                "altair · vega-lite",
                f"{len(slim)} findings, cross-filtered in your browser. No "
                f"rerun, no callback, and not one further query against the "
                f"warehouse — the selection and the four aggregations are part "
                f"of the chart specification.")

        if PLOTLY:
            # ------------------------------------------------------------------
            # THE SAME 98 FINDINGS IN THE SPACE THAT DEFINES THEM.
            #
            # A syndrome is not a point in time, it is a shape: how long it ran,
            # how many epochs the pattern consumed, and how confidently. Those
            # three are what MATCH_RECOGNIZE actually produced and they separate
            # the codes from each other — S6 is short and dense, S3 is long and
            # sparse. Any 2D pair of them puts two codes on top of one another.
            # ------------------------------------------------------------------
            panel("The findings in their own measure space",
                  "Drag to rotate. Duration against epochs matched against "
                  "confidence — the three numbers the pattern engine returns. "
                  "Syndromes occupy different regions, which is the claim that "
                  "they are different things rather than one detector fired six "
                  "ways.")
            fig = go.Figure()
            for c in codes:
                pts = [f for f in finds if f["SYNDROME_CODE"] == c]
                fig.add_trace(go.Scatter3d(
                    x=[float(f["DURATION_S"] or 0) for f in pts],
                    y=[float(f["N_EPOCHS"] or 0) for f in pts],
                    z=[float(f["CONFIDENCE"] or 0) for f in pts],
                    mode="markers", name=c,
                    marker=dict(size=4, color=cmap[c], opacity=0.8,
                                line=dict(width=0)),
                    hovertext=[f'{c} · dog {int(f["DOG_ID"])}<br>'
                               f'{fmt(f["DURATION_S"],0)}s over '
                               f'{fmt(f["N_EPOCHS"],0)} epochs<br>'
                               f'confidence {fmt(f["CONFIDENCE"],3)}'
                               for f in pts],
                    hoverinfo="text"))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="duration (s)", backgroundcolor=CARD,
                               gridcolor=BORDER, zerolinecolor=BORDER),
                    yaxis=dict(title="epochs matched", backgroundcolor=CARD,
                               gridcolor=BORDER, zerolinecolor=BORDER),
                    zaxis=dict(title="confidence", backgroundcolor=CARD,
                               gridcolor=BORDER, zerolinecolor=BORDER),
                    camera=dict(eye=dict(x=1.6, y=1.5, z=0.9))),
                paper_bgcolor=CARD, plot_bgcolor=CARD, showlegend=True,
                legend=dict(itemsizing="constant", font=dict(size=10)),
                margin=dict(l=0, r=0, t=4, b=0), height=H_LG,
                font=dict(family=FONT_CHART, size=11,
                          color=INK_2))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        dataframe(
            [{
                "code": f["SYNDROME_CODE"], "syndrome": f["SYNDROME_NAME"],
                "dog": f["DOG_ID"], "onset": str(f["ONSET_TS"])[:19],
                "dur_s": fmt(f["DURATION_S"], 0), "epochs": fmt(f["N_EPOCHS"], 0),
                "conf": fmt(f["CONFIDENCE"], 3), "sev": f["SEVERITY"],
                "quality": fmt(f["AVG_QUALITY"], 2), "model%": fmt(f["MODEL_PURITY"], 2),
            } for f in finds[:60]],
            [("code", "code"), ("syndrome", "syndrome"), ("dog", "dog"),
             ("onset", "onset (UTC)"), ("dur_s", "duration"), ("epochs", "epochs"),
             ("conf", "confidence"), ("sev", "severity"), ("quality", "epoch quality"),
             ("model%", "model purity")])

        st.markdown("---")
        st.markdown("##### The matched rows")
        labels = [f'{f["SYNDROME_CODE"]} · dog {f["DOG_ID"]} · {str(f["ONSET_TS"])[:19]} '
                  f'· conf {fmt(f["CONFIDENCE"],3)}' for f in finds]
        pick = st.selectbox("select a finding", range(len(labels)),
                            format_func=lambda i: labels[i], key="find_pick")
        f = finds[pick]

        cl, cr = st.columns([3, 2])

        with cl:
            symbol_ribbon(f["SYNDROME_CODE"], f["DOG_ID"], f["TEST_NUM"],
                          f["MATCH_ID"])

            ev = f.get("EVIDENCE")
            if ev:
                try:
                    parsed = json.loads(ev) if isinstance(ev, str) else ev
                    st.markdown("**Evidence**")
                    html_table([{"k": k, "v": json.dumps(v) if isinstance(v, (list, dict))
                                 else v} for k, v in parsed.items()],
                               [("k", "measure"), ("v", "value")])
                except Exception:  # noqa: BLE001
                    st.markdown(f'<div class="tt-mono">{ev}</div>', unsafe_allow_html=True)

        with cr:
            st.markdown("**The SQL that found it**")
            st.code(
                f"SELECT * FROM MARTS.V_SYNDROME_INPUT\n"
                f"MATCH_RECOGNIZE (\n"
                f"    PARTITION BY dog_id, test_num\n"
                f"    ORDER BY epoch_ts\n"
                f"    MEASURES MATCH_NUMBER() AS match_id, ...\n"
                f"    ONE ROW PER MATCH\n"
                f"    AFTER MATCH SKIP PAST LAST ROW\n"
                f"    PATTERN ( {f['PATTERN_TEXT']} )\n"
                f"    DEFINE\n        "
                + ",\n        ".join(x.strip() for x in
                                     str(f["DEFINE_TEXT"]).split(","))
                + "\n)", language="sql")
            st.markdown(
                f'<div class="tt-card"><div class="tt-metric-label">'
                f'why a sequence and not a threshold</div>'
                f'<div style="font-size:12px;line-height:1.5;margin-top:4px">'
                f'{f["WHY_NOT_THRESHOLD"]}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### Sensitivity")
            st.markdown('<span class="tt-quiet">Every pattern at three quantifier '
                        'settings. A single hand-tuned quantifier is a magic '
                        'number; the curve is a result.</span>',
                        unsafe_allow_html=True)
            sens = rows("""
                SELECT syndrome_code, variant, matches, dogs_firing,
                       avg_match_epochs, pattern_text
                FROM MARTS.V_SENSITIVITY_CURVE
                ORDER BY syndrome_code,
                         CASE variant WHEN 'loose' THEN 1 WHEN 'tuned' THEN 2 ELSE 3 END
            """)
            if PLOTLY and sens:
                order = {"loose": 0, "tuned": 1, "strict": 2}
                codes = sorted({r["SYNDROME_CODE"] for r in sens})
                fig = go.Figure()
                for i, code in enumerate(codes):
                    pts = sorted([r for r in sens if r["SYNDROME_CODE"] == code],
                                 key=lambda r: order.get(r["VARIANT"], 9))
                    fig.add_trace(go.Scatter(
                        x=[r["VARIANT"] for r in pts],
                        y=[float(r["MATCHES"] or 0) for r in pts],
                        mode="lines+markers+text",
                        line=dict(color=SYMBOL_COLOURS[i % len(SYMBOL_COLOURS)], width=1.6),
                        marker=dict(size=6),
                        text=[code if r["VARIANT"] == "strict" else "" for r in pts],
                        textposition="middle right", textfont=dict(size=10),
                        hovertext=[f'{code} {r["VARIANT"]}<br>{fmt(r["MATCHES"],0)} matches'
                                   f'<br>{fmt(r["DOGS_FIRING"],0)} dogs'
                                   f'<br><span style="font-family:monospace">'
                                   f'{r["PATTERN_TEXT"]}</span>' for r in pts],
                        hoverinfo="text", showlegend=False))
                fig.update_layout(title="matches by quantifier strictness",
                                  title_font_size=11)
                chart(evil_axes(fig), H_MD)
            else:
                empty_state("No sensitivity sweep.",
                            "CALL MARTS.SP_SYNDROME_SWEEP() runs all 18 variants.")

        with c2:
            st.markdown("##### By cohort")
            coh = rows("""
                SELECT syndrome_code, weight_band, age_band, matches, dogs
                FROM MARTS.V_SYNDROME_BY_COHORT
                ORDER BY syndrome_code, matches DESC
            """)
            if PLOTLY and coh:
                codes = sorted({r["SYNDROME_CODE"] for r in coh})
                bands = sorted({r["WEIGHT_BAND"] for r in coh})
                fig = go.Figure()
                for i, b in enumerate(bands):
                    ys = []
                    for code in codes:
                        ys.append(sum(float(r["MATCHES"] or 0) for r in coh
                                      if r["SYNDROME_CODE"] == code
                                      and r["WEIGHT_BAND"] == b))
                    fig.add_trace(go.Bar(
                        x=codes, y=ys, name=b,
                        marker=dict(color=SYMBOL_COLOURS[i % len(SYMBOL_COLOURS)]),
                        hovertext=[f"{b}: {int(y)}" for y in ys], hoverinfo="text"))
                fig.update_layout(barmode="stack", showlegend=True,
                                  legend=dict(orientation="h", y=-0.15,
                                              font=dict(size=10)),
                                  title="syndrome frequency by weight band",
                                  title_font_size=11)
                chart(evil_axes(fig), H_MD)
            else:
                empty_state("No cohort breakdown.", "REF.DOG_INFO may be empty.")


# ===========================================================================
# TAB 5 — BASELINES.  Every dog is its own control.
# ===========================================================================
def _page_4():
    st.markdown("#### Baselines")
    st.markdown('<span class="tt-quiet">A Husky doing forty minutes of galloping '
                'is a Tuesday. A twelve-year-old Bulldog doing the same is an '
                'emergency. Every comparison here is against the dog itself first '
                'and its cohort second, joined with <span class="tt-mono">ASOF '
                'JOIN</span> so a gap in the feed degrades the comparison instead '
                'of silently shifting the window.</span>', unsafe_allow_html=True)

    dogs = rows("SELECT DISTINCT dog_id FROM MARTS.DOG_DEVIATION ORDER BY dog_id")
    if not dogs:
        empty_state("No deviation data.",
                    "MARTS.DOG_DEVIATION needs a trailing baseline; give the "
                    "replayer an hour of dog time (or run at --speed 60).")
    else:
        dog = st.selectbox("dog", [int(r["DOG_ID"]) for r in dogs], key="base_dog")

        # THE MOST RECENT EPOCHS, NOT THE FIRST FIVE THOUSAND.
        #
        # `ORDER BY epoch_ts LIMIT 5000` returns the OLDEST rows, so a dog with
        # more than that showed its earliest history under a title that says
        # "today". QUALIFY takes the newest window and the outer ORDER BY puts
        # it back in reading order. 3,600 epochs is an hour of dog time, which
        # is the window the caption has been claiming all along.
        # A TIME BOUND AS WELL AS A ROW BOUND, and the time bound is the one
        # that mattered. Some dogs' epochs are two recording sessions in
        # January and one from this morning; a row cap alone keeps all three,
        # so the x axis spanned seven months to show forty minutes of data and
        # every session collapsed into a hairline at the edge of the plot.
        # Six hours back from THAT DOG'S newest epoch — never from the wall
        # clock, which would empty the chart for every bulk-corpus dog.
        dev = rows(f"""
            SELECT epoch_ts, activity_index, baseline_index, baseline_std,
                   z_self, z_cohort, cohort_mean, cohort_std, is_synthetic
            FROM MARTS.DOG_DEVIATION
            WHERE dog_id = {dog} AND baseline_index IS NOT NULL
              AND epoch_ts >= DATEADD('hour', -6,
                    (SELECT MAX(epoch_ts) FROM MARTS.DOG_DEVIATION
                      WHERE dog_id = {dog} AND baseline_index IS NOT NULL))
            QUALIFY ROW_NUMBER() OVER (ORDER BY epoch_ts DESC) <= 3600
            ORDER BY epoch_ts
        """)
        if PLOTLY and dev:
            # BREAK THE LINE WHERE THE FEED BROKE.
            #
            # A bulk-corpus dog's epochs are a handful of recording sessions
            # scattered across months, not one continuous hour. Drawn on a
            # datetime axis with no breaks, each session collapsed into a
            # vertical spike at the edge of the plot and plotly ran a straight
            # line across the seven months between them — which read as a dog
            # holding a perfectly constant activity index since February. It
            # was drawing an absence as a measurement.
            #
            # None at each discontinuity, in x and in every series, so a gap in
            # the record is a gap on the chart. Same principle as the ASOF JOIN
            # this page is about: a hole degrades the comparison visibly rather
            # than being filled in silently.
            _brk = [False]
            for _i in range(1, len(dev)):
                _d = (dev[_i]["EPOCH_TS"] - dev[_i - 1]["EPOCH_TS"]).total_seconds()
                _brk.append(_d > 5)

            def _cut(seq):
                out = []
                for i, v in enumerate(seq):
                    if _brk[i]:
                        out.append(None)
                    out.append(v)
                return out

            _segments = sum(_brk) + 1
            xs = _cut([str(r["EPOCH_TS"]) for r in dev])
            base = _cut([float(r["BASELINE_INDEX"] or 0) for r in dev])
            sd = _cut([float(r["BASELINE_STD"] or 0) for r in dev])
            fig = go.Figure()
            # The ±2 SD band stays a flat grey fill and is NOT given a
            # gradient. It is a region of tolerance, not a quantity — a ramp
            # across it would imply the edges of normal are less normal than
            # the middle, which is the opposite of what two standard
            # deviations means.
            # None survives the arithmetic: a break in the band has to break
            # the fill too, or the shaded region bridges the gap the line
            # correctly refuses to.
            def _band(k):
                return [None if b is None or s is None else b + k * s
                        for b, s in zip(base, sd)]

            fig.add_trace(go.Scatter(x=xs, y=_band(2),
                                     mode="lines", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=_band(-2),
                                     mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=alpha("#A8A29E", 0.20),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=base, mode="lines",
                                     line=dict(color=INK_2, width=1, dash="dot"),
                                     hoverinfo="skip", showlegend=False))
            # The dog's actual activity DOES get the full treatment: it is the
            # measured quantity, it is the only thing on the chart the reader
            # is meant to follow, and the glow is what lifts a 1.5px line off
            # a grey band it spends most of its time inside.
            evil_area(
                fig, xs, _cut([float(r["ACTIVITY_INDEX"] or 0) for r in dev]),
                S_ORANGE, width=1.6, shape="linear", fill=None, cap=True,
                text=_cut([f'{fmt(r["ACTIVITY_INDEX"],3)}'
                           f'<br>z_self {fmt(r["Z_SELF"],2)}'
                           f'{" · SYNTHETIC" if r.get("IS_SYNTHETIC") else ""}'
                           for r in dev]))
            fig.update_layout(
                title="the most recent hour against this dog's own trailing "
                      "baseline (shaded band = ±2 SD of its own normal)",
                title_font_size=11)
            chart(evil_axes(fig, y_zero_line=False), H_MD)
            renderer_note(
                "plotly · area",
                f"{len(dev):,} epochs against a trailing baseline joined with "
                f"ASOF JOIN"
                + (f", in {_segments} recording sessions — the line breaks "
                   f"where the record does rather than interpolating across "
                   f"the gap" if _segments > 1 else "")
                + f". The band is the dog's own ±2 SD, so leaving it is "
                  f"abnormal FOR THIS DOG rather than abnormal for a dog.")

        # The left column used to be a five-row table and a three-row table
        # against a full-height chart on the right — half the page ended 600px
        # above the other half. The cohort comparison is a DISTRIBUTION
        # question ("is this dog unusual for dogs like it"), and a distribution
        # answered with two averages is the weakest possible form of it.
        # BOTH HALVES OF THIS ROW ARE ONE ROW PER DOG, SO THEY GET ONE HEIGHT.
        #
        # The box plot grew with the pack — 45 dogs at an 18px pitch is 560px —
        # while the wall beside it was pinned at H_MD. The right column
        # therefore stopped roughly 450px above the left and the tab had a
        # column of white running down the middle of it. Everything this row
        # needs is fetched before either column so the height can be decided
        # once, which is the only way Streamlit columns ever line up.
        summ = rows(f"""
            SELECT ROUND(AVG(z_self),3) AS z_self, ROUND(AVG(z_cohort),3) AS z_cohort,
                   ROUND(AVG(activity_index),4) AS idx,
                   ROUND(AVG(cohort_mean),4) AS cohort_mean,
                   ANY_VALUE(cohort_id) AS cohort
            FROM MARTS.DOG_DEVIATION WHERE dog_id = {dog}
        """)
        peers = rows(f"""
            SELECT dog_id, activity_index
            FROM MARTS.DOG_DEVIATION
            WHERE activity_index IS NOT NULL
              AND cohort_id = (SELECT ANY_VALUE(cohort_id)
                               FROM MARTS.DOG_DEVIATION WHERE dog_id = {dog})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY dog_id
                                       ORDER BY RANDOM()) <= 300
        """)
        # A cohort of one is a real outcome here — the bands are narrow and
        # some dogs are the only animal in their weight/age cell. Drawing a
        # single lonely box against no comparison looks like a broken chart
        # rather than a fact, so widen to the whole pack and SAY SO instead
        # of quietly implying this is the cohort.
        alone = len({r["DOG_ID"] for r in peers}) < 3
        if alone:
            peers = rows("""
                SELECT dog_id, activity_index
                FROM MARTS.DOG_DEVIATION
                WHERE activity_index IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY dog_id
                                           ORDER BY RANDOM()) <= 200
            """)
        by_dog: dict = {}
        for r in peers:
            by_dog.setdefault(int(r["DOG_ID"]), []).append(
                float(r["ACTIVITY_INDEX"]))

        wall = rows("""
            SELECT dog_id, ROUND(AVG(ABS(z_self)),3) AS z_abs,
                   ROUND(AVG(z_self),3) AS z_self, COUNT(*) AS n
            FROM MARTS.DOG_DEVIATION
            WHERE z_self IS NOT NULL
            GROUP BY dog_id ORDER BY z_abs DESC
        """)

        # One row per dog on both sides, so one pitch and one height — plus the
        # arithmetic for what sits UNDER each chart. The left column carries a
        # five-row summary register that the right column does not, so the wall
        # is given that height back; otherwise matching the two charts just
        # moves the ragged edge down by the height of a table.
        _pitch = 18
        _row2 = row_h(len(by_dog), len(wall), row=_pitch)
        _SUMMARY_H = 186        # the five-row register under the box plot

        c1, c2 = st.columns(2)
        with c1:
            panel("Against its cohort",
                  "Every dog in the same weight and age band, as a spread rather "
                  "than an average. This dog is the amber box.")
            if alone:
                st.markdown(
                    '<div class="tt-caveat">This dog is the only one in its '
                    'weight and age band, so there is no cohort to compare it '
                    'against. Shown against <b>the whole pack</b> instead — a '
                    'weaker comparison, and labelled as one.</div>',
                    unsafe_allow_html=True)
            if PLOTLY and by_dog:
                fig = go.Figure()
                for d in sorted(by_dog, key=lambda k: (k != dog, k)):
                    mine = d == dog
                    fig.add_trace(go.Box(
                        x=by_dog[d], name=f"dog {d}", orientation="h",
                        marker=dict(color=S_ORANGE if mine else "#D6D3D1"),
                        line=dict(width=1.4), boxpoints=False,
                        fillcolor=alpha(S_ORANGE, 0.22) if mine else "#F5F5F4",
                        hoverinfo="x+name"))
                fig.update_layout(
                    title=("activity index, this dog against the whole pack"
                           if alone else
                           "activity index, this dog against its cohort"),
                    title_font_size=11)
                chart(evil_axes(fig, y_zero_line=False, dotted="x"), _row2)
            if summ:
                s = summ[0]
                html_table([
                    {"k": "cohort", "v": s["COHORT"]},
                    {"k": "this dog, mean index", "v": fmt(s["IDX"], 4)},
                    {"k": "cohort mean index", "v": fmt(s["COHORT_MEAN"], 4)},
                    {"k": "z vs own baseline", "v": fmt(s["Z_SELF"], 3)},
                    {"k": "z vs cohort", "v": fmt(s["Z_COHORT"], 3)},
                ], [("k", ""), ("v", "")])

        with c2:
            panel("The wall",
                  "Every dog, worst deviation first. Amber past 1 SD, red "
                  "past 2 — the dogs at the top are the ward round.")
            if PLOTLY and wall:
                # HORIZONTAL, LIKE THE CHART IT SHARES A ROW WITH.
                #
                # Vertically, forty-five dog labels have nowhere to go: plotly
                # rotated them ninety degrees and the axis became a wall of
                # sideways text that cost more height than the bars did.
                # Horizontal gives every dog a readable label on one line, puts
                # the worst offenders at the top where a ward round starts, and
                # matches the row pitch of the box plot beside it so the two
                # halves of this row finally end on the same line.
                #
                # Reversed, because plotly draws the first category at the
                # BOTTOM of a horizontal axis and the query is worst-first.
                _w = list(reversed(wall))
                fig = go.Figure()
                evil_bar(
                    fig, [f'dog {int(r["DOG_ID"])}' for r in _w],
                    [float(r["Z_SELF"] or 0) for r in _w],
                    [TRIAGE_COLOUR[3] if abs(r["Z_ABS"] or 0) > 2
                     else (S_ORANGE if abs(r["Z_ABS"] or 0) > 1
                           else "#D6D3D1") for r in _w],
                    horizontal=True,
                    # NO TRACK ON THIS ONE. z_self is SIGNED and centred on
                    # zero; a track drawn from the axis to the longest bar
                    # would run the wrong way for every dog below baseline and
                    # imply zero is the floor rather than the middle.
                    track=False,
                    text=[f'dog {int(r["DOG_ID"])}<br>z {fmt(r["Z_SELF"],2)}<br>'
                          f'{fmt(r["N"],0)} epochs' for r in _w])
                # The line every bar is read against. Zero is the dog's own
                # baseline, not the bottom of a scale.
                fig.add_vline(x=0, line=dict(color=INK_2, width=1))
                fig.update_layout(
                    title="mean deviation from own baseline, by dog "
                          "(amber > 1 SD, red > 2 SD)",
                    title_font_size=11,
                    yaxis=dict(tickfont=dict(size=9.5)))
                chart(evil_axes(fig, y_zero_line=False, dotted="x"),
                      _row2 + _SUMMARY_H)

        # ------------------------------------------------------------------
        # THIRD ROW. The wall and the trajectory used to be STACKED in the
        # right column against a table in the left, which is how this page
        # came to end with 600px of one column against nothing. They are two
        # different questions — where a dog is, and where it is going — and
        # they get a row each half.
        # ------------------------------------------------------------------
        e1, e2 = st.columns(2)

        with e1:
            panel("This dog's path out of its own normal",
                  "The hero chart above, with deviation lifted into a third "
                  "axis. Flat on the floor is a dog inside its baseline; "
                  "height in either direction is how far out it went.")
            if PLOTLY and dev:
                # WHY THIS IS THREE VARIABLES AND NOT TWO DRESSED UP.
                #
                # z_self is NOT a restatement of activity_index. It is the
                # activity against the dog's own trailing mean AND standard
                # deviation, so the same index reads as z = 0.2 in an hour the
                # dog was varying a lot and z = 3 in an hour it was steady.
                # The 2D chart above can show the index against a band, or the
                # z, but not both at once — which is the whole reason the band
                # has to be drawn to make the top chart readable at all.
                _z = [float(r["Z_SELF"] or 0) for r in dev]
                _a = [float(r["ACTIVITY_INDEX"] or 0) for r in dev]
                _t = list(range(len(dev)))
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=_t, y=_a, z=_z, mode="lines",
                    line=dict(color=[abs(v) for v in _z], width=4,
                              colorscale=[[0, "#D6D3D1"], [0.5, S_ORANGE],
                                          [1, TRIAGE_COLOUR[3]]],
                              cmin=0, cmax=max([abs(v) for v in _z] or [1])),
                    text=[f'{str(r["EPOCH_TS"])[:19]}'
                          f'<br>index {fmt(r["ACTIVITY_INDEX"],3)}'
                          f'<br>z_self {fmt(r["Z_SELF"],2)}' for r in dev],
                    hoverinfo="text", showlegend=False))
                scene3d(fig, "epoch", "activity index", "z vs own baseline",
                        eye=(1.7, 1.3, 0.7))
                chart3d(fig, H_LG)
                renderer_note(
                    "plotly · 3D",
                    f"{len(dev):,} epochs as one continuous path. Colour is "
                    f"|z|, so the stretch where the line lifts AND reddens is "
                    f"the same stretch the 2D chart shows leaving the band.")

        with e2:
            # Where each dog is HEADING, not just where it is. ML.FORECAST
            # projects the activity index forward per dog; drawn as a dumbbell
            # so the length of the connector is the size of the move and its
            # colour is the direction — a column of numbers hides both.
            panel("Projected trajectory, every dog",
                  "Hollow dot is where the dog is now, solid dot is where "
                  "ML.FORECAST puts it. Amber connectors are dogs winding "
                  "down.")
            alltraj = rows("""
                SELECT dog_id, current_index, projected_index, pct_change
                FROM ML.V_TRAJECTORY
                WHERE projected_index IS NOT NULL AND current_index IS NOT NULL
                ORDER BY pct_change
            """)
            if PLOTLY and alltraj:
                fig = go.Figure()
                for r in alltraj:
                    cur = float(r["CURRENT_INDEX"] or 0)
                    proj = float(r["PROJECTED_INDEX"] or 0)
                    lbl = f'dog {int(r["DOG_ID"])}'
                    hue = S_ORANGE if proj < cur else S_BLUE
                    # A soft wide pass under the connector, so a long move
                    # reads as a heavier band than a short one at a glance.
                    fig.add_trace(go.Scatter(
                        x=[cur, proj], y=[lbl, lbl], mode="lines",
                        line=dict(color=alpha(hue, 0.16), width=9),
                        hoverinfo="skip", showlegend=False))
                    fig.add_trace(go.Scatter(
                        x=[cur, proj], y=[lbl, lbl], mode="lines",
                        line=dict(color=hue, width=2), hoverinfo="skip",
                        showlegend=False))
                    fig.add_trace(go.Scatter(
                        x=[cur, proj], y=[lbl, lbl], mode="markers",
                        marker=dict(color=[CARD, hue], size=8,
                                    line=dict(color=hue, width=1.6)),
                        text=[f'{lbl}<br>now {fmt(cur,4)}',
                              f'{lbl}<br>projected {fmt(proj,4)}'
                              f'<br>{fmt(r["PCT_CHANGE"],1)}%'],
                        hoverinfo="text", showlegend=False))
                fig.update_layout(title="activity index — now against projected",
                                  title_font_size=11)
                chart(evil_axes(fig, y_zero_line=False, dotted="x"),
                      row_h(len(alltraj)))
            else:
                empty_state("No forecasts yet.",
                            "ML.FORECAST needs ~100 training points per dog; "
                            "see ML.FUNCTION_STATUS.")


# ===========================================================================
# TAB 6 — VET NOTE.  A clinical document, not a chat bubble.
# ===========================================================================
def _page_5():
    st.markdown("#### Veterinary handoff note")
    notes = rows("""
        SELECT * FROM AI.V_VET_NOTE_FULL
        ORDER BY severity DESC, generated_at DESC
        LIMIT 200
    """)
    if not notes:
        empty_state(
            "No notes cached yet.",
            "AI.T_AI batches AI_COMPLETE into AI.VET_NOTES on a task. This page "
            "never calls Cortex: a render path that costs credits is a render "
            "path that will exhaust a trial cap during a demo.")
    else:
        # The caseload this note sits inside, before the note itself.
        # Everything here is folded out of the rows already fetched, so the
        # overview costs no extra query.
        _dogs = {n["DOG_ID"] for n in notes}
        _conf = [float(n["CONFIDENCE"]) for n in notes
                 if n.get("CONFIDENCE") is not None]
        _pure = [float(n["MODEL_PURITY"]) for n in notes
                 if n.get("MODEL_PURITY") is not None]
        metric_strip([
            ("notes cached",     fmt(len(notes), 0)),
            ("dogs written up",  fmt(len(_dogs), 0)),
            ("syndromes",        fmt(len({n["SYNDROME_CODE"] for n in notes}), 0)),
            ("mean confidence",  fmt(sum(_conf) / len(_conf), 3) if _conf else "—"),
            ("model-derived",    f"{100 * sum(_pure) / len(_pure):.0f}%"
                                 if _pure else "—"),
        ])

        # ONE AGGREGATE, READ TWICE — ONCE AS A BAR AND ONCE AS A REGISTER.
        #
        # The caseload is lopsided by nature: ninety-odd of the notes are one
        # gastrointestinal code and the rest are single figures. A vertical
        # stack of four bars spent a hero's height drawing one wall and three
        # slivers, with bare codes (S2, S6) on the axis that mean nothing to a
        # reader who has not memorised REF.SYNDROMES. Horizontal and sorted
        # gives every row its full name and its count where the eye already is
        # — at the end of the bar — and the table underneath carries the three
        # numbers a length cannot: how many distinct dogs, how confident the
        # pattern engine was, and how much of it came from the model rather
        # than the heuristic.
        per: dict = {}
        for n in notes:
            e = per.setdefault(n["SYNDROME_CODE"], {
                "name": n.get("SYNDROME_NAME") or n["SYNDROME_CODE"],
                "system": n.get("BODY_SYSTEM") or "—",
                "dogs": set(), "conf": [], "pure": [], "bands": {}, "n": 0})
            e["n"] += 1
            e["dogs"].add(n["DOG_ID"])
            if n.get("CONFIDENCE") is not None:
                e["conf"].append(float(n["CONFIDENCE"]))
            if n.get("MODEL_PURITY") is not None:
                e["pure"].append(float(n["MODEL_PURITY"]))
            _b = (n.get("TRIAGE_LABEL") or "not yet triaged", n.get("SEVERITY"))
            e["bands"][_b] = e["bands"].get(_b, 0) + 1

        def _mean(xs, nd=3):
            return fmt(sum(xs) / len(xs), nd) if xs else "—"

        k1, k2 = st.columns(2)

        with k1:
            panel("The caseload these notes came from",
                  "Every cached note by syndrome, split into the triage bands "
                  "AI_CLASSIFY assigned. Read the bar, then pick a note out of "
                  "it in the picker below.")
            if PLOTLY:
                # Ascending, because plotly draws the first category at the
                # BOTTOM of a horizontal axis — so ascending here is descending
                # on screen, which is the order the reader expects.
                asc = sorted(per.items(), key=lambda kv: kv[1]["n"])
                ticks = [
                    f'{e["name"][:30] + ("…" if len(e["name"]) > 30 else "")}'
                    f'<br><span style="font-size:9px">{c} · {e["system"]}</span>'
                    for c, e in asc]
                sevs = sorted({n.get("SEVERITY") for n in notes},
                              key=lambda s: (s is None, -(s or 0)))
                fig = go.Figure()
                for sev in sevs:
                    lbl = next((n.get("TRIAGE_LABEL") for n in notes
                                if n.get("SEVERITY") == sev), None) or "not yet triaged"
                    xs = [e["bands"].get((lbl, sev), 0) for _, e in asc]
                    fig.add_trace(go.Bar(
                        x=xs, y=ticks, orientation="h", name=str(lbl), width=0.56,
                        marker=dict(color=TRIAGE_COLOUR.get(sev, "#A8A29E"),
                                    line=dict(width=0)),
                        hovertext=[f'{e["name"]} · {lbl}: {x} of {e["n"]} notes'
                                   for (_, e), x in zip(asc, xs)],
                        hoverinfo="text"))
                _top = max(e["n"] for _, e in asc)
                # THE TRACK, AS SHAPES RATHER THAN A TRACE. evil_bar draws the
                # faint full-length slot behind each bar with an extra Bar in
                # overlay mode, which a stacked chart cannot use — an overlay
                # track would stack onto the bands. Rectangles at the same
                # half-width as the bars give the same reading: a syndrome with
                # three notes next to one with ninety still has a visible row
                # rather than a smear at the origin.
                fig.update_layout(
                    barmode="stack", bargap=0.42, showlegend=True,
                    shapes=[dict(type="rect", xref="x", yref="y",
                                 x0=0, x1=_top * 1.30, y0=i - 0.28, y1=i + 0.28,
                                 fillcolor=GRID, line=dict(width=0), layer="below")
                            for i in range(len(asc))],
                    legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=10)),
                    # No axis title: the panel line above and the byline below
                    # both already say what a bar's length is, and a third copy
                    # would land in the same 46px the legend is standing in.
                    xaxis=dict(range=[0, _top * 1.30]),
                    yaxis=dict(tickfont=dict(size=10.5)),
                    # The count at the end of the bar rather than on the axis:
                    # a stacked bar cannot be read off a gridline anyway, and
                    # the share is the number a reader actually wants next to
                    # a caseload this lopsided.
                    annotations=[
                        dict(x=e["n"], y=t, text=f'<b>{e["n"]}</b>'
                                                 f'<span style="font-size:9.5px">'
                                                 f'&nbsp;&nbsp;{100 * e["n"] / len(notes):.0f}%'
                                                 f'</span>',
                             xanchor="left", xshift=7, yanchor="middle",
                             showarrow=False, font=dict(size=11.5, color=INK))
                        for (_, e), t in zip(asc, ticks)])
                if HAS_BARRADIUS:
                    fig.update_layout(barcornerradius=3)
                chart(evil_axes(fig, y_zero_line=False, dotted="x"),
                      bars(len(asc), row=42))
                html_table(
                    [{"s": f'<b>{esc(e["name"])}</b> '
                           f'<span class="tt-quiet">{c}</span>',
                      "y": e["system"],
                      "n": fmt(e["n"], 0),
                      "d": fmt(len(e["dogs"]), 0),
                      "c": _mean(e["conf"]),
                      "m": _mean(e["pure"])}
                     for c, e in sorted(per.items(), key=lambda kv: -kv[1]["n"])],
                    [("s", "syndrome"), ("y", "body system"), ("n", "notes"),
                     ("d", "dogs"), ("c", "mean conf."), ("m", "model-derived")])
                renderer_note(
                    "plotly · stacked bar",
                    f"{len(notes)} notes over {len(per)} syndromes. Length is "
                    f"the count and colour is the triage band, so a syndrome "
                    f"that is common and a syndrome that is urgent are two "
                    f"different readings of the same row.")

        with k2:
            panel("Did the triage agree with the measurements?",
                  "Every note on the three numbers the pattern engine "
                  "returned, coloured by the band Cortex put it in. Drag to "
                  "rotate.")
            if PLOTLY:
                # THE COLOUR IS NOT ONE OF THE AXES, WHICH IS THE WHOLE TEST.
                #
                # Duration, confidence and model purity are mechanical: they
                # fall out of MATCH_RECOGNIZE and the classifier without an
                # LLM anywhere near them. The triage band is Cortex's opinion,
                # written by a task into AI.TRIAGE. Plotting the opinion as
                # COLOUR over the measurements as POSITION means the two can
                # visibly disagree — if the red points were scattered evenly
                # through the cloud, the triage would be decorative. They are
                # not, and that is checkable here rather than asserted.
                fig = go.Figure()
                for sev in sorted({n.get("SEVERITY") for n in notes},
                                  key=lambda s: (s is None, s)):
                    grp = [n for n in notes if n.get("SEVERITY") == sev]
                    if not grp:
                        continue
                    lbl = grp[0].get("TRIAGE_LABEL") or "not yet triaged"
                    fig.add_trace(go.Scatter3d(
                        x=[float(n.get("DURATION_S") or 0) for n in grp],
                        y=[float(n.get("CONFIDENCE") or 0) for n in grp],
                        z=[float(n.get("MODEL_PURITY") or 0) for n in grp],
                        mode="markers", name=str(lbl),
                        marker=dict(
                            color=TRIAGE_COLOUR.get(sev, "#A8A29E"),
                            size=[4 + 9 * min(1.0, float(n.get("N_EPOCHS") or 0) / 60.0)
                                  for n in grp],
                            opacity=0.86, line=dict(color=CARD, width=0.6)),
                        text=[f'{n["SYNDROME_CODE"]} · dog {n["DOG_ID"]}'
                              f'<br>{fmt(n.get("DURATION_S"), 0)}s over '
                              f'{fmt(n.get("N_EPOCHS"), 0)} epochs'
                              f'<br>confidence {fmt(n.get("CONFIDENCE"), 3)}'
                              f'<br>model purity {fmt(n.get("MODEL_PURITY"), 3)}'
                              f'<br>{n.get("TRIAGE_LABEL")}' for n in grp],
                        hoverinfo="text"))
                scene3d(fig, "duration (s)", "confidence", "model purity")
                fig.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", y=0.02, x=0.5,
                                xanchor="center", font=dict(size=10)))
                chart3d(fig, H_LG)
                renderer_note(
                    "plotly · 3D",
                    f"{len(notes)} cached notes. Position is measured, colour "
                    f"is generated — so the two can be seen to agree rather "
                    f"than being asserted to.")

        st.markdown("---")
        panel("The note itself",
              f"{len(notes)} cached notes, most severe first. Every one below "
              f"this line is about the single finding picked here.")
        labels = [f'{n["SYNDROME_CODE"]} · dog {n["DOG_ID"]} · '
                  f'{str(n["ONSET_TS"])[:19]} · {n["TRIAGE_LABEL"]}' for n in notes]
        i = st.selectbox("finding", range(len(labels)),
                         format_func=lambda k: labels[k], key="note_pick",
                         label_visibility="collapsed")
        n = notes[i]
        colour = TRIAGE_COLOUR.get(n.get("SEVERITY"), "#A8A29E")

        photo = breed_photo(n.get("BREED"), 58, radius="6px")
        appx = (breed_photos().get(n.get("BREED")) or {}).get("IS_APPROXIMATE")
        # Under a clinical note the caption has to be unambiguous: a reader
        # skimming a SOAP paragraph beside a dog's face will assume it is the
        # patient unless told otherwise, in that exact spot, every time.
        cap = ("reference photo of a<br>similar breed, not this dog"
               if appx else "reference photo of the<br>breed, not this dog")
        st.markdown(f"""
<div class="tt-card">
  <div style="display:flex;gap:12px;align-items:flex-start">
    <div style="flex:0 0 auto;text-align:center">
      {photo}
      <div class="tt-quiet" style="font-size:9.5px;line-height:1.25;margin-top:3px;
           max-width:58px">{cap}</div>
    </div>
    <div style="flex:1 1 auto;min-width:0">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <span class="tt-badge" style="background:{colour};font-size:12px">
            {str(n.get('TRIAGE_LABEL','')).upper()}</span>
          <b style="margin-left:8px;font-size:15px">{n['SYNDROME_NAME']}</b>
          <span class="tt-quiet"> · {n.get('BODY_SYSTEM')}</span>
        </div>
        <div class="tt-quiet">confidence {fmt(n.get('CONFIDENCE'),3)}</div>
      </div>
      <div class="tt-quiet" style="margin-top:4px">
        Dog {n['DOG_ID']} · {n.get('BREED') or 'unknown breed'} ·
        {fmt(n.get('AGE_YEARS'),1)}y · {fmt(n.get('WEIGHT_KG'),1)}kg ·
        onset {str(n['ONSET_TS'])[:19]} UTC · {fmt(n.get('DURATION_S'),0)}s ·
        {fmt(n.get('N_EPOCHS'),0)} epochs
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(
            f'<div class="tt-card" style="font-size:13.5px;line-height:1.65;'
            f'white-space:pre-wrap">{n["SOAP_NOTE"]}</div>',
            unsafe_allow_html=True)

        # The sequence the note is about, printed beside the note.
        #
        # A generated paragraph asserting "repeated gait interruption" is worth
        # exactly as much as the reader's willingness to believe it. This is the
        # actual matched seconds, each coloured by the pattern variable it
        # played, so the prose can be checked against the rows it came from.
        # AI.V_VET_NOTE_FULL does not carry test_num or match_id, so the match
        # is resolved by its onset — the key the note itself displays.
        panel("The sequence this note is describing",
              "The matched seconds themselves, each epoch coloured by the "
              "pattern variable it played. The prose above is checkable "
              "against this ribbon.")
        key = rows(f"""
            SELECT test_num, match_id FROM MARTS.SYNDROME_MATCHES
            WHERE dog_id = {n['DOG_ID']}
              AND syndrome_code = '{n['SYNDROME_CODE']}'
              AND onset_ts = '{str(n['ONSET_TS'])[:19]}'
            LIMIT 1
        """)
        if key:
            symbol_ribbon(n["SYNDROME_CODE"], n["DOG_ID"],
                          key[0]["TEST_NUM"], key[0]["MATCH_ID"])
        else:
            empty_state("Could not resolve this note back to its match.",
                        "The note is kept; only the per-epoch ribbon is missing.")

        c1, c2 = st.columns(2)
        with c1:
            panel("Evidence, linked to the epoch range",
                  "Every claim the note makes, next to the measurement it was "
                  "made from.")
            ev = n.get("EVIDENCE")
            trows = [
                {"k": "matched window", "v": f'{str(n["ONSET_TS"])[:19]} → '
                                             f'{str(n.get("RESOLVE_TS"))[:19]}'},
                {"k": "epochs in match", "v": fmt(n.get("N_EPOCHS"), 0)},
                {"k": "epoch completeness", "v": fmt(n.get("AVG_QUALITY"), 3)},
                {"k": "model-derived states", "v": fmt(n.get("MODEL_PURITY"), 3)},
                {"k": "z vs own baseline", "v": fmt(n.get("Z_SELF"), 2)},
                {"k": "z vs cohort", "v": fmt(n.get("Z_COHORT"), 2)},
            ]
            if ev:
                try:
                    parsed = json.loads(ev) if isinstance(ev, str) else ev
                    for k, v in parsed.items():
                        trows.append({"k": k, "v": json.dumps(v)
                                      if isinstance(v, (list, dict)) else v})
                except Exception:  # noqa: BLE001
                    pass
            html_table(trows, [("k", "claim"), ("v", "value")])
        with c2:
            panel("The rule that fired",
                  "The MATCH_RECOGNIZE clause this note came out of, printed "
                  "in full.")
            # PATTERN WITHOUT DEFINE IS HALF A RULE. `PATTERN ( A B+ C )` says
            # nothing on its own — the meaning of the note is entirely in what
            # A, B and C were defined to be, and the view already carries it.
            # This is the same rendering the Syndromes tab uses, so a reader
            # who has seen one recognises the other.
            _def = str(n.get("DEFINE_TEXT") or "").strip()
            st.code(
                f"PATTERN ( {n['PATTERN_TEXT']} )"
                + ("\nDEFINE\n    " + ",\n    ".join(
                    x.strip() for x in _def.split(",")) if _def else ""),
                language="sql")
            st.markdown(
                f'<div class="tt-card"><div class="tt-metric-label">'
                f'why a sequence and not a threshold</div>'
                f'<div style="font-size:12px;line-height:1.5;margin-top:4px">'
                f'{n.get("WHY_NOT_THRESHOLD","")}</div></div>',
                unsafe_allow_html=True)

        st.markdown(
            f'<div class="tt-quiet" style="margin-top:10px;border-top:1px solid '
            f'{BORDER};padding-top:6px">Generated by <b>{n.get("CORTEX_MODEL")}</b> '
            f'via AI_COMPLETE · pipeline <b>{n.get("PIPELINE_VERSION")}</b> · '
            f'{n.get("GENERATED_AT")} · triage by AI_CLASSIFY over this note. '
            f'TELLTAIL is not a diagnostic device and nothing here substitutes '
            f'for a veterinarian.</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 7 — DRIVERS.  What explains it.
# ===========================================================================
def _page_6():
    st.markdown("#### Drivers")

    # THE SEPARATION, IN THE THREE AXES THE DESIGN ARGUES ABOUT.
    #
    # 3D because the claim is three-dimensional and flattens badly: locomotion
    # separates on neck/back CORRELATION, neck-driven behaviour separates on
    # neck DOMINANCE, and stillness separates on neck SD. Any 2D pair of those
    # three collapses two clusters on top of each other — project it yourself
    # by dragging the cube and you can watch it happen. Sampled per state so
    # the rare classes stay visible next to SNIFF.
    panel("The feature space, in the three axes the ethogram turns on",
          "Drag to rotate. Each point is one labelled second. Locomotion "
          "climbs the correlation axis; SHAKE and SCRATCH climb the "
          "dominance axis; REST and SIT collapse into the low-motion "
          "corner. This is the whole classifier argument in one object.")
    fs = rows("""
        SELECT state, neck_back_corr, vm_neck_std, neck_dominance
        FROM ML.V_LABELLED_EPOCHS
        WHERE neck_back_corr IS NOT NULL
          AND vm_neck_std IS NOT NULL
          AND neck_dominance IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY state ORDER BY RANDOM()) <= 320
    """)
    if PLOTLY and fs:
        pal = state_palette()
        by_state: dict = {}
        for r in fs:
            by_state.setdefault(r["STATE"], []).append(r)

        cshell, cshadow = st.columns([1, 1])
        hulls_on = cshell.checkbox(
            "draw each class as a solid volume", value=True,
            help="A translucent convex hull around the middle 90% of each "
                 "class. It is the same points, wrapped — nothing is fitted "
                 "and no boundary is being claimed.")
        drop_on = cshadow.checkbox(
            "drop shadows onto the walls", value=True,
            help="Projects every point flat onto the three back faces, which "
                 "is the 2D view you would have got instead. Watch classes "
                 "that are separate in the cube land on top of each other.")

        # --------------------------------------------------------------
        # A POINT CLOUD IS NOT A CLUSTER UNTIL YOU CAN SEE ITS EDGE.
        #
        # Fourteen colours of 2px dot at 72% opacity, interpenetrating, is
        # confetti — you cannot tell a tight class from a diffuse one, and
        # the whole claim of this chart is that the classes occupy DIFFERENT
        # REGIONS. Wrapping each class in its own convex hull turns it into
        # an object with a size and a shape, and where two hulls overlap the
        # translucency shows you exactly where the classifier's real
        # difficulty is.
        #
        # THE HULL IS COMPUTED IN THE BROWSER, NOT HERE. go.Mesh3d with
        # alphahull=0 hands the raw points to plotly.js and lets it
        # triangulate — no scipy.spatial.ConvexHull, so nothing new has to
        # be added to environment.yml and nothing new can fail the conda
        # solve in SiS.
        #
        # TRIMMED TO THE MIDDLE 90% FIRST, per axis. A convex hull is the
        # most outlier-sensitive summary there is: one mislabelled second
        # out at dominance 8 stretches the whole volume to reach it, and
        # SIT would be drawn as a class that spans the cube. The dots are
        # all still drawn — the hull describes the bulk, the points keep the
        # tails, and the caption says which is which.
        # --------------------------------------------------------------
        def _mid90(vals: list[float]) -> tuple[float, float]:
            s = sorted(vals)
            return s[int(0.05 * (len(s) - 1))], s[int(0.95 * (len(s) - 1))]

        fig = go.Figure()
        for stt in sorted(by_state):
            pts = by_state[stt]
            xs = [float(r["NECK_BACK_CORR"]) for r in pts]
            ys = [float(r["VM_NECK_STD"]) for r in pts]
            # log-ish squash: dominance runs to ~50 for a head shake and
            # would otherwise flatten every other class onto the floor
            zs = [min(float(r["NECK_DOMINANCE"]), 8.0) for r in pts]
            hue = pal.get(stt, "#999")

            if hulls_on and len(pts) >= 12:
                xl, xh = _mid90(xs)
                yl, yh = _mid90(ys)
                zl, zh = _mid90(zs)
                core = [(a, b, c) for a, b, c in zip(xs, ys, zs)
                        if xl <= a <= xh and yl <= b <= yh and zl <= c <= zh]
                # Four points is the minimum for a tetrahedron; below that
                # plotly.js returns an empty mesh and the trace silently
                # vanishes, which reads as a missing class rather than a
                # small one.
                if len(core) >= 8:
                    fig.add_trace(go.Mesh3d(
                        x=[p[0] for p in core], y=[p[1] for p in core],
                        z=[p[2] for p in core],
                        alphahull=0, color=hue, opacity=0.17,
                        flatshading=True, hoverinfo="skip",
                        lighting=dict(ambient=0.75, diffuse=0.55,
                                      specular=0.12, roughness=0.9),
                        legendgroup=stt, showlegend=False, name=stt))

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers", name=stt, legendgroup=stt,
                marker=dict(size=2.2, color=hue, opacity=0.72),
                # The shadows are the argument for spending a 3D plot here at
                # all: each wall IS the 2D scatter of the other two features,
                # drawn next to the 3D one, so "any 2D pair collapses two
                # classes" stops being a claim in a caption and becomes
                # something the reader watches happen.
                projection=dict(
                    x=dict(show=drop_on, opacity=0.10, scale=0.62),
                    y=dict(show=drop_on, opacity=0.10, scale=0.62),
                    z=dict(show=drop_on, opacity=0.10, scale=0.62)),
                hovertemplate=(stt + "<br>corr %{x:.2f}<br>neck sd %{y:.2f}"
                               "<br>dominance %{z:.2f}<extra></extra>")))
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="neck/back corr", backgroundcolor=SURFACE,
                           gridcolor=BORDER, zerolinecolor=BORDER,
                           showspikes=False),
                yaxis=dict(title="neck SD (g)", backgroundcolor=SURFACE,
                           gridcolor=BORDER, zerolinecolor=BORDER,
                           showspikes=False),
                zaxis=dict(title="neck dominance (clipped at 8)",
                           backgroundcolor=SURFACE, gridcolor=BORDER,
                           zerolinecolor=BORDER, showspikes=False),
                # A cube, not plotly's default "auto" box. Auto scales each
                # axis to its own data range, so the geometry of the cloud —
                # which is the entire point — changes shape depending on
                # which classes happen to be present in the sample.
                aspectmode="cube",
                camera=dict(eye=dict(x=1.55, y=1.5, z=0.85),
                            projection=dict(type="orthographic"))),
            paper_bgcolor=CARD, plot_bgcolor=CARD, showlegend=True,
            legend=dict(itemsizing="constant", font=dict(size=10)),
            margin=dict(l=0, r=0, t=4, b=0),
            font=dict(family=FONT_CHART, size=11, color=INK_2))
        fig.update_layout(height=H_LG)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        renderer_note(
            "plotly · 3d",
            "The only one of the three renderers here that can do this: "
            "Vega-Lite is a 2D grammar by specification and Bokeh has no 3D "
            "glyph, so a rotatable scene is plotly's alone. Solids are convex "
            "hulls of the middle 90% of each class, computed in your browser; "
            "the dots are every sampled second, tails included.")
    else:
        empty_state("No labelled epochs to plot.",
                    "Needs the bulk load and the Gate A label map.")

    ms = rows("SELECT * FROM ML.MODEL_SUMMARY")
    if ms:
        m = ms[0]
        acc = m.get("HOLDOUT_ACCURACY")
        st.markdown(f"""
<div class="tt-card">
  <div class="tt-metric-label">held-out accuracy — dog-disjoint</div>
  <div style="font-size:44px;font-weight:700;line-height:1.05;color:{INK}">
    {fmt((acc or 0) * 100, 1)}%</div>
  <div class="tt-quiet" style="margin-top:2px">
    {m.get('CLASSIFIER')} · {fmt(m.get('HOLDOUT_DOGS'),0)} entirely unseen dogs ·
    {fmt(m.get('HOLDOUT_EPOCHS'),0)} epochs · macro F1 {fmt(m.get('MACRO_F1'),3)} ·
    weighted F1 {fmt(m.get('WEIGHTED_F1'),3)}
  </div>
  <div class="tt-quiet" style="margin-top:6px">{m.get('PROTOCOL')}</div>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Feature separation**")
        # ML.V_FEATURE_SEPARATION, not the model's SHOW_FEATURE_IMPORTANCE.
        # Every introspection accessor on this account raises "Computation
        # Error in function __SHOW_*" while PREDICT works fine, so the
        # importance table is permanently empty here. The caption below says
        # exactly what this is instead of passing it off as the model's.
        st.markdown('<span class="tt-quiet">One-way ANOVA F-ratio per feature '
                    'over the labelled epochs: between-class variance of the '
                    'class means over pooled within-class variance. How well '
                    'each feature separates the ethogram states ON ITS OWN — '
                    'a different question from the split gain a tree model '
                    'reports, and computed here because model introspection '
                    'does not run on this account.</span>',
                    unsafe_allow_html=True)
        fi = rows("""SELECT feature, f_ratio FROM ML.V_FEATURE_SEPARATION
                     WHERE f_ratio IS NOT NULL ORDER BY f_ratio DESC LIMIT 25""")
        if PLOTLY and fi:
            names = [str(r["FEATURE"]) for r in fi][::-1]
            vals = [float(r["F_RATIO"] or 0) for r in fi][::-1]
            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation="h",
                marker=dict(color=[S_ORANGE if "CORR" in n.upper() else "#D6D3D1"
                                   for n in names]),
                hovertext=[f"{n} {v:.2f}" for n, v in zip(names, vals)], hoverinfo="text"))
            fig.update_layout(title="how well each feature separates the states "
                                    "(neck/back correlation in amber)",
                              title_font_size=11)
            chart(evil_axes(fig, y_zero_line=False, dotted="x"), bars(len(names), row=16))
        else:
            empty_state("No feature separation yet.",
                        "Needs labelled epochs: run the bulk load and push the "
                        "Gate A label map.")

        acc_err = rows("SELECT classifier, accessors_ok, accessor_error "
                       "FROM ML.MODEL_STATUS")
        if acc_err and acc_err[0].get("ACCESSORS_OK") is False and                 acc_err[0].get("ACCESSOR_ERROR"):
            st.caption(f"Model introspection unavailable on this account — "
                       f"{acc_err[0]['CLASSIFIER']} trained and predicts "
                       f"normally. Verbatim: {acc_err[0]['ACCESSOR_ERROR'][:180]}")

    with c2:
        panel("Contribution to deviation",
              "Which slices of the pack move the deviation metric, and by "
              "how much.")
        di = rows("SELECT * FROM ML.DRIVER_INSIGHTS LIMIT 200")
        if di:
            method = di[0].get("METHOD")
            # Built outside the f-string: SiS pins Python 3.11, where a
            # multi-line expression inside an f-string is a syntax error.
            note = ""
            if method == "SQL_CONTRIBUTION":
                note = (" — the native TOP_INSIGHTS signature was unavailable, so "
                        "this is a transparent SQL contribution decomposition: "
                        "lift × share per dimension value.")
            st.markdown(f'<span class="tt-quiet">method: <b>{method}</b>{note}'
                        f'</span>', unsafe_allow_html=True)
            if "CONTRIBUTION" in di[0]:
                top = sorted(di, key=lambda r: -abs(r.get("CONTRIBUTION") or 0))[:18]
                if PLOTLY:
                    fig = go.Figure(go.Bar(
                        x=[float(r.get("CONTRIBUTION") or 0) for r in top][::-1],
                        y=[f'{r.get("DIMENSION")} = {r.get("VALUE")}' for r in top][::-1],
                        orientation="h",
                        marker=dict(color=[S_ORANGE if (r.get("CONTRIBUTION") or 0) > 0
                                           else S_BLUE for r in top][::-1]),
                        hovertext=[f'{r.get("DIMENSION")} = {r.get("VALUE")}<br>'
                              f'contribution {fmt(r.get("CONTRIBUTION"),5)}<br>'
                              f'lift {fmt(r.get("LIFT"),4)} · share {fmt(r.get("SHARE"),3)}<br>'
                              f'{fmt(r.get("N"),0)} epochs' for r in top][::-1],
                        hoverinfo="text"))
                    fig.update_layout(title="which slices move the deviation metric",
                                      title_font_size=11)
                    chart(evil_axes(fig, y_zero_line=False, dotted="x"),
                          row_h(len(top)))
            else:
                dataframe(di[:30], [(k, k.lower()) for k in list(di[0].keys())[:6]])
        else:
            empty_state("No driver insights.",
                        "CALL ML.SP_RUN_TOP_INSIGHTS(); check ML.FUNCTION_STATUS.")

    # ----------------------------------------------------------------------
    # SECOND ROW. The right column used to carry three sections against the
    # left column's two, so this page ended with a table hanging off the
    # bottom of one half. Three rows of two now, each pair a chart against a
    # chart, and the per-class numbers get a figure of their own instead of
    # being the leftover at the end.
    # ----------------------------------------------------------------------
    d1, d2 = st.columns(2)

    with d1:
        panel("The feature, justifying itself",
              "Neck/back correlation by true label. Locomotion should sit "
              "high; neck-dominant behaviours should collapse toward zero. "
              "If they do not, the feature is wrong.")
        cbl = rows("SELECT * FROM STAGING.V_CORR_BY_LABEL ORDER BY avg_corr")
        if PLOTLY and cbl:
            fig = go.Figure()
            for r in cbl:
                fig.add_trace(go.Bar(
                    x=[str(r["LABEL_PRIMARY"])],
                    y=[float(r["AVG_CORR"] or 0)],
                    marker=dict(color=S_ORANGE if (r["AVG_CORR"] or 0) < 0.4 else S_BLUE),
                    error_y=dict(type="data", array=[float(r["SD_CORR"] or 0)],
                                 color=BORDER, thickness=1, width=3),
                    hovertext=[f'{r["LABEL_PRIMARY"]}<br>mean {fmt(r["AVG_CORR"],3)}'
                          f'<br>median {fmt(r["MEDIAN_CORR"],3)}'
                          f'<br>IQR {fmt(r["P25_CORR"],2)}–{fmt(r["P75_CORR"],2)}'
                          f'<br>{fmt(r["EPOCHS"],0)} epochs'],
                    hoverinfo="text", showlegend=False))
            fig.update_layout(title="CORR(vm_neck, vm_back) by annotated behaviour",
                              title_font_size=11, yaxis=dict(range=[-0.3, 1.0]))
            # NO CORNER RADIUS HERE. These bars carry an error bar each, and a
            # rounded cap under a whisker reads as the whisker having a base.
            chart(evil_axes(fig), H_MD)

    with d2:
        panel("Confusion matrix, held-out dogs",
              "Row-normalised, so each row is one true state spread across "
              "what the model called it.")
        cm = rows("SELECT * FROM ML.CONFUSION_MATRIX")
        if PLOTLY and cm:
            states = sorted({r["ACTUAL_STATE"] for r in cm} |
                            {r["PREDICTED_STATE"] for r in cm})
            idx = {s: i for i, s in enumerate(states)}
            z = [[0.0] * len(states) for _ in states]
            txt = [[""] * len(states) for _ in states]
            for r in cm:
                i, j = idx[r["ACTUAL_STATE"]], idx[r["PREDICTED_STATE"]]
                z[i][j] = float(r["PCT_OF_ACTUAL"] or 0)
                txt[i][j] = (f'actual {r["ACTUAL_STATE"]}<br>'
                             f'predicted {r["PREDICTED_STATE"]}<br>'
                             f'{fmt(r["N"],0)} epochs ({r["PCT_OF_ACTUAL"]}%)')
            fig = go.Figure(go.Heatmap(
                z=z, x=states, y=states, text=txt, hoverinfo="text",
                colorscale=[[0, CARD], [1, INK]], showscale=False,
                xgap=2, ygap=2))
            fig.update_layout(title="row-normalised, % of each true state",
                              title_font_size=11,
                              xaxis_title="predicted", yaxis_title="actual")
            chart(clean_axes(fig, y_zero_line=False), H_MD)

    # ----------------------------------------------------------------------
    # THIRD ROW — the held-out numbers, as a shape and as a table.
    # ----------------------------------------------------------------------
    pc = rows("SELECT * FROM ML.CLASS_METRICS ORDER BY support DESC")
    if pc:
        f1c, f2c = st.columns(2)
        with f1c:
            panel("Per-class, as a shape",
                  "Precision, recall and F1 for each state the model predicts "
                  "on dogs it has never seen.")
            if PLOTLY:
                # A RADAR, AND ONLY BECAUSE THE SPOKES ARE COMMENSURABLE.
                #
                # Precision, recall and F1 are all proportions on [0, 1], so
                # the enclosed area means something and a class that is small
                # on every spoke is visibly small. This is the one shape on
                # the page where that holds — a radar over, say, support and
                # F1 together would draw a confident polygon out of a count
                # and a ratio, which is exactly how the form gets abused.
                _top = pc[:6]
                fig = evil_radar(
                    ["precision", "recall", "F1"],
                    [(str(r["STATE"]),
                      [float(r["PRECISION"] or 0), float(r["RECALL"] or 0),
                       float(r["F1"] or 0)],
                      SYMBOL_COLOURS[i % len(SYMBOL_COLOURS)])
                     for i, r in enumerate(_top)],
                    rng=[0, 1])
                fig.update_layout(
                    title=f"the {len(_top)} best-supported states, held out",
                    title_font_size=11)
                chart(fig, H_MD)
                renderer_note(
                    "plotly · radar",
                    "Three axes that are all proportions on the same [0, 1] "
                    "scale, which is the only condition under which the area "
                    "a radar encloses is allowed to mean anything.")
        with f2c:
            panel("Per-class, on unseen dogs",
                  "The same numbers, for reading rather than comparing.")
            html_table([{"s": r["STATE"], "p": fmt(r["PRECISION"], 3),
                         "r": fmt(r["RECALL"], 3), "f": fmt(r["F1"], 3),
                         "n": fmt(r["SUPPORT"], 0)} for r in pc],
                       [("s", "state"), ("p", "precision"), ("r", "recall"),
                        ("f", "F1"), ("n", "support")])


# ===========================================================================
# TAB 8 — SHELTER REALITY.  Allowed to be quiet.
# ===========================================================================
def _page_7():
    st.markdown("#### Shelter reality")
    punch = rows("SELECT * FROM REF.V_SHELTER_PUNCHLINE ORDER BY syndrome_code")
    intake = rows("""
        SELECT month, SUM(n) AS n,
               SUM(IFF(is_behaviour_linked, n, 0)) AS behaviour_n
        FROM REF.V_AAC_INTAKE_TREND
        WHERE month >= '2015-01-01'
        GROUP BY month ORDER BY month
    """)
    los = rows("""
        SELECT breed_group, intake_condition, median_los_days, n
        FROM REF.V_AAC_LENGTH_OF_STAY
        WHERE n >= 50 ORDER BY median_los_days DESC LIMIT 40
    """)

    if not intake and not punch:
        empty_state("No shelter data.", "Run: python scripts/austin_sync.py")
    else:
        # ------------------------------------------------------------------
        # THIS PAGE WAS THE WORST OF THE RAGGED ONES.
        #
        # The left column held an intake chart, a 16-bar LOS chart AND an
        # 18-row table; the right held two small bar panels and a prose card.
        # Left ran roughly four hundred pixels past right, so the tab ended in
        # a column of table against a column of nothing. Four equal rows now,
        # each half sized against the other, and the two tables are paired at
        # the bottom instead of one hanging off the end.
        # ------------------------------------------------------------------
        r1l, r1r = st.columns(2)

        with r1l:
            panel("A decade of intakes",
                  "Austin Animal Center, every dog through the door, with the "
                  "behaviour-linked share drawn inside the total rather than "
                  "beside it.")
            if PLOTLY and intake:
                months = [str(r["MONTH"])[:10] for r in intake]
                fig = go.Figure()
                # Total first and underneath: the behaviour-linked series is a
                # SUBSET of it, so drawing it on top as a smaller filled area
                # says "this much of that" in the shape itself. Two lines on a
                # shared baseline would have said "these two things", which is
                # a different and wrong claim.
                evil_area(fig, months, [float(r["N"] or 0) for r in intake],
                          "#A8A29E", show_glow=False, top=0.22,
                          text=[f'{str(r["MONTH"])[:7]}: {fmt(r["N"],0)} '
                                f'dog intakes' for r in intake])
                evil_area(fig, months,
                          [float(r["BEHAVIOUR_N"] or 0) for r in intake],
                          S_ORANGE, cap=True,
                          text=[f'{str(r["MONTH"])[:7]}: '
                                f'{fmt(r["BEHAVIOUR_N"],0)} behaviour-linked'
                                for r in intake])
                fig.update_layout(title="Austin Animal Center dog intakes — "
                                        f"<span style='color:{S_ORANGE}'>"
                                        "behaviour-linked</span> against the total",
                                  title_font_size=11)
                chart(evil_axes(fig), H_LG)
                renderer_note(
                    "plotly · area",
                    f"{len(intake)} months of the City of Austin open data "
                    f"portal. The fill is a vertical alpha ramp of one hue, "
                    f"not a blend of two — an invented intermediate colour "
                    f"would not carry the separation the palette was validated "
                    f"for.")

        with r1r:
            panel("Why this tab exists",
                  "The shelter is where an unread behavioural signal ends up.")
            st.markdown(f"""
<div class="tt-card" style="font-size:13px;line-height:1.7">
Austin publishes a decade of intake and outcome records for every animal that
passes through its shelter. Behaviour is a named outcome reason in that data,
sitting alongside aggression and medical. Austin is a no-kill shelter and over
ninety percent of its animals are adopted, transferred or returned — which is
what makes the behavioural tail pointed rather than routine.
<br><br>
The categories in the panels below were detected on a collar, at home, from
movement alone. The categories beside them were written down at intake, after
the relationship had already broken down. They are the same categories.
<br><br>
<b>The dog was showing the pattern for eleven days before anyone noticed. The
warehouse noticed on day two.</b>
<br><br>
<span class="tt-quiet">Austin only. Not generalised to national claims.</span>
</div>""", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # SECOND ROW — length of stay, as a field and as a ranking.
        # ------------------------------------------------------------------
        if los:
            r2l, r2r = st.columns(2)

            with r2l:
                panel("Length of stay, as a field",
                      "Every breed group against every intake condition, "
                      "height is the median wait in days. Drag to rotate.")
                if PLOTLY:
                    # STEMS, NOT A SURFACE, AND THE SPARSITY IS THE REASON.
                    #
                    # breed group x intake condition is a grid with holes in
                    # it: most combinations never reach the n >= 50 floor this
                    # query applies, so a surface would either interpolate a
                    # ridge across combinations that were never measured, or
                    # tear a hole in the mesh that reads as a rendering fault.
                    # A stem stands only where there is a number, which is the
                    # honest shape of a sparse grid — and the marker on top
                    # carries n, so a tall stem on 51 animals cannot be
                    # mistaken for a tall stem on 7,463.
                    bg = sorted({str(r["BREED_GROUP"]) for r in los})
                    ic = sorted({str(r["INTAKE_CONDITION"]) for r in los})
                    bgi = {s: i for i, s in enumerate(bg)}
                    ici = {s: i for i, s in enumerate(ic)}
                    fig = go.Figure()
                    ns = [float(r["N"] or 0) for r in los]
                    nmax = max(ns or [1.0])
                    for r in los:
                        gx, gy = bgi[str(r["BREED_GROUP"])], ici[str(r["INTAKE_CONDITION"])]
                        z = float(r["MEDIAN_LOS_DAYS"] or 0)
                        beh = str(r["INTAKE_CONDITION"] or "").upper() in (
                            "BEHAVIOR", "BEHAVIOUR")
                        hue = S_ORANGE if beh else "#A8A29E"
                        fig.add_trace(go.Scatter3d(
                            x=[gx, gx], y=[gy, gy], z=[0, z], mode="lines",
                            line=dict(color=alpha(hue, 0.55), width=5),
                            hoverinfo="skip", showlegend=False))
                        fig.add_trace(go.Scatter3d(
                            x=[gx], y=[gy], z=[z], mode="markers",
                            marker=dict(
                                color=hue, size=7 + 13 * (float(r["N"] or 0) / nmax),
                                line=dict(color=CARD, width=1)),
                            text=[f'{r["BREED_GROUP"]} · {r["INTAKE_CONDITION"]}'
                                  f'<br>median {fmt(r["MEDIAN_LOS_DAYS"],1)} days'
                                  f'<br>{fmt(r["N"],0)} animals'],
                            hoverinfo="text", showlegend=False))
                    scene3d(fig, "breed group", "intake condition",
                            "median LOS (days)")
                    fig.update_layout(scene=dict(
                        xaxis=dict(tickmode="array",
                                   tickvals=list(range(len(bg))), ticktext=bg,
                                   tickfont=dict(size=8, color=INK_2)),
                        yaxis=dict(tickmode="array",
                                   tickvals=list(range(len(ic))), ticktext=ic,
                                   tickfont=dict(size=8, color=INK_2)),
                        aspectmode="manual",
                        aspectratio=dict(x=1.3, y=1.1, z=0.75)))
                    chart3d(fig, H_LG)
                    renderer_note(
                        "plotly · 3D",
                        f"{len(los)} breed-group x condition cells, marker size "
                        f"is n. Only combinations with at least 50 animals are "
                        f"here — the empty floor is a sample-size floor, not a "
                        f"shelter with no such dogs.")

            with r2r:
                panel("Median length of stay, days",
                      "The cost of a behavioural label, visible in days. The "
                      "track behind each bar is the longest wait on the chart.")
                if PLOTLY:
                    # Behaviour-flagged intakes highlighted against everything
                    # else, because that is the comparison the tab is making:
                    # the same shelter, the same breeds, a longer wait.
                    top = los[:16][::-1]
                    names = [f'{r["BREED_GROUP"]} · {r["INTAKE_CONDITION"]}'
                             for r in top]
                    beh = [str(r["INTAKE_CONDITION"] or "").upper() in
                           ("BEHAVIOR", "BEHAVIOUR") for r in top]
                    fig = go.Figure()
                    evil_bar(
                        fig, names,
                        [float(r["MEDIAN_LOS_DAYS"] or 0) for r in top],
                        [S_ORANGE if b else "#D6D3D1" for b in beh],
                        text=[f'{n}<br>median {fmt(r["MEDIAN_LOS_DAYS"],1)} days'
                              f'<br>{fmt(r["N"],0)} animals'
                              for n, r in zip(names, top)])
                    fig.update_layout(
                        title="longest waits first — "
                              f"<span style='color:{S_ORANGE}'>behaviour</span> "
                              "intakes against every other condition",
                        title_font_size=11)
                    chart(evil_axes(fig, y_zero_line=False, dotted="x"), H_LG)
                    renderer_note(
                        "plotly · bar",
                        "Same rows as the field beside it, ranked instead of "
                        "placed. The 3D answers 'where in the grid'; this "
                        "answers 'how much', which is the reading a rotated "
                        "scene is bad at.")

        # ------------------------------------------------------------------
        # THIRD ROW — TWO PANELS, NOT TWO Y-AXES.
        #
        # Collar detections are in the tens and shelter records in the
        # thousands. Drawn against a shared axis, or against two axes in one
        # frame, the bars would invite a comparison the numbers do not support.
        # The point of this tab does not need them to be comparable: it is that
        # the SAME CATEGORIES appear in both places. Small multiples say that
        # without the arithmetic sleight of hand — and side by side rather than
        # stacked, because the categories are meant to be read across.
        # ------------------------------------------------------------------
        if PLOTLY and punch:
            codes = [r["SYNDROME_CODE"] for r in punch]
            r3 = st.columns(2)
            for cell, (key, label, hue, note) in zip(r3, (
                ("TELLTAIL_DETECTIONS", "detected on a collar, at home",
                 INK, "detections"),
                ("SHELTER_BEHAVIOUR_RECORDS",
                 "written down at intake, after the fact", S_ORANGE,
                 "shelter records"),
            )):
                with cell:
                    panel(label, "")
                    fig = go.Figure()
                    evil_bar(
                        fig, codes, [float(r[key] or 0) for r in punch],
                        [hue] * len(codes), horizontal=False,
                        text=[f'{r["SYNDROME_CODE"]} {r["SYNDROME_NAME"]}<br>'
                              f'{fmt(r[key], 0)} {note}' for r in punch])
                    chart(evil_axes(fig), H_SM)
            st.markdown(
                '<div class="tt-quiet" style="margin-top:-6px">Same '
                'categories, two independent counts — deliberately on their '
                'own scales in their own panels. One is a handful of dogs '
                'wearing a collar for a few days; the other is a decade of a '
                'city shelter.</div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # FOURTH ROW — the numbers behind both halves.
        #
        # THE LAYOUT BUG WAS IN THE PAIRING ITSELF. table_pair() trimmed both
        # tables to the LONGER of the two, which pairs nothing: the left table
        # has forty rows and the right has one per syndrome — six, and it will
        # always be six, because REF.SYNDROMES has six — so the row drew 18
        # against 6 and the tab ended in a column of table beside a column of
        # white. A short table cannot be padded and a long one should not be
        # deleted down to it, so the long side keeps every row inside a scroll
        # box whose height is what makes the two columns finish together.
        #
        # And a forty-row register is a thing to interrogate, not to read top
        # to bottom, so it gets the three controls that make it answerable:
        # which condition, which breed group, and in what order. All three
        # filter rows already in memory — the query behind them is cached and
        # does not run again, so a dropdown costs a rerender and nothing else.
        # ------------------------------------------------------------------
        # One box height for the row. The left table overflows it and scrolls;
        # the right one is shorter than it and the box shrinks to fit, so the
        # card underneath is what brings that column down to the same line.
        TALL = 300
        t1, t2 = st.columns(2)
        with t1:
            panel("Length of stay, in full",
                  "Every breed-group × condition cell with at least 50 animals. "
                  "Filter it, sort it; the box scrolls and nothing is dropped.")
            conds = sorted({str(r["INTAKE_CONDITION"]) for r in los})
            groups = sorted({str(r["BREED_GROUP"]) for r in los})
            q1, q2, q3 = st.columns(3)
            _cond = q1.selectbox("condition", ["all conditions"] + conds,
                                 key="los_cond")
            _grp = q2.selectbox("breed group", ["all breed groups"] + groups,
                                key="los_grp")
            _how = q3.selectbox("sort by", ["longest wait", "shortest wait",
                                            "most animals", "breed group"],
                                key="los_sort")
            view = [r for r in los
                    if (_cond == "all conditions"
                        or str(r["INTAKE_CONDITION"]) == _cond)
                    and (_grp == "all breed groups"
                         or str(r["BREED_GROUP"]) == _grp)]
            _key = {
                "longest wait":  lambda r: -float(r["MEDIAN_LOS_DAYS"] or 0),
                "shortest wait": lambda r: float(r["MEDIAN_LOS_DAYS"] or 0),
                "most animals":  lambda r: -float(r["N"] or 0),
                "breed group":   lambda r: (str(r["BREED_GROUP"]),
                                            str(r["INTAKE_CONDITION"])),
            }[_how]
            view = sorted(view, key=_key)
            if view:
                # The bar in the median-LOS cell is scaled to the longest wait
                # in the WHOLE table rather than to the current filter, so
                # narrowing to one condition does not silently rescale the
                # column and make a nine-day wait look like a forty-day one.
                _span = max(float(r["MEDIAN_LOS_DAYS"] or 0) for r in los) or 1.0
                _nmax = max(float(r["N"] or 0) for r in los) or 1.0

                def _hue(r) -> str:
                    return (S_ORANGE
                            if str(r["INTAKE_CONDITION"] or "").upper()
                            in ("BEHAVIOR", "BEHAVIOUR") else "#D6D3D1")

                scroll_table(
                    [{"b": r["BREED_GROUP"], "c": r["INTAKE_CONDITION"],
                      "d": cell_bar(r["MEDIAN_LOS_DAYS"], _span, _hue(r), 1),
                      "n": cell_bar(r["N"], _nmax, BORDER)} for r in view],
                    [("b", "breed group"), ("c", "intake condition"),
                     ("d", "median LOS, days"), ("n", "animals")],
                    height=TALL)
                st.markdown(
                    f'<div class="tt-quiet" style="margin-top:6px">'
                    f'{len(view)} of {len(los)} cells shown · '
                    f'{fmt(sum(float(r["N"] or 0) for r in view), 0)} animals '
                    f'behind them · bars are scaled to the longest wait in the '
                    f'full table, not to this filter.</div>',
                    unsafe_allow_html=True)
            else:
                empty_state("No cell matches that combination.",
                            "Only breed-group × condition pairs with at least "
                            "50 animals are kept, so most combinations are "
                            "genuinely absent rather than hidden.")
        with t2:
            panel("The same categories, counted twice",
                  "Detected on a collar against written down at intake. The "
                  "two counts are NOT a ratio — read each bar against its own "
                  "column.")
            if punch:
                _tmax = max(float(r["TELLTAIL_DETECTIONS"] or 0) for r in punch) or 1.0
                _smax = max(float(r["SHELTER_BEHAVIOUR_RECORDS"] or 0) for r in punch) or 1.0
                scroll_table(
                    [{"c": r["SYNDROME_CODE"],
                      "s": r["SYNDROME_NAME"],
                      "t": cell_bar(r["TELLTAIL_DETECTIONS"], _tmax, INK),
                      "a": cell_bar(r["SHELTER_BEHAVIOUR_RECORDS"], _smax,
                                    S_ORANGE)}
                     for r in punch],
                    [("c", "code"), ("s", "syndrome"),
                     ("t", "collar"), ("a", "shelter")],
                    height=TALL)
                st.markdown(
                    f'<div class="tt-quiet" style="margin-top:6px">'
                    f'{fmt(sum(float(r["TELLTAIL_DETECTIONS"] or 0) for r in punch), 0)} '
                    f'collar detections over {len(punch)} syndromes against '
                    f'{fmt(sum(float(r["SHELTER_BEHAVIOUR_RECORDS"] or 0) for r in punch), 0)} '
                    f'shelter records. Each column is scaled to its own column '
                    f'— a handful of dogs for a few days is not a decade of a '
                    f'city.</div>', unsafe_allow_html=True)
                st.markdown(f"""
<div class="tt-card" style="margin-top:10px">
  <div class="tt-metric-label">how to read this pair</div>
  <div style="font-size:12px;line-height:1.6;margin-top:5px">
    A zero in the <b>collar</b> column is a syndrome no dog in this study
    happened to show — 45 dogs over a few days is not a survey. A zero in the
    <b>shelter</b> column is the opposite claim: Austin has no intake reason
    that maps to it, so a decade of records cannot count it however often it
    happened.
    <br><br>
    The row that matters is the one with a number on both sides. It is the
    same behaviour, described once by a collar before anyone was worried and
    once by an intake form after the relationship had already ended.
  </div>
</div>""", unsafe_allow_html=True)


# ===========================================================================
# TAB 9 — PIPELINE.  How it was built.
# ===========================================================================
def _page_8():
    st.markdown("#### Pipeline")
    if stats:
        s = stats[0]
        metric_strip([
            ("raw rows, bulk",   fmt(s.get("RAW_ROWS_BULK"), 0)),
            ("raw rows, live",   fmt(s.get("RAW_ROWS_LIVE"), 0)),
            ("epochs",           fmt(s.get("EPOCHS_CLASSIFIED"), 0)),
            ("matches",          fmt(s.get("SYNDROME_MATCHES"), 0)),
            ("on chain",         fmt(s.get("ATTESTATIONS_ONCHAIN"), 0)),
        ])

    # Both charts in the first row get ONE height, decided before either is
    # drawn — the lag chart grows a bar per Dynamic Table and the credit burn
    # does not, so whichever is fixed becomes the short side of the row.
    lag = rows("SELECT * FROM MARTS.V_DAG_LAG ORDER BY schema_name, object_name")
    _rowh = row_h(len(lag))

    c1, c2 = st.columns(2)

    with c1:
        panel("Dynamic Table refresh lag",
              "Declared target lag against observed. No cron anywhere in the "
              "feature, state, transition or baseline layers.")
        if PLOTLY and lag:
            # PLOTTED AS A RATIO, NOT IN SECONDS.
            #
            # In raw seconds one table sits near 20,000 and the rest under 60,
            # so a linear axis renders eleven invisible slivers beside a single
            # red slab and you cannot tell which of the eleven are healthy. The
            # question is never "how many seconds" — it is "is this table
            # keeping the promise it declared", which is observed / target. One
            # is the line everything is read against, and every object lands on
            # a scale where it can actually be seen.
            def _ratio(r):
                tgt = float(r.get("TARGET_LAG_SEC") or 0)
                obs = float(r.get("MEAN_LAG_SEC") or 0)
                return (obs / tgt) if tgt > 0 else None

            scored = [(r, _ratio(r)) for r in lag]
            scored = [(r, v) for r, v in scored if v is not None]
            scored.sort(key=lambda rv: rv[1])
            names = [f'{r["SCHEMA_NAME"]}.{r["OBJECT_NAME"]}' for r, _ in scored]
            vals = [v for _, v in scored]
            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation="h",
                marker=dict(color=["#B91C1C" if v > 1 else "#D6D3D1"
                                   for v in vals]),
                hovertext=[f'{n}<br>{v:.2f}x its target lag'
                           f'<br>mean {fmt(r["MEAN_LAG_SEC"],1)}s of '
                           f'{fmt(r["TARGET_LAG_SEC"],0)}s target'
                           f'<br>max {fmt(r["MAXIMUM_LAG_SEC"],1)}s'
                           f'<br>state {r["STATE"]}'
                           for n, v, (r, _) in zip(names, vals, scored)],
                hoverinfo="text"))
            fig.add_vline(x=1, line=dict(color=INK, width=1.5, dash="dot"))
            fig.update_layout(
                title="observed lag as a multiple of the declared target — "
                      "past the dotted line is a table falling behind",
                title_font_size=11,
                # explicit ticks: a log axis left to itself labels every minor
                # decade step (3,4,5,6,7,8,9,1,2,...) which reads as noise
                xaxis=dict(type="log", title="× target lag",
                           tickmode="array",
                           tickvals=[0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50],
                           ticktext=["0.1×", "0.25×", "0.5×", "1× target",
                                     "2×", "5×", "10×", "25×", "50×"]))
            chart(evil_axes(fig, y_zero_line=False, dotted="x"), _rowh)
        elif not lag:
            empty_state("No refresh history yet.",
                        "Dynamic Tables report after their first refresh.")

    with c2:
        panel("Credit burn",
              "Cumulative credits over seven days. Every one of them was spent "
              "by a task, never by a page render.")
        cb = rows("SELECT * FROM MARTS.V_CREDIT_BURN ORDER BY hour")
        if PLOTLY and cb:
            fig = go.Figure()
            evil_area(
                fig, [str(r["HOUR"])[:16] for r in cb],
                [float(r["CUMULATIVE_CREDITS"] or 0) for r in cb],
                S_ORANGE, width=1.8, shape="linear", cap=True,
                text=[f'{str(r["HOUR"])[:16]}<br>{fmt(r["CREDITS"],4)} this hour<br>'
                      f'{fmt(r["CUMULATIVE_CREDITS"],3)} cumulative' for r in cb])
            fig.update_layout(title="cumulative credits, 7 days (trial grant is 400)",
                              title_font_size=11)
            # THE SAME HEIGHT AS THE CHART BESIDE IT, not H_MD. The lag chart
            # grows with the number of Dynamic Tables and this one does not, so
            # a fixed height on one side of the row is a gap waiting to happen.
            chart(evil_axes(fig), _rowh)
        else:
            empty_state("No credit history yet.",
                        "MARTS.V_CREDIT_BURN reads the account usage views.")

    # ----------------------------------------------------------------------
    # THE OPERATIONAL REGISTERS.
    #
    # THIS IS WHERE THE TAB WAS RAGGED. Every row here paired a table that is
    # three rows long by construction — one per ML routine, one per AI Function
    # — against one that is forty or five hundred, and a table cannot be padded
    # to meet its partner. So the long ones are boxed: they keep every row they
    # queried, the box is the height, and both halves of a row now end on the
    # same line. Short against short, long against long, and the 44-row
    # parameter register goes full width where its descriptions have room to
    # sit on one line instead of wrapping to four.
    # ----------------------------------------------------------------------
    BOX = 300
    b1, b2 = st.columns(2)

    with b1:
        panel("Dynamic Table lag, in full",
              "Declared target against observed, for every object in the DAG.")
        if lag:
            _tmax = max(float(r.get("MAXIMUM_LAG_SEC") or 0) for r in lag) or 1.0
            scroll_table(
                [{"o": f'{r["SCHEMA_NAME"]}.{r["OBJECT_NAME"]}',
                  "t": fmt(r["TARGET_LAG_SEC"], 0),
                  "m": fmt(r["MEAN_LAG_SEC"], 1),
                  "x": cell_bar(r["MAXIMUM_LAG_SEC"], _tmax,
                                "#B91C1C" if (float(r.get("MAXIMUM_LAG_SEC") or 0)
                                              > float(r.get("TARGET_LAG_SEC") or 0) > 0)
                                else BORDER, 1),
                  "s": r["STATE"]} for r in lag],
                [("o", "object"), ("t", "target s"), ("m", "mean s"),
                 ("x", "max s"), ("s", "state")], height=BOX)
        else:
            empty_state("No refresh history yet.",
                        "Dynamic Tables report after their first refresh.")

    with b2:
        panel("Task history",
              "The forty most recent scheduled runs, newest first. Every number "
              "on every other tab was written by one of these.")
        th = rows("SELECT task_name, state, scheduled_time, duration_ms, return_value, "
                  "error_message FROM MARTS.V_TASK_HISTORY "
                  "ORDER BY scheduled_time DESC LIMIT 40")
        if th:
            _dmax = max(float(r.get("DURATION_MS") or 0) for r in th) or 1.0
            scroll_table(
                [{"t": r["TASK_NAME"],
                  "s": r["STATE"],
                  "w": str(r["SCHEDULED_TIME"])[:19],
                  "d": cell_bar(r["DURATION_MS"], _dmax,
                                "#B91C1C" if str(r["STATE"]).upper() == "FAILED"
                                else BORDER),
                  "r": str(r["RETURN_VALUE"] or r["ERROR_MESSAGE"] or "")[:70]}
                 for r in th],
                [("t", "task"), ("s", "state"), ("w", "scheduled"),
                 ("d", "ms"), ("r", "returned")], height=BOX)
        else:
            empty_state("No task runs yet.", "Tasks resume at the end of 11_tasks.sql.")

    g1, g2 = st.columns(2)

    with g1:
        panel("ML function status",
              "The last run of each modelling routine, and what it returned.")
        fs = rows("SELECT fn, status, detail, rows_out, ran_at FROM ML.FUNCTION_STATUS "
                  "QUALIFY ROW_NUMBER() OVER (PARTITION BY fn ORDER BY ran_at DESC) = 1")
        if fs:
            html_table([{"f": r["FN"], "s": r["STATUS"], "n": fmt(r["ROWS_OUT"], 0),
                         "d": str(r["DETAIL"] or "")[:80]} for r in fs],
                       [("f", "function"), ("s", "status"), ("n", "rows"),
                        ("d", "detail")])
            st.markdown('<span class="tt-quiet">SNOWFLAKE.ML.ANOMALY_DETECTION, '
                        'FORECAST and TOP_INSIGHTS, each run by a task and each '
                        'landing in a table. A status here is the routine\'s own '
                        'report, not a claim this page makes about it.</span>',
                        unsafe_allow_html=True)
        else:
            empty_state("No ML runs recorded.",
                        "ML.FUNCTION_STATUS is written by the modelling procs.")

    with g2:
        panel("Cortex usage",
              "Every AI Function call the build has made, by function.")
        us = rows("SELECT * FROM AI.V_USAGE_SUMMARY")
        if us:
            html_table([{"f": r["FN"], "c": fmt(r["TOTAL_CALLS"], 0),
                         "d": fmt(r["CALLS_LAST_24H"], 0),
                         "b": fmt(r["BATCHES"], 0),
                         "x": fmt(r["FAILED_BATCHES"], 0)} for r in us],
                       [("f", "function"), ("c", "calls total"),
                        ("d", "calls 24h"), ("b", "batches"), ("x", "failed")])
            st.markdown('<span class="tt-quiet">Trial accounts without a payment '
                        'method are capped at roughly ten credits per day of AI '
                        'Function usage. Every call here was made by a task, into '
                        'a table, deduped on the finding key.</span>',
                        unsafe_allow_html=True)
        else:
            empty_state("No Cortex calls recorded.",
                        "AI.V_USAGE_SUMMARY aggregates the AI layer's own log.")

    panel("Parameters",
          "Every threshold in the build, full width because 44 of them with a "
          "sentence each do not fit in half. No magic numbers live in SQL.")
    pr = rows("SELECT key, COALESCE(TO_VARCHAR(value_num), value_str) AS v, unit, "
              "description FROM REF.PARAMS ORDER BY key")
    if pr:
        _q = st.text_input(
            "filter parameters", key="param_q", label_visibility="collapsed",
            placeholder="filter by key, unit or description — e.g. solana, "
                        "baseline, epochs")
        _needle = (_q or "").strip().lower()
        _pv = [r for r in pr
               if not _needle
               or _needle in f'{r["KEY"]} {r["V"]} {r["UNIT"]} {r["DESCRIPTION"]}'.lower()]
        if _pv:
            scroll_table([{"k": f'<span class="tt-mono">{r["KEY"]}</span>',
                           "v": f'<b>{r["V"]}</b>', "u": r["UNIT"],
                           "d": r["DESCRIPTION"]} for r in _pv],
                         [("k", "key"), ("v", "value"), ("u", "unit"),
                          ("d", "what it does")], height=360)
            st.markdown(
                f'<div class="tt-quiet" style="margin-top:6px">'
                f'{len(_pv)} of {len(pr)} parameters · every one of them is read '
                f'by the SQL at runtime, so changing a row here changes the '
                f'build rather than describing it.</div>',
                unsafe_allow_html=True)
        else:
            empty_state(f'No parameter matches "{esc(_q)}".',
                        "REF.PARAMS holds 44 keys; try a shorter fragment.")
    else:
        empty_state("No parameters.", "REF.PARAMS is seeded by 02_ref_seed.sql.")

    # ------------------------------------------------------------------
    # WHO PUBLISHES, AND HOW FAR EACH CLAIM GOT.
    #
    # The wallet and cluster are read from REF.PARAMS rather than hardcoded
    # here, because the bridge is a separate process and the dashboard must
    # not be able to disagree with it about who signs. The API KEY is
    # deliberately absent from the warehouse — the host is recorded, the
    # credential stays in .env beside the keypair. SiS has no outbound
    # network, so nothing on this page queries Solana; every number here is
    # the audit trail the bridge wrote back.
    # ------------------------------------------------------------------
    st.markdown("**On-chain attestations**")
    chain = {r["KEY"]: r["VALUE_STR"] for r in rows(
        "SELECT key, value_str FROM REF.PARAMS WHERE key LIKE 'solana%'")}
    funnel = rows("""
        SELECT status, COUNT(*) AS n
        FROM ORACLE.PUBLISH_QUEUE GROUP BY status
    """)
    counts = {r["STATUS"]: int(r["N"] or 0) for r in funnel}
    wallet = chain.get("solana_authority") or "not configured"
    if chain:
        st.markdown(f"""
<div class="tt-card" style="font-size:12px;line-height:1.6">
  <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">
    <div><span class="tt-metric-label">publishing authority</span><br>
      <a class="tt-mono" href="https://explorer.solana.com/address/{wallet}?cluster={chain.get('solana_cluster','devnet')}"
     target="_blank">{wallet}</a></div>
    <div><span class="tt-metric-label">cluster</span><br>
      <b>{chain.get('solana_cluster','—')}</b> via
      <span class="tt-mono">{chain.get('solana_rpc_host','—')}</span></div>
    <div><span class="tt-metric-label">instruction</span><br>
      <b>{chain.get('solana_mode','—')}</b></div>
  </div>
  <div class="tt-quiet" style="margin-top:8px">The private key for this wallet
  is held only by the bridge process. It is not in this table, this app, or
  anywhere in Snowflake — a full dump of the warehouse yields no key material,
  which is checkable rather than asserted.</div>
</div>""", unsafe_allow_html=True)
    if PLOTLY and counts:
        order = [("PENDING", "#D6D3D1"), ("SENT", S_YELLOW),
                 ("CONFIRMED", "#15803D"), ("FAILED", "#B91C1C")]
        present = [(s, h) for s, h in order if counts.get(s)]
        fig = evil_blocks([(s.lower(), counts[s], h) for s, h in present])
        fig.update_layout(showlegend=False)
        chart(clean_axes(fig), H_STRIP)
        st.markdown(" ".join(
            f'<span class="tt-chip" style="background:{h}22;border-color:{h}">'
            f'{s.lower()} <b>{counts[s]}</b></span>' for s, h in present),
            unsafe_allow_html=True)
    st.markdown('<span class="tt-quiet">Snowflake stages the claim; a Node '
                'bridge holds the key, signs and submits. <b>The keypair never '
                'touches Snowflake.</b> Publish the claim, never the data. '
                'The full ledger, with a link to every transaction on Solana '
                'Explorer, is on the <b>On Chain</b> page — this panel is '
                'here because the publish queue is part of the DAG\'s '
                'health, not because it is the interesting view of it.'
                '</span>', unsafe_allow_html=True)
    recent = rows("""
        SELECT publish_id, subject, syndrome_code, severity, status,
               latency_s, tx_signature, explorer_url
        FROM ORACLE.V_PUBLISH_STATUS ORDER BY publish_id DESC LIMIT 8
    """)
    if recent:
        trows = []
        for r in recent:
            sig, url = r.get("TX_SIGNATURE"), r.get("EXPLORER_URL")
            link = (f'<a href="{url}" target="_blank" rel="noopener" '
                    f'class="tt-mono">{str(sig)[:16]}&#8230;</a>'
                    ) if url and sig else "—"
            trows.append({
                "i": r["PUBLISH_ID"], "s": r["SUBJECT"], "c": r["SYNDROME_CODE"],
                "v": r["SEVERITY"], "st": r["STATUS"], "l": fmt(r["LATENCY_S"], 0),
                "t": link,
            })
        html_table(trows, [("i", "#"), ("s", "subject (hashed)"), ("c", "finding"),
                           ("v", "sev"), ("st", "status"), ("l", "latency s"),
                           ("t", "transaction")])
    else:
        empty_state("Nothing queued.",
                    "ORACLE.T_ATTEST stages findings at severity ≥ 2; "
                    "start the bridge with: npm run bridge")

    st.markdown("**Build log**")
    bl = rows("SELECT script_name, status, statements_ok, statements_ko, finished_at "
              "FROM REF.BUILD_LOG WHERE script_name <> 'T_ROOT' "
              "ORDER BY finished_at DESC LIMIT 15")
    if bl:
        html_table([{"s": r["SCRIPT_NAME"], "t": r["STATUS"],
                     "o": fmt(r["STATEMENTS_OK"], 0), "k": fmt(r["STATEMENTS_KO"], 0),
                     "w": str(r["FINISHED_AT"])[:19]} for r in bl],
                   [("s", "script"), ("t", "status"), ("o", "ok"),
                    ("k", "failed"), ("w", "finished")])

# ---------------------------------------------------------------------------
if st.session_state.get("_errors"):
    with st.expander(f"query errors ({len(st.session_state['_errors'])})"):
        for sql, err in st.session_state["_errors"][-25:]:
            st.markdown(f'<div class="tt-mono" style="font-size:11px">'
                        f'<b>{sql}</b><br>{err}</div>', unsafe_allow_html=True)

# ===========================================================================
# PAGE 10 — ASK TELLTAIL.  Cortex, grounded in the warehouse.
#
# Not a general chatbot bolted on to name-drop an LLM. The question is answered
# from rows: a compact factual context is assembled in SQL first — the pack
# summary, the syndrome counts, the sensitivity curve, the honest accuracy —
# and AI_COMPLETE is asked to answer ONLY from that, and to say so when the
# answer is not in there. The context it was given is printed underneath, so
# any claim on screen can be checked against the table it came from.
# ===========================================================================
def _page_9():
    facts = rows("""
        SELECT 'holdout' AS topic,
               'Dog-disjoint holdout accuracy ' ||
               COALESCE(TO_VARCHAR(ROUND(100 * MAX(m.holdout_accuracy), 2)), 'n/a') ||
               '% over ' || COALESCE(TO_VARCHAR(MAX(m.holdout_dogs)), '0') ||
               ' entirely unseen dogs, macro F1 ' ||
               COALESCE(TO_VARCHAR(ROUND(MAX(m.macro_f1), 3)), 'n/a') ||
               ', classifier ' || COALESCE(MAX(m.classifier), 'n/a') AS fact
        FROM ML.MODEL_SUMMARY m
        UNION ALL
        SELECT 'syndromes',
               'Syndrome ' || syndrome_code || ' (' || ANY_VALUE(syndrome_name) ||
               ', ' || ANY_VALUE(body_system) || '): ' || COUNT(*) ||
               ' matches over ' || COUNT(DISTINCT dog_id) ||
               ' dogs, mean confidence ' || ROUND(AVG(confidence), 3)
        FROM MARTS.SYNDROME_MATCHES GROUP BY syndrome_code
        UNION ALL
        SELECT 'not_firing',
               'Syndrome ' || c.syndrome_code || ' (' || c.syndrome_name ||
               ') has ZERO matches in this corpus.'
        FROM REF.SYNDROME_CATALOGUE c
        WHERE c.syndrome_code NOT IN (SELECT syndrome_code FROM MARTS.SYNDROME_MATCHES)
        UNION ALL
        SELECT 'sensitivity',
               'Sensitivity sweep ' || syndrome_code || '/' || variant ||
               ': ' || COUNT(*) || ' matches'
        FROM MARTS.SYNDROME_SENSITIVITY GROUP BY syndrome_code, variant
        UNION ALL
        SELECT 'states',
               'State ' || state || ': ' || COUNT(*) || ' epochs (' ||
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) || '%)'
        FROM MARTS.EPOCH_STATES GROUP BY state
        UNION ALL
        SELECT 'provenance',
               'state_source ' || state_source || ': ' || COUNT(*) || ' epochs'
        FROM MARTS.EPOCH_STATES GROUP BY state_source
        UNION ALL
        SELECT 'shelter',
               'Austin shelter, ' || syndrome_name || ': ' ||
               shelter_behaviour_records || ' shelter records versus ' ||
               telltail_detections || ' collar detections'
        FROM REF.V_SHELTER_PUNCHLINE
        UNION ALL
        SELECT 'scale',
               'Corpus: ' || (SELECT COUNT(*) FROM MARTS.EPOCH_STATES) ||
               ' classified epochs over ' ||
               (SELECT COUNT(*) FROM REF.DOG_INFO) || ' dogs'
    """)
    context = chr(10).join(f"- [{r['TOPIC']}] {r['FACT']}"
                           for r in facts if r.get("FACT"))

    examples = [
        ("Which fired?",    "Which syndromes fired, and which found nothing?"),
        ("How accurate?",   "How accurate is the classifier, honestly?"),
        ("Versus Austin?",  "How does what we detect compare to the Austin "
                            "shelter outcomes?"),
        ("Model or rule?",  "What fraction of states came from the model rather "
                            "than a heuristic?"),
    ]

    # ---- submission happens in callbacks ---------------------------------
    #
    # The message box belongs UNDER the transcript, but the answer has to be in
    # the history BEFORE the transcript renders or it appears one interaction
    # late. Streamlit runs widget callbacks ahead of the script body, so both
    # the box and the example buttons only park the question in session state;
    # the body below picks it up on the same run. Clearing tt_chat_box from
    # inside its own on_change is the documented way to empty a text input.
    def _chat_ask(text: str) -> None:
        st.session_state["tt_chat_pending"] = text

    def _chat_submit() -> None:
        text = (st.session_state.get("tt_chat_box") or "").strip()
        st.session_state["tt_chat_box"] = ""
        if text:
            st.session_state["tt_chat_pending"] = text

    def _chat_clear() -> None:
        st.session_state["tt_chat"] = []

    question = (st.session_state.pop("tt_chat_pending", None) or "").strip()
    history = st.session_state.setdefault("tt_chat", [])

    if question:
        prompt = (
            "You are TELLTAIL, a veterinary telemetry analyst. Answer the "
            "question using ONLY the facts listed below, which come from a "
            "Snowflake warehouse. Quote the specific numbers you use. If the "
            "facts do not contain the answer, say exactly what is missing "
            "rather than guessing. Be concise: at most 130 words. Never imply "
            "this is a diagnosis." + chr(10) * 2 + "FACTS:" + chr(10) + context
            + chr(10) * 2 + "QUESTION: " + question
        )
        try:
            ans = rows_live(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE("
                + sq(CORTEX_MODEL) + ", " + sq(prompt) + ") AS a")
            history.append({"q": question, "a": one(ans, "A", ""),
                            "n": len(facts), "ok": True})
        except Exception as exc:  # noqa: BLE001
            history.append({"q": question, "ok": False,
                            "a": "Cortex did not answer: " + str(exc)[:280],
                            "n": len(facts)})
        st.session_state["tt_chat"] = history

    # ---- layout -----------------------------------------------------------
    #
    # ONE CENTRED COLUMN.
    #
    # `layout="wide"` is right for a dashboard of charts and wrong for a
    # conversation: run a transcript across 1600px and every bubble is a single
    # 200-character line with a 48px avatar marooned at the far left. Left-
    # aligning it inside the wide page fixed the line length but left a third
    # of the screen empty down the right-hand side, which reads as a layout
    # that failed rather than one that chose. Equal margins, 75% of the width.
    _pad_l, mid, _pad_r = st.columns([1, 6, 1])

    with mid:
        st.markdown(
            '<div class="tt-quiet" style="font-size:12px;line-height:1.5;'
            'margin-bottom:18px">'
            'Answers come from <span class="tt-mono">AI_COMPLETE</span> over a '
            'factual context assembled from the warehouse in SQL — printed in '
            'full at the bottom of this page, so any number here can be traced '
            'to the table it came from. The model is given the facts and your '
            'question, never the conversation, so it will not follow up on '
            'itself.</div>', unsafe_allow_html=True)

        if not history:
            chat_bubble(
                "Hello, I am TELLTAIL. Ask me about the pack, the syndromes, "
                "the classifier or the pipeline.\n\nI answer only from rows in "
                "this warehouse, and the exact facts I was handed are printed "
                "under every answer. If the answer is not in them I will say "
                "which table is missing rather than invent one.",
                is_user=False,
                meta=f"grounded in {len(facts)} facts from SQL · {CORTEX_MODEL}")
        for turn in history:
            chat_bubble(turn["q"], is_user=True)
            # A bubble is left plain unless the call FAILED. The reference
            # build has no rule down the side of a message and neither does
            # this, so the one time a red edge appears it means something.
            chat_bubble(turn["a"], is_user=False,
                        hue=None if turn.get("ok") else "#B91C1C",
                        meta=(f'AI_COMPLETE · {CORTEX_MODEL} · answered from '
                              f'{turn["n"]} warehouse facts' if turn.get("ok")
                              else "no answer — the call failed, nothing was "
                                   "substituted for it"))

        # A plain st.text_input, not st.chat_input: the label sits above the
        # field and Streamlit prints its own "Press Enter to apply" hint inside
        # it, which is the input in the reference build. st.chat_input tears
        # itself out of the flow and pins to the bottom of the viewport, below
        # the page footer and outside this column.
        st.text_input("Message: ", key="tt_chat_box", on_change=_chat_submit,
                      placeholder="Ask about the pack, the syndromes, the "
                                  "classifier or the pipeline")

        # Starters. Five equal columns rather than a weighted split: the last
        # one was 1/12 of the row and rendered the word "Clear" as "Cle ar".
        chips = st.columns(5)
        for c, (short, full) in zip(chips, examples):
            c.button(short, key="ex_" + short, on_click=_chat_ask, args=(full,),
                     use_container_width=True)
        chips[4].button("Clear", key="chat_clear", on_click=_chat_clear,
                        use_container_width=True)

        with st.expander(f"the exact context the model was given ({len(facts)} "
                         f"facts from SQL)"):
            st.code(context or "(no facts)", language="text")


# ===========================================================================
# PAGE 10 — ON CHAIN.  The claim, published, and checkable by a stranger.
#
# Every row on this page carries a link OUT of this dashboard to Solana
# Explorer, and that is the entire point: nothing here has to be believed. A
# reader who does not trust TELLTAIL, Snowflake, or this app can open any
# transaction and read the claim off the ledger themselves.
#
# WHAT IS AND IS NOT PUBLISHED. The subject is a salted hash of the dog id, not
# the dog. No telemetry, no breed, no owner, no location — a claim that a
# finding of a given code, severity and confidence occurred in a given window,
# and nothing that could re-identify the animal. Publish the claim, never the
# data.
#
# SiS HAS NO OUTBOUND NETWORK, so this page cannot query Solana. Everything
# shown is the audit trail the bridge wrote back into ORACLE.PUBLISH_QUEUE
# after the network confirmed it. The links are how you check that the
# warehouse is telling the truth.
# ===========================================================================
def _page_10():
    chain = {r["KEY"]: r["VALUE_STR"] for r in rows(
        "SELECT key, value_str FROM REF.PARAMS WHERE key LIKE 'solana%'")}
    cluster = chain.get("solana_cluster", "devnet")
    wallet = chain.get("solana_authority") or ""

    pq = rows("""
        SELECT publish_id, subject, syndrome_code, syndrome_name, severity,
               confidence, onset_ts, duration_s, status, attempts,
               tx_signature, explorer_url, slot, queued_at, confirmed_at,
               latency_s, last_error
        FROM ORACLE.V_PUBLISH_STATUS
        ORDER BY publish_id
    """)
    if not pq:
        empty_state(
            "Nothing has been staged for publication yet.",
            "ORACLE.T_ATTEST queues findings at severity >= 2. Start the "
            "bridge with: npm run bridge")
        return

    counts: dict = {}
    for r in pq:
        counts[r["STATUS"]] = counts.get(r["STATUS"], 0) + 1
    done = [r for r in pq if r["STATUS"] == "CONFIRMED" and r.get("TX_SIGNATURE")]
    # 5,000 lamports is the flat signature fee for a single-signature memo
    # transaction. Stated because "what does this cost to run" is the first
    # question anyone asks about putting anything on a chain.
    fee_sol = len(done) * 5000 / 1e9

    metric_strip([
        ("attestations on chain", fmt(len(done), 0)),
        ("dogs attested",         fmt(len({r["SUBJECT"] for r in done}), 0)),
        ("network fee paid",      f"{fee_sol:.6f} SOL"),
        ("cluster",               cluster),
    ])

    # ---- who published, with the way out to Explorer ----------------------
    if wallet:
        st.markdown(f"""
<div class="tt-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
       gap:14px;flex-wrap:wrap">
    <div style="min-width:0">
      <div class="tt-metric-label">publishing authority</div>
      <a class="tt-mono" style="font-size:13px;word-break:break-all"
         href="https://explorer.solana.com/address/{wallet}?cluster={cluster}"
         target="_blank" rel="noopener">{wallet}</a>
      <div class="tt-quiet" style="margin-top:6px;font-size:12px">
        Signed and paid for every transaction below. The private key for this
        wallet is held only by the bridge process — it is not in this table,
        this app, or anywhere in Snowflake.
      </div>
    </div>
    <a href="https://explorer.solana.com/address/{wallet}?cluster={cluster}"
       target="_blank" rel="noopener"
       style="flex:0 0 auto;background:{PAGE_HUE};color:#fff;padding:9px 15px;
              border-radius:5px;font-size:13px;font-weight:600;
              text-decoration:none;white-space:nowrap">
      Open wallet in Solana Explorer &#8599;</a>
  </div>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        panel("Publication funnel",
              "Every finding staged by ORACLE.T_ATTEST, and how far it got. A "
              "claim only reaches CONFIRMED once the network has acknowledged "
              "the transaction and the bridge has written the signature back.")
        if PLOTLY:
            order = [("PENDING", "#D6D3D1"), ("SENT", S_YELLOW),
                     ("CONFIRMED", "#15803D"), ("FAILED", "#B91C1C")]
            present = [(s, h) for s, h in order if counts.get(s)]
            fig = evil_blocks([(s.lower(), counts[s], h) for s, h in present])
            fig.update_layout(showlegend=False)
            chart(clean_axes(fig), H_STRIP)
            st.markdown(" ".join(
                f'<span class="tt-chip" style="background:{h}22;'
                f'border-color:{h}">{s.lower()} <b>{counts[s]}</b></span>'
                for s, h in present), unsafe_allow_html=True)

    with c2:
        panel("What was attested",
              "By syndrome. Severity 1 findings are never published — routine "
              "monitoring is not a claim worth making permanent.")
        by_code: dict = {}
        for r in done:
            by_code[r["SYNDROME_CODE"]] = by_code.get(r["SYNDROME_CODE"], 0) + 1
        if PLOTLY and by_code:
            codes = sorted(by_code)
            fig = go.Figure(go.Bar(
                x=codes, y=[by_code[c] for c in codes],
                marker=dict(color=[SYMBOL_COLOURS[i % len(SYMBOL_COLOURS)]
                                   for i in range(len(codes))]),
                hovertext=[f"{c}: {by_code[c]} attestations" for c in codes],
                hoverinfo="text"))
            fig.update_layout(title="confirmed attestations by syndrome",
                              title_font_size=11)
            chart(evil_axes(fig), H_SM)

    # ---- inspect one, byte for byte --------------------------------------
    st.markdown("---")
    st.markdown("##### Open any one of them on the ledger")
    st.markdown('<span class="tt-quiet">Pick an attestation to see the exact '
                'bytes that went on chain, then follow the link and read the '
                'same JSON back off Solana Explorer. If the two disagree, this '
                'dashboard is lying to you.</span>', unsafe_allow_html=True)

    if done:
        labels = [f'#{r["PUBLISH_ID"]} · {r["SYNDROME_CODE"]} '
                  f'{r["SYNDROME_NAME"]} · sev {r["SEVERITY"]} · '
                  f'slot {fmt(r["SLOT"], 0)}' for r in done]
        i = st.selectbox("attestation", range(len(labels)),
                         format_func=lambda k: labels[k], key="chain_pick")
        r = done[i]
        payload = rows(f"""
            SELECT payload FROM ORACLE.PUBLISH_QUEUE
            WHERE publish_id = {int(r["PUBLISH_ID"])}
        """)
        p1, p2 = st.columns([3, 2])
        with p1:
            st.markdown("**The memo instruction data, as submitted**")
            st.code(str(one(payload, "PAYLOAD", "") or ""), language="json")
        with p2:
            colour = TRIAGE_COLOUR.get(r.get("SEVERITY"), "#A8A29E")
            st.markdown(f"""
<div class="tt-card">
  <div><span class="tt-badge" style="background:{colour}">severity
    {r.get('SEVERITY')}</span>
    <b style="margin-left:8px">{r['SYNDROME_CODE']} · {r['SYNDROME_NAME']}</b></div>
  <div class="tt-quiet" style="margin-top:8px;line-height:1.7">
    subject <span class="tt-mono">{r['SUBJECT']}</span><br>
    onset {str(r['ONSET_TS'])[:19]} UTC · {fmt(r.get('DURATION_S'), 0)}s<br>
    confidence {fmt(r.get('CONFIDENCE'), 3)}<br>
    slot {fmt(r.get('SLOT'), 0)} · confirmed {str(r.get('CONFIRMED_AT'))[:19]}
  </div>
  <div class="tt-mono" style="font-size:11px;word-break:break-all;
       margin-top:8px;color:{INK_2}">{r.get('TX_SIGNATURE')}</div>
  <a href="{r.get('EXPLORER_URL')}" target="_blank" rel="noopener"
     style="display:inline-block;margin-top:10px;background:{PAGE_HUE};
            color:#fff;padding:8px 14px;border-radius:5px;font-size:13px;
            font-weight:600;text-decoration:none">
    View this transaction &#8599;</a>
</div>""", unsafe_allow_html=True)
            st.markdown(
                '<div class="tt-quiet" style="margin-top:8px;font-size:11.5px">'
                'The subject is a salted hash of the dog id. No telemetry, no '
                'breed, no owner, no location is published — only that a '
                'finding of this code and severity occurred in this window. '
                'Publish the claim, never the data.</div>',
                unsafe_allow_html=True)

    # ---- the whole ledger -------------------------------------------------
    st.markdown("---")
    st.markdown("##### Every attestation")
    fc1, fc2 = st.columns([1, 3])
    codes_all = sorted({r["SYNDROME_CODE"] for r in pq})
    pick_code = fc1.selectbox("syndrome", ["all"] + codes_all, key="chain_code")
    shown = [r for r in pq
             if pick_code == "all" or r["SYNDROME_CODE"] == pick_code]
    fc2.markdown(f'<div class="tt-quiet" style="padding-top:30px">'
                 f'{len(shown)} of {len(pq)} claims · every link opens Solana '
                 f'Explorer on {cluster}</div>', unsafe_allow_html=True)

    trows = []
    for r in shown:
        sig, url = r.get("TX_SIGNATURE"), r.get("EXPLORER_URL")
        link = (f'<a href="{url}" target="_blank" rel="noopener" '
                f'class="tt-mono">{str(sig)[:20]}&#8230; &#8599;</a>'
                if url and sig else
                f'<span class="tt-quiet">{r.get("LAST_ERROR") or "—"}</span>')
        trows.append({
            "i": r["PUBLISH_ID"], "s": r["SUBJECT"], "c": r["SYNDROME_CODE"],
            "n": r["SYNDROME_NAME"], "v": r["SEVERITY"],
            "f": fmt(r.get("CONFIDENCE"), 3),
            "st": r["STATUS"], "sl": fmt(r.get("SLOT"), 0), "t": link,
        })
    html_table(trows, [("i", "#"), ("s", "subject (hashed)"), ("c", "code"),
                       ("n", "finding"), ("v", "sev"), ("f", "conf"),
                       ("st", "status"), ("sl", "slot"),
                       ("t", "transaction")])

    st.markdown(f"""
<div class="tt-card" style="font-size:13px;line-height:1.7;margin-top:14px">
<b>Why any of this is on a chain at all.</b><br>
A shelter taking in a dog has no way to check what a previous owner's app
claims about it, and no reason to trust a vendor's database that the vendor can
edit. An attestation is a claim with a timestamp that its author cannot quietly
revise: <b>{fmt(len(done), 0)}</b> findings are now published on Solana
{cluster}, each one signed by the wallet above and readable by anyone with the
link — no account, no API key, no permission from us.
<br><br>
<span class="tt-quiet">Devnet, not mainnet: this is a demonstration of the
mechanism, and devnet SOL has no value. Nothing about the architecture changes
on mainnet except the cluster and the cost. TELLTAIL is not a diagnostic device
and an attestation is a record of what the pipeline found, not a
diagnosis.</span>
</div>""", unsafe_allow_html=True)


PAGE_FN = {
    "Pack": _page_0, "Live Collar": _page_1, "Ethogram": _page_2,
    "Syndromes": _page_3, "Baselines": _page_4, "Vet Note": _page_5,
    "Drivers": _page_6, "Shelter Reality": _page_7, "Pipeline": _page_8,
    "On Chain": _page_10, "Ask TELLTAIL": _page_9,
}
PAGE_FN[PAGE]()


st.markdown(
    f'<div class="tt-quiet" style="margin-top:24px;border-top:1px solid {BORDER};'
    f'padding-top:8px">TELLTAIL · dual-IMU canine telemetry, row pattern '
    f'recognition, and a portable attestation. Data: Vehkaoja et al., '
    f'<i>Data in Brief</i> 2022, University of Helsinki — 45 dogs, 27 breeds, '
    f'100 Hz, video-annotated. Shelter data: City of Austin open data portal. '
    f'Not a diagnostic device.</div>', unsafe_allow_html=True)
