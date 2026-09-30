#!/usr/bin/env python3
"""
Does the synthetic demo signal actually classify as the behaviour it claims?

scripts/demo_spike.py does not write states. It writes 100 Hz accelerometer and
gyroscope SIGNAL and lets the real pipeline classify it — which is what makes
the demo a demonstration rather than a puppet show, and also what makes it
silently breakable. Nudge a threshold in REF.PARAMS and the "scratch" signal
quietly becomes a STAND, MATCH_RECOGNIZE finds nothing, and you discover it
while recording the video on Sunday evening.

So: synthesise each recipe, run the real feature math (mirrored in
tests/epoch_features.py), apply the real state ladder with thresholds parsed out
of warehouse/02_ref_seed.sql, and assert.

Then do the same for whole syndrome signatures, end to end: signal -> features
-> states -> MATCH_RECOGNIZE. If this file is green, `demo_spike.py --syndrome
S1` will produce an S1 match in the warehouse, on the rules-classifier path.

    python tests/test_demo_signal.py
    python tests/test_demo_signal.py --calibrate     # print the feature table
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import demo_spike as ds  # noqa: E402
from epoch_features import (  # noqa: E402
    ACTIVITY_CLASS,
    epoch_features,
    ladder_state,
    load_params,
    rules_state,
)
from match_recognize_sim import compile_pattern, match_partition  # noqa: E402
from test_syndrome_patterns import CATALOGUE  # noqa: E402

PARAMS = load_params()
SEED = 4242

# States the rules fallback cannot express. SIT and STAND are both "upright and
# stationary" and separating them needs a per-dog posture reference the fallback
# does not have; PLAY is irregular by definition. Both are available on the
# ML.CLASSIFICATION path, and no syndrome depends on either.
MODEL_ONLY = {"SIT", "PLAY"}


def features_for(state: str, n_epochs: int = 8, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [epoch_features(ds.synth_second(state, float(k), rng)) for k in range(n_epochs)]


def classify_sequence(sequence: list[tuple[str, int]], seed: int = SEED) -> list[dict]:
    """signal -> features -> states, with the PAUSE context rung wired up the
    way the SQL wires it (locomotion within two epochs either side)."""
    rng = np.random.default_rng(seed)
    feats: list[dict] = []
    t = 0.0
    for state, n in sequence:
        for _ in range(n):
            feats.append(epoch_features(ds.synth_second(state, t, rng)))
            t += 1.0

    base = [rules_state(f, PARAMS) for f in feats]

    # PLAIN locomotion only — a pacing epoch does not count as gait for the
    # PAUSE rung, mirroring `is_plain_loco` in MARTS.EPOCH_STATES. Otherwise the
    # alert stand between two pacing runs is promoted to PAUSE and S5 never
    # fires.
    loco = [
        s in ("WALK", "TROT", "GALLOP")
        and not (s in ("WALK", "TROT")
                 and f["yaw_consistency"] <= PARAMS["pace_yaw_consistency_max"]
                 and f["yaw_abs_mean"] >= PARAMS["pace_yaw_activity_min"])
        for s, f in zip(base, feats)
    ]

    out = []
    for i, f in enumerate(feats):
        before = any(loco[max(0, i - 2):i])
        after = any(loco[i + 1:i + 3])
        state, source = ladder_state(f, PARAMS, loco_before=before, loco_after=after)
        out.append({"epoch_ts": i, "state": state, "source": source, "dog_id": 1,
                    "test_num": 1,
                    # S3 matches activity_class, mirroring MARTS.V_SYNDROME_INPUT
                    "activity_class": ACTIVITY_CLASS.get(state, "OTHER")})
    return out


def singleton_diagnostic_states() -> set[str]:
    """States exempt from smoothing, parsed out of REF.ETHOGRAM in the seed SQL.

    Read rather than hardcoded, so adding a syndrome whose pattern needs a new
    bare variable moves this test with it.
    """
    text = (Path(__file__).resolve().parent.parent
            / "warehouse" / "02_ref_seed.sql").read_text(encoding="utf-8")
    import re
    block = re.search(r"INSERT INTO REF\.ETHOGRAM(.*?)AS v\(state,", text, re.S)
    assert block, "could not locate the REF.ETHOGRAM insert"
    out = set()
    for m in re.finditer(
        r"\('([A-Z_]+)',\s*'[^']*',\s*'[^']*',\s*'[^']*',\s*(?:TRUE|FALSE),\s*(TRUE|FALSE)",
        block.group(1),
    ):
        if m.group(2) == "TRUE":
            out.add(m.group(1))
    assert out, "parsed no singleton_diagnostic states"
    return out


SINGLETON_DIAGNOSTIC = singleton_diagnostic_states()


def despeckle(states: list[dict]) -> list[dict]:
    """The guarded three-point filter from MARTS.EPOCH_STATES.

    An isolated epoch flanked by two identical different states is replaced by
    them — UNLESS its state is singleton-diagnostic, because for those states a
    single epoch IS the clinical event. Smoothing the head shake out of
    SCRATCH SHAKE SCRATCH destroys the alternation S1 is defined by.
    """
    out = [dict(s) for s in states]
    for i in range(1, len(states) - 1):
        cur = states[i]["state"]
        if cur in SINGLETON_DIAGNOSTIC:
            continue
        prev, nxt = states[i - 1]["state"], states[i + 1]["state"]
        if prev == nxt and prev != cur:
            out[i]["state"] = prev
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_every_recipe_produces_its_state():
    wrong = []
    for state in ds.RECIPES:
        if state in MODEL_ONLY:
            continue
        loco = state == "PAUSE"
        for f in features_for(state):
            got, _src = ladder_state(f, PARAMS, loco_before=loco, loco_after=loco)
            if got != state:
                wrong.append((state, got, round(f["vm_neck_std"], 3),
                              round(f["neck_back_corr"], 3)))
                break
    assert not wrong, (
        "synthetic recipes no longer classify as the behaviour they claim:\n  "
        + "\n  ".join(f"{want} -> {got}  (vm_std {v}, corr {c})"
                      for want, got, v, c in wrong)
        + "\n  Re-calibrate: python tests/test_demo_signal.py --calibrate"
    )


def test_model_only_states_degrade_predictably():
    """SIT must fall back to STAND rather than to something wrong. A SIT epoch
    landing as SCRATCH would put a spurious symbol into a syndrome sequence."""
    for state in MODEL_ONLY & set(ds.RECIPES):
        for f in features_for(state, n_epochs=4):
            got, _ = ladder_state(f, PARAMS)
            assert got in ("STAND", "REST"), (
                f"{state} degraded to {got}; the rules fallback should read it as "
                f"an ordinary still posture, not as a behaviour"
            )


def test_coupling_drives_the_correlation_feature():
    """The claim the whole project rests on: two sensors moving together means
    whole-body locomotion, two sensors decoupled means the neck moved and the
    body did not. If synthesis cannot reproduce that separation, neither the
    demo nor the thesis holds."""
    coupled = ["WALK", "TROT", "GALLOP"]
    decoupled = ["SCRATCH", "SHAKE", "SNIFF"]

    lo = min(np.mean([f["neck_back_corr"] for f in features_for(s)]) for s in coupled)
    hi = max(np.mean([f["neck_back_corr"] for f in features_for(s)]) for s in decoupled)

    assert lo > 0.80, f"locomotion correlation is only {lo:.3f}; expected well above 0.8"
    assert hi < 0.35, f"neck-dominant correlation reaches {hi:.3f}; expected near zero"
    assert lo - hi > 0.5, (
        f"the two families are only {lo - hi:.3f} apart in correlation; "
        f"CORR(vm_neck, vm_back) is not separating them"
    )


def test_neck_dominance_separates_shake_from_scratch():
    """S1's entire signal is the ALTERNATION of head shake and scratch bout. If
    the two collapse into one state the pattern degenerates and never fires."""
    sc = np.mean([f["vm_neck_std"] for f in features_for("SCRATCH")])
    sh = np.mean([f["vm_neck_std"] for f in features_for("SHAKE")])
    lo, hi = PARAMS["scratch_vm_std_min"], PARAMS["scratch_vm_std_max"]

    assert lo < sc < hi, f"SCRATCH vm_std {sc:.3f} is outside its band [{lo}, {hi}]"
    assert sh > PARAMS["shake_vm_std_min"], (
        f"SHAKE vm_std {sh:.3f} does not clear its floor {PARAMS['shake_vm_std_min']}"
    )
    # margin, not just correctness: both should sit away from the boundary
    assert sc - lo > 0.08 and hi - sc > 0.08, (
        f"SCRATCH vm_std {sc:.3f} sits on the edge of [{lo}, {hi}]; a small "
        f"threshold change would silently reclassify it"
    )
    assert sh - PARAMS["shake_vm_std_min"] > 0.10, (
        f"SHAKE vm_std {sh:.3f} sits on its floor {PARAMS['shake_vm_std_min']}"
    )


def test_yaw_geometry_is_not_a_phase_artefact():
    """yaw_consistency = |mean yaw| / mean|yaw| decides CIRCLE and PACE. If the
    oscillating component does not integrate to zero over an epoch, the ratio
    depends on where the epoch boundary happens to fall and CIRCLE fires on
    scratching. Checked across offsets, which is where that bug shows up."""
    for state, want in [("CIRCLE", "high"), ("PACE", "low"), ("SCRATCH", "low")]:
        rng = np.random.default_rng(7)
        vals = [epoch_features(ds.synth_second(state, off, rng))["yaw_consistency"]
                for off in (0.0, 0.13, 0.37, 0.5, 0.71, 0.94)]
        if want == "high":
            assert min(vals) >= PARAMS["circle_yaw_consistency_min"], (
                f"{state} yaw_consistency drops to {min(vals):.3f} at some epoch "
                f"offset; CIRCLE would flicker")
        else:
            assert max(vals) <= PARAMS["pace_yaw_consistency_max"], (
                f"{state} yaw_consistency reaches {max(vals):.3f} at some epoch "
                f"offset; it would be misread as circling")


def test_each_syndrome_signature_fires_end_to_end():
    """The full chain, offline: synthetic signal -> real feature math -> real
    state ladder -> real MATCH_RECOGNIZE pattern.

    This is the check that says `demo_spike.py --syndrome S4` will actually
    produce an S4 match, rather than producing plausible-looking states that
    miss the quantifiers by one.
    """
    failures = []
    for code, sequence in sorted(ds.SIGNATURES.items()):
        states = despeckle(classify_sequence(sequence))
        cat = CATALOGUE[code]
        cp = compile_pattern(cat["pattern_text"], cat["define_text"])
        matches = match_partition(cp, states)
        if not matches:
            failures.append(
                f"{code} ({cat['syndrome_name']}) produced no match\n"
                f"      pattern : {cat['pattern_text']}\n"
                f"      intended: {' '.join(f'{s}x{n}' for s, n in sequence)}\n"
                f"      actual  : {' '.join(s['state'] for s in states)}"
            )
    assert not failures, (
        "demo_spike signatures do not produce their syndrome:\n  "
        + "\n  ".join(failures)
    )


def test_signatures_only_fire_their_own_syndrome():
    """An S1 injection must not also trip S3. Cross-firing would make every demo
    ambiguous about what was actually detected."""
    cross = []
    for code, sequence in sorted(ds.SIGNATURES.items()):
        states = despeckle(classify_sequence(sequence))
        for other, cat in sorted(CATALOGUE.items()):
            if other == code:
                continue
            cp = compile_pattern(cat["pattern_text"], cat["define_text"])
            if match_partition(cp, states):
                cross.append(f"{code} signature also matched {other}")
    assert not cross, "cross-firing signatures:\n  " + "\n  ".join(cross)


def test_despeckle_does_not_invent_states():
    """The smoothing filter may only replace an epoch with a state that already
    occurs adjacent to it. Anything else would be inventing behaviour."""
    rng = np.random.default_rng(3)
    vocab = ["REST", "WALK", "TROT", "SCRATCH", "STAND"]
    seq = [{"state": rng.choice(vocab), "epoch_ts": i} for i in range(2000)]
    out = despeckle(seq)
    for i, (a, b) in enumerate(zip(seq, out)):
        if a["state"] != b["state"]:
            assert b["state"] == seq[i - 1]["state"] == seq[i + 1]["state"], (
                f"despeckle at {i} produced {b['state']!r}, which is not the "
                f"flanking state"
            )


def test_despeckle_never_erases_a_diagnostic_singleton():
    """The bug this guard exists for, pinned so it cannot come back.

    S1's head shake sits alone between two scratch bouts. S2's pause sits alone
    between two stride runs. S5's alert stand sits alone between two pacing
    runs. An unguarded three-point filter deletes all three, and every one of
    those syndromes then silently never fires — the worst possible failure mode,
    because nothing errors.
    """
    cases = [
        (["SCRATCH", "SHAKE", "SCRATCH"], "SHAKE", "S1 head shake"),
        (["WALK", "PAUSE", "WALK"], "PAUSE", "S2 stride interruption"),
        (["PACE", "STAND", "PACE"], "STAND", "S5 alert stand"),
        (["REST", "SLOW_TRANSITION", "REST"], "SLOW_TRANSITION", "S4 lever-up"),
    ]
    for seq, protected, why in cases:
        rowset = [{"state": s, "epoch_ts": i} for i, s in enumerate(seq)]
        assert despeckle(rowset)[1]["state"] == protected, (
            f"the despeckle filter erased the {why}. REF.ETHOGRAM."
            f"singleton_diagnostic must be TRUE for {protected}."
        )

    # ...and it still removes genuine speckle
    rowset = [{"state": s, "epoch_ts": i}
              for i, s in enumerate(["SCRATCH", "WALK", "SCRATCH"])]
    assert despeckle(rowset)[1]["state"] == "SCRATCH", (
        "a lone WALK inside a scratch bout should be smoothed away; otherwise "
        "itch{3,} stays broken by classifier flicker"
    )


# ---------------------------------------------------------------------------

def calibrate() -> int:
    print(f"\n{'recipe':<17} {'vm_mean':>8} {'vm_std':>7} {'corr':>7} {'domin':>7} "
          f"{'pitchM':>7} {'pitchV':>7} {'yawC':>6} {'yawA':>6} {'dynB':>6}  "
          f"{'->state':<16} source")
    print("-" * 138)
    for state in ds.RECIPES:
        fs = features_for(state)
        f = {k: float(np.mean([x[k] for x in fs])) for k in fs[0]}
        loco = state == "PAUSE"
        got, src = ladder_state(f, PARAMS, loco_before=loco, loco_after=loco)
        dyn = abs(f["vm_back_mean"] - PARAMS["gravity_ref"])
        note = "  (model-only)" if state in MODEL_ONLY else (
            "" if got == state else "  <-- MISMATCH")
        print(f"{state:<17} {f['vm_neck_mean']:8.3f} {f['vm_neck_std']:7.3f} "
              f"{f['neck_back_corr']:7.3f} {f['neck_dominance']:7.2f} "
              f"{f['pitch_neck_mean']:7.3f} {f['pitch_var']:7.3f} "
              f"{f['yaw_consistency']:6.2f} {f['yaw_abs_mean']:6.2f} {dyn:6.3f}  "
              f"{got:<16} {src}{note}")

    print(f"\n{'syndrome':<6} {'states produced'}")
    print("-" * 138)
    for code, sequence in sorted(ds.SIGNATURES.items()):
        states = despeckle(classify_sequence(sequence))
        cat = CATALOGUE[code]
        cp = compile_pattern(cat["pattern_text"], cat["define_text"])
        n = len(match_partition(cp, states))
        runs: list[list] = []
        for s in states:
            if runs and runs[-1][0] == s["state"]:
                runs[-1][1] += 1
            else:
                runs.append([s["state"], 1])
        print(f"{code:<6} {n} match(es)   " + " ".join(f"{a}x{b}" for a, b in runs))
    return 0


def main() -> int:
    if "--calibrate" in sys.argv:
        return calibrate()

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    print(f"\nTELLTAIL — demo signal validation  ({len(tests)} tests)")
    print("=" * 64)
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except BaseException as exc:  # noqa: BLE001
            print(f"  FAIL  {name}")
            failures.append((name, exc))
    print("=" * 64)
    if failures:
        for name, exc in failures:
            print(f"\n--- {name} ---\n{exc}")
        print(f"\n{len(failures)} of {len(tests)} failed.")
        print("Inspect the numbers:  python tests/test_demo_signal.py --calibrate")
        return 1
    print(f"\nAll {len(tests)} passed. Every synthetic recipe classifies as the")
    print("behaviour it claims, and each syndrome signature fires its own")
    print("pattern and only its own — on the rules-classifier path, offline.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
