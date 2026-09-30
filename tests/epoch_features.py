"""
A Python mirror of STAGING.EPOCH_FEATURES and the MARTS.EPOCH_STATES ladder.

This exists so scripts/demo_spike.py can be verified offline. The spike
synthesises 100 Hz IMU signal and relies on the real pipeline to classify it —
which is what makes the demo a demonstration rather than a puppet show, and
also what makes it silently breakable: change a threshold in REF.PARAMS and the
"scratch" signal quietly becomes a STAND, MATCH_RECOGNIZE finds nothing, and
you discover it while recording the video.

So the feature math and the state ladder are reimplemented here, the thresholds
are parsed out of warehouse/02_ref_seed.sql rather than copied, and
tests/test_demo_signal.py asserts every recipe still produces the state it
claims to.

This mirrors the RULES classifier path (warehouse/05) plus the geometry rungs of
the ladder (warehouse/06). It does not mirror ML.CLASSIFICATION, which has no
offline equivalent. That is the correct scope: the spike needs to be robust
against the documented fallback classifier, and the geometry rungs are pure
threshold arithmetic that the model never sees anyway.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REF_SEED = REPO / "warehouse" / "02_ref_seed.sql"

# Column order used throughout: neck ax ay az gx gy gz, back ax ay az gx gy gz
NECK_AX, NECK_AY, NECK_AZ, NECK_GX, NECK_GY, NECK_GZ = range(6)
BACK_AX, BACK_AY, BACK_AZ, BACK_GX, BACK_GY, BACK_GZ = range(6, 12)


def load_params() -> dict[str, float]:
    """Parse REF.PARAMS numeric values out of the seed SQL.

    Reading the SQL rather than hardcoding means a threshold change in one place
    moves the test with it, which is the only way this check keeps its value.
    """
    text = REF_SEED.read_text(encoding="utf-8")
    block = re.search(
        r"INSERT INTO REF\.PARAMS.*?AS v\(key, value_num", text, re.S
    )
    if not block:
        raise RuntimeError("could not locate the REF.PARAMS insert in 02_ref_seed.sql")

    out: dict[str, float] = {}
    for m in re.finditer(
        r"\('([a-z0-9_]+)',\s*(-?[0-9.]+|NULL)\s*,", block.group(0)
    ):
        key, val = m.group(1), m.group(2)
        if val != "NULL":
            out[key] = float(val)
    if "shake_corr_max" not in out:
        raise RuntimeError(f"params parse looks wrong; got {sorted(out)}")
    return out


def load_activity_class() -> dict[str, str]:
    """state -> activity_class, parsed out of the REF.ETHOGRAM seed.

    Parsed rather than copied for the same reason load_params is: S3 matches on
    activity_class, so a regrouping in the warehouse that this file did not
    follow would leave the offline suite validating a pattern the warehouse no
    longer runs.
    """
    text = REF_SEED.read_text(encoding="utf-8")
    block = re.search(r"INSERT INTO REF\.ETHOGRAM(.*?)AS v\(state,", text, re.S)
    if not block:
        raise RuntimeError("could not locate the REF.ETHOGRAM insert in 02_ref_seed.sql")
    out = {
        m.group(1): m.group(2)
        for m in re.finditer(
            r"\('([A-Z_]+)',\s*'[^']*',\s*'[^']*',\s*'[^']*',\s*"
            r"(?:TRUE|FALSE),\s*(?:TRUE|FALSE),\s*'([A-Z_]+)'",
            block.group(1),
        )
    }
    if not out:
        raise RuntimeError("parsed no states out of the REF.ETHOGRAM insert")
    return out


ACTIVITY_CLASS = load_activity_class()


def epoch_features(block: np.ndarray) -> dict[str, float]:
    """One second of samples (n, 12) -> the feature vector the SQL computes.

    STDDEV in Snowflake is the SAMPLE standard deviation, so ddof=1 throughout.
    Getting that wrong shifts every threshold comparison slightly, which is
    exactly the kind of near-miss this whole file exists to catch.
    """
    n = block.shape[0]
    vm_neck = np.sqrt(
        block[:, NECK_AX] ** 2 + block[:, NECK_AY] ** 2 + block[:, NECK_AZ] ** 2
    )
    vm_back = np.sqrt(
        block[:, BACK_AX] ** 2 + block[:, BACK_AY] ** 2 + block[:, BACK_AZ] ** 2
    )
    gm_neck = np.sqrt(
        block[:, NECK_GX] ** 2 + block[:, NECK_GY] ** 2 + block[:, NECK_GZ] ** 2
    )
    sma_neck = np.abs(block[:, NECK_AX]) + np.abs(block[:, NECK_AY]) + np.abs(block[:, NECK_AZ])
    sma_back = np.abs(block[:, BACK_AX]) + np.abs(block[:, BACK_AY]) + np.abs(block[:, BACK_AZ])

    pitch_neck = np.arctan2(
        block[:, NECK_AX],
        np.sqrt(block[:, NECK_AY] ** 2 + block[:, NECK_AZ] ** 2),
    )
    roll_neck = np.arctan2(
        block[:, NECK_AY],
        np.sqrt(block[:, NECK_AX] ** 2 + block[:, NECK_AZ] ** 2),
    )

    yaw = block[:, BACK_GZ]

    # zero-crossing rate of the epoch-mean-removed neck magnitude
    ac = vm_neck - vm_neck.mean()
    sgn = np.sign(ac)
    crossings = int(np.sum((sgn[1:] != sgn[:-1]) & (sgn[1:] != 0)))

    std_neck = float(np.std(vm_neck, ddof=1))
    std_back = float(np.std(vm_back, ddof=1))

    return {
        "n_samples": n,
        "vm_neck_mean": float(vm_neck.mean()),
        "vm_neck_std": std_neck,
        "vm_back_mean": float(vm_back.mean()),
        "vm_back_std": std_back,
        "gyro_neck_mean": float(gm_neck.mean()),
        "sma_neck": float(sma_neck.mean()),
        "sma_back": float(sma_back.mean()),
        "zcr_neck": crossings / max(n - 1, 1),
        "pitch_neck_mean": float(pitch_neck.mean()),
        "pitch_var": float(np.std(pitch_neck, ddof=1)),
        "roll_var": float(np.std(roll_neck, ddof=1)),
        "yaw_mean": float(yaw.mean()),
        "yaw_abs_mean": float(np.abs(yaw).mean()),
        "yaw_consistency": (
            float(abs(yaw.mean()) / np.abs(yaw).mean()) if np.abs(yaw).mean() else 0.0
        ),
        "neck_back_corr": float(np.corrcoef(vm_neck, vm_back)[0, 1]),
        "neck_dominance": std_neck / std_back if std_back else float("inf"),
        "activity_index": float((sma_neck.mean() + sma_back.mean()) / 2.0),
    }


def rules_state(f: dict[str, float], p: dict[str, float]) -> str:
    """ML.V_RULES_STATE from warehouse/05_ml_classification.sql, in order."""
    if (f["vm_neck_std"] < p["rules_rest_vm_std_max"]
            and f["pitch_var"] < p["rules_rest_pitch_var_max"]):
        return "REST"
    if (f["vm_neck_std"] > p["shake_vm_std_min"]
            and f["neck_back_corr"] < p["shake_corr_max"]
            and f["neck_dominance"] > p["neck_dominance_min"]):
        return "SHAKE"
    if (p["scratch_vm_std_min"] <= f["vm_neck_std"] <= p["scratch_vm_std_max"]
            and f["neck_back_corr"] < p["scratch_corr_max"]
            and f["neck_dominance"] > p["neck_dominance_min"]):
        return "SCRATCH"
    if (f["pitch_neck_mean"] < p["rules_sniff_pitch_max"]
            and p["rules_sniff_vm_std_min"] <= f["vm_neck_std"] <= p["rules_sniff_vm_std_max"]):
        return "SNIFF"
    if (f["neck_back_corr"] > p["rules_gallop_corr_min"]
            and f["vm_neck_mean"] > p["rules_gallop_vm_min"]):
        return "GALLOP"
    if (f["neck_back_corr"] > p["rules_trot_corr_min"]
            and f["vm_neck_mean"] > p["rules_trot_vm_min"]):
        return "TROT"
    if (f["neck_back_corr"] > p["rules_walk_corr_min"]
            and f["vm_neck_mean"] > p["rules_walk_vm_min"]):
        return "WALK"
    return "STAND"


def ladder_state(
    f: dict[str, float],
    p: dict[str, float],
    *,
    has_shake: bool = False,
    has_scratch: bool = False,
    has_pace: bool = False,
    loco_before: bool = False,
    loco_after: bool = False,
) -> tuple[str, str]:
    """MARTS.EPOCH_STATES from warehouse/06_marts_dt.sql. Returns (state, source).

    Precedence is identical to the SQL, deliberately including the ORDER of the
    rungs — the order is the design, and a reordering here would make the test
    pass while the warehouse did something else.
    """
    model = rules_state(f, p)
    dyn_back = abs(f["vm_back_mean"] - p["gravity_ref"])

    # rung 0 — quality
    if f["n_samples"] < p["epoch_min_samples"] or np.isnan(f["neck_back_corr"]):
        return "UNKNOWN", "LOW_QUALITY"

    # rung 1 — geometry
    if (f["yaw_consistency"] >= p["circle_yaw_consistency_min"]
            and f["yaw_abs_mean"] >= p["circle_yaw_activity_min"]
            and dyn_back <= p["circle_translation_max"]):
        return "CIRCLE", "GEOMETRY"
    if (not has_pace
            and model in ("WALK", "TROT")
            and f["yaw_consistency"] <= p["pace_yaw_consistency_max"]
            and f["yaw_abs_mean"] >= p["pace_yaw_activity_min"]):
        return "PACE", "GEOMETRY"
    if (model in ("REST", "SIT", "STAND")
            and f["pitch_var"] >= p["slowrise_pitch_var_min"]
            and f["vm_neck_std"] <= p["slowrise_vm_std_max"]):
        return "SLOW_TRANSITION", "GEOMETRY"

    # rung 2 — neck-dominant, only where the labels do not exist
    if (not has_shake
            and f["vm_neck_std"] > p["shake_vm_std_min"]
            and f["neck_back_corr"] < p["shake_corr_max"]
            and f["neck_dominance"] > p["neck_dominance_min"]):
        return "SHAKE", "HEURISTIC"
    if (not has_scratch
            and p["scratch_vm_std_min"] <= f["vm_neck_std"] <= p["scratch_vm_std_max"]
            and f["neck_back_corr"] < p["scratch_corr_max"]
            and f["neck_dominance"] > p["neck_dominance_min"]):
        return "SCRATCH", "HEURISTIC"

    # rung 3 — stillness bracketed by locomotion
    if (model in ("STAND", "SIT", "WALK")
            and f["vm_neck_std"] < p["pause_vm_std_max"]
            and loco_before and loco_after):
        return "PAUSE", "CONTEXT"

    # rung 4 — the classifier
    return model, "RULES"
