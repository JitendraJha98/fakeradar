import itertools
import math

import numpy as np
from lenslaw.gate_m import THRESHOLDS, auroc, gate_m

UNLAWFUL = ("step", "position", "missing_foreground")


def rec(source, variant, condition="clean", split="dev", scene=None, **scores):
    base = {
        "id": f"{source}-{variant}-{condition}-{scene}",
        "source": source,
        "variant": variant,
        "condition": condition,
        "split": split,
        "scene": scene,
        "applicable": True,
        "n": 100,
        "blur_spread": 4.0,
        "resid_lens": 0.1,
        "r2_lens": 0.9,
    }
    base.update(scores)
    return base


def passing(split="dev"):
    """A record set on which every check passes with room to spare."""
    out = []
    for i in range(20):
        scene = f"s{i}"
        out.append(rec("pairs", "wide_depth_narrow", split=split, scene=scene))
        out.append(rec("pairs", "wide_depth_self", split=split, scene=scene))
        out.append(rec("pairs", "narrow_depth_self", split=split, scene=scene, blur_spread=1.2))
    for cond in ("clean", "s050_q70"):
        for i in range(40):
            out.append(rec("real_wide", "photo", cond, split, i, resid_lens=0.10 + 0.002 * i))
            out.append(rec("render", "lawful", cond, split, i, resid_lens=0.10 + 0.001 * i))
            for mode in UNLAWFUL:
                out.append(rec("render", mode, cond, split, i, resid_lens=0.40 + 0.001 * i))
    return out


def edit(records, n, **changes):
    """Apply `changes` to the first `n` records matching the non-score keys given."""
    match = {k: changes.pop(k) for k in ("source", "variant", "condition") if k in changes}
    hit = [r for r in records if all(r[k] == v for k, v in match.items())][:n]
    for r in hit:
        r.update(changes)
    assert len(hit) == n
    return records


def failed(result):
    return {k for k in ("M1", "M2", "M3", "M4", "M5") if not result[k]["passed"]}


def test_thresholds_are_the_ones_fixed_in_the_spec():
    assert THRESHOLDS == {
        "M1_r2": 0.5,
        "M1_share_depth_narrow": 0.80,
        "M1_share_depth_self": 0.70,
        "M2_share": 0.90,
        "M3_auroc": 0.85,
        "M4_fpr": 0.05,
        "M4_unlawful_flagged": 0.50,
        "M4_lawful_flagged": 0.10,
        "M5_condition": "s050_q70",
        "M5_auroc_drop": 0.10,
        "M5_unlawful_flagged": 0.35,
    }


def test_a_clean_record_set_passes_every_check():
    result = gate_m(passing(), "dev")
    assert failed(result) == set()
    assert result["passed"] is True


def test_m1_needs_eighty_percent_of_pairs_to_fit_with_depth_from_the_narrow_shot():
    at_limit = edit(passing(), 4, variant="wide_depth_narrow", r2_lens=0.4)
    below = edit(passing(), 5, variant="wide_depth_narrow", r2_lens=0.4)
    assert failed(gate_m(at_limit, "dev")) == set()
    assert failed(gate_m(below, "dev")) == {"M1"}


def test_m1_needs_seventy_percent_of_pairs_to_fit_with_depth_from_the_wide_shot():
    at_limit = edit(passing(), 6, variant="wide_depth_self", r2_lens=0.4)
    below = edit(passing(), 7, variant="wide_depth_self", r2_lens=0.4)
    assert failed(gate_m(at_limit, "dev")) == set()
    assert failed(gate_m(below, "dev")) == {"M1"}


def test_m1_ignores_pairs_without_visible_depth_of_field():
    records = edit(passing(), 10, variant="wide_depth_narrow", r2_lens=0.0, applicable=False)
    result = gate_m(records, "dev")
    assert failed(result) == set()
    assert result["M1"]["value"]["n_depth_narrow"] == 10


def test_m2_needs_the_wide_shot_to_show_more_blur_spread_in_ninety_percent_of_pairs():
    at_limit = edit(passing(), 2, variant="narrow_depth_self", blur_spread=9.0)
    below = edit(passing(), 3, variant="narrow_depth_self", blur_spread=9.0)
    assert failed(gate_m(at_limit, "dev")) == set()
    assert failed(gate_m(below, "dev")) == {"M2"}


def test_m3_fails_when_one_violation_type_is_not_separated_from_lawful_renders():
    records = passing()
    for r in records:
        if r["variant"] == "step" and r["condition"] == "clean":
            r["resid_lens"] = 0.10 + 0.001 * r["scene"]
    result = gate_m(records, "dev")
    assert failed(result) == {"M3"}
    assert result["M3"]["value"]["step"] == 0.5


def test_m4_fails_when_more_than_ten_percent_of_lawful_renders_are_flagged():
    at_limit = edit(passing(), 4, variant="lawful", condition="clean", resid_lens=0.5)
    above = edit(passing(), 5, variant="lawful", condition="clean", resid_lens=0.5)
    assert failed(gate_m(at_limit, "dev")) == set()
    assert failed(gate_m(above, "dev")) == {"M4"}


def test_m5_fails_when_transport_destroys_the_separation():
    records = passing()
    for r in records:
        if r["variant"] in UNLAWFUL and r["condition"] == "s050_q70":
            r["resid_lens"] = 0.10 + 0.001 * r["scene"]
    assert failed(gate_m(records, "dev")) == {"M5"}


def test_the_flagging_threshold_always_comes_from_dev_real_photos():
    records = passing("dev") + passing("test")
    for r in records:
        if r["split"] == "test" and r["source"] == "real_wide":
            r["resid_lens"] = 5.0  # would hide every violation if it set the threshold
    result = gate_m(records, "test")
    assert failed(result) == set()
    assert result["M4"]["value"]["threshold"] < 0.2


def test_auroc_counts_ties_as_half_and_matches_a_brute_force_count():
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 6, 15).astype(float)
    neg = rng.integers(0, 6, 12).astype(float)
    brute = np.mean(
        [1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg)]
    )
    assert math.isclose(auroc(pos, neg), brute)
    assert auroc([1.0, 1.0], [1.0]) == 0.5
    assert math.isnan(auroc([], [1.0]))
