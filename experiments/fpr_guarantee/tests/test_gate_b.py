import numpy as np
import pytest
from fprcal.gate_b import GATE_B, gate_b

PLATFORMS = ("orig", "facebook", "instagram", "telegram", "twitter", "whatsapp")


def records(seed=0, n_phones=27, per=60, shift=None):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_phones):
        role = "cal" if p < 16 else "audit"
        phone_effect = rng.normal(0, 0.1)
        for platform in PLATFORMS:
            for s in rng.normal(phone_effect, 1.0, per):
                bump = shift[1] if shift and role == "audit" and platform == shift[0] else 0.0
                out.append(
                    {
                        "phone": f"D{p:02d}",
                        "role": role,
                        "platform": platform,
                        "score": float(s + bump),
                    }
                )
    return out


def test_pass_rule_is_the_one_fixed_in_the_plan():
    assert GATE_B == {
        "levels": (0.05, 0.01),
        "ratio": 1.5,
        "max_failed": 1,
        "hard_ratio": 2.0,
        "bootstrap": 2000,
        "held_out": "telegram",
    }


def test_exchangeable_phones_pass_the_gate():
    result = gate_b(records(), n_boot=200)
    assert result["passed"] is True
    assert len(result["checks"]) == 10
    assert not any(k.startswith("telegram") for k in result["checks"])


def test_a_shifted_platform_fails_its_checks_and_the_gate():
    result = gate_b(records(shift=("facebook", 1.0)), n_boot=200)
    assert result["passed"] is False
    assert result["checks"]["facebook@0.05"]["passed"] is False
    assert result["checks"]["facebook@0.05"]["rate"] > 0.10
    assert result["checks"]["twitter@0.05"]["passed"] is True


def test_held_out_platform_is_reported_but_sets_no_threshold():
    result = gate_b(records(shift=("telegram", 1.0)), n_boot=200)
    assert result["passed"] is True
    assert result["held_out"]["platform"] == "telegram"
    assert result["held_out"]["rates"]["0.05"] > 0.10


def test_a_phone_in_both_roles_is_rejected():
    rows = records()
    rows[0] = dict(rows[0], role="audit")
    with pytest.raises(ValueError, match="D00"):
        gate_b(rows, n_boot=50)


def test_same_seed_gives_the_same_result():
    assert gate_b(records(), n_boot=100, seed=3) == gate_b(records(), n_boot=100, seed=3)
