"""Config invariants — most importantly: configs/*.yaml must mirror the
built-in presets in config.py (the presets are the source of truth)."""

from pathlib import Path

import pytest

from fakeradar.config import DetectorConfig, get_preset

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("tier", ["fast", "accurate"])
def test_yaml_matches_builtin_preset(tier):
    yaml_cfg = DetectorConfig.from_yaml(CONFIG_DIR / f"{tier}.yaml")
    assert yaml_cfg.to_dict() == get_preset(tier).to_dict(), (
        f"configs/{tier}.yaml has drifted from the {tier!r} preset in config.py — "
        "keep both in sync (see CLAUDE.md)."
    )


def test_dict_roundtrip():
    cfg = get_preset("accurate")
    again = DetectorConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_enabled_branches():
    fast = get_preset("fast")
    assert [b.name for b in fast.enabled_branches()] == ["gradient", "frequency"]
    accurate = get_preset("accurate")
    assert [b.name for b in accurate.enabled_branches()] == ["gradient", "frequency", "semantic"]


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown tier"):
        get_preset("turbo")
