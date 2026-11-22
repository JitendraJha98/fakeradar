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


def test_unknown_top_level_key_raises():
    d = get_preset("fast").to_dict()
    d["max_crop"] = 6  # typo must not silently fall back to the default
    with pytest.raises(ValueError, match="Unknown detector config keys"):
        DetectorConfig.from_dict(d)


@pytest.mark.parametrize(
    "override",
    [
        {"max_crops": 0},
        {"max_crops": 10},
        {"crop_size": 16},
        {"threshold_real": 0.7, "threshold_fake": 0.6},
        {"threshold_fake": 1.5},
    ],
)
def test_invalid_values_raise(override):
    d = {**get_preset("fast").to_dict(), **override}
    with pytest.raises(ValueError):
        DetectorConfig.from_dict(d)


@pytest.mark.parametrize("name", ["train_fast.yaml", "train_accurate.yaml"])
def test_training_configs_resolve(name):
    """Every shipped training config points at a loadable detector config."""
    import yaml

    from fakeradar.training.train import resolve_detector_config

    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
    det = resolve_detector_config(cfg["detector"], CONFIG_DIR)
    assert det.tier in {"fast", "accurate"}
    assert {"train_manifest", "val_manifest"} <= set(cfg["data"])


def test_detector_spec_preset_name_and_inline(tmp_path):
    from fakeradar.training.train import resolve_detector_config

    assert resolve_detector_config("accurate", tmp_path).tier == "accurate"
    inline = resolve_detector_config(get_preset("fast").to_dict(), tmp_path)
    assert inline.to_dict() == get_preset("fast").to_dict()
    with pytest.raises(FileNotFoundError):
        resolve_detector_config("nope.yaml", tmp_path)


def test_changing_a_preset_copy_does_not_leak_into_the_builtin_preset():
    from fakeradar.config import get_preset

    cfg = get_preset("fast")
    cfg.branches[0].params["use_coherence"] = False
    cfg.fusion.dropout = 0.9
    fresh = get_preset("fast")
    assert "use_coherence" not in fresh.branches[0].params
    assert fresh.fusion.dropout == 0.1
