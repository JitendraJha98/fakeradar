from pathlib import Path

import pytest

pytest.importorskip("torch")  # fakeradar.training imports the torch datasets

from PIL import Image  # noqa: E402

from fakeradar.training.datasets import read_manifest  # noqa: E402
from fakeradar.training.genimage import (  # noqa: E402
    build_manifests,
    discover_generators,
    find_split_dir,
)


def _fake_genimage(root: Path, n=20) -> Path:
    """Mimic the released archives: ROOT/<gen>/<subset>/<split>/{ai,nature}/."""
    layout = {
        "stable_diffusion_v_1_4": "imagenet_ai_0419_sdv4",
        "Midjourney": "imagenet_midjourney",
    }
    for gen, subset in layout.items():
        for split in ("train", "val"):
            for label_dir in ("ai", "nature"):
                d = root / gen / subset / split / label_dir
                d.mkdir(parents=True)
                for i in range(n):
                    ext = ".JPEG" if label_dir == "nature" else ".png"  # ImageNet naming
                    Image.new("RGB", (8, 8)).save(d / f"{i}{ext}", format="PNG")
    return root


def test_nested_and_flat_layouts_found(tmp_path):
    root = _fake_genimage(tmp_path / "GenImage")
    assert find_split_dir(root, "Midjourney", "val").parts[-2:] == ("imagenet_midjourney", "val")
    (root / "flat" / "val").mkdir(parents=True)
    assert find_split_dir(root, "flat", "val") == root / "flat" / "val"
    assert find_split_dir(root, "missing", "val") is None
    assert discover_generators(root) == ["Midjourney", "flat", "stable_diffusion_v_1_4"]


def test_build_manifests_protocol(tmp_path):
    root = _fake_genimage(tmp_path / "GenImage")
    out = tmp_path / "manifests"
    res = build_manifests(
        root, ["stable_diffusion_v_1_4"], ["Midjourney", "stable_diffusion_v_1_4"], out
    )

    # default names are exactly what configs/train_fast.yaml reads
    assert res["train"][0].name == "genimage_stable_diffusion_v_1_4_train.csv"
    assert res["val"][0].name == "genimage_stable_diffusion_v_1_4_val.csv"
    assert res["test"][0].name == "genimage_stable_diffusion_v_1_4_test.csv"

    train, val, test = (read_manifest(res[s][0]) for s in ("train", "val", "test"))
    assert len(train) + len(val) == 40  # 20 ai + 20 nature in SD1.4/train
    # hold-out val is stratified and carved from TRAIN, never from GenImage's val
    assert {r["label"] for r in val} == {0, 1}
    assert not {r["path"] for r in train} & {r["path"] for r in val}
    assert all("/train/" in Path(r["path"]).as_posix() for r in train + val)
    assert all("/val/" in Path(r["path"]).as_posix() for r in test)
    assert len(test) == 80
    assert {r["generator"] for r in test} == {"Midjourney", "stable_diffusion_v_1_4", "real"}
    assert all(Path(r["path"]).is_absolute() for r in train + val + test)


def test_build_manifests_all_and_errors(tmp_path):
    root = _fake_genimage(tmp_path / "GenImage", n=4)
    res = build_manifests(root, ["Midjourney"], ["all"], tmp_path / "m", name="mj")
    assert res["test"][0].name == "mj_test.csv"
    assert res["test"][1] == 2 * 2 * 4  # both generators x {ai, nature} x 4 images

    with pytest.raises(FileNotFoundError, match="nope"):
        build_manifests(root, ["nope"], None, tmp_path / "m")
    with pytest.raises(ValueError, match="val_fraction"):
        build_manifests(root, ["Midjourney"], None, tmp_path / "m", val_fraction=1.0)


def test_cli_prepare_genimage(tmp_path):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from fakeradar.cli import app

    root = _fake_genimage(tmp_path / "GenImage", n=4)
    out = tmp_path / "m"
    r = CliRunner().invoke(
        app,
        [
            "prepare-genimage",
            str(root),
            "-g",
            "stable_diffusion_v_1_4",
            "-t",
            "all",
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out / "genimage_stable_diffusion_v_1_4_test.csv").exists()

    r = CliRunner().invoke(app, ["prepare-genimage", str(root), "-g", "nope", "--out", str(out)])
    assert r.exit_code != 0 and "nope" in r.output
