import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from fakeradar.training.datasets import ManifestDataset, read_manifest  # noqa: E402


def _write_images(tmp_path, n=4, size=(80, 70)):
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        Image.effect_noise(size, 30 + i).convert("RGB").save(p)
        paths.append(p)
    return paths


def test_read_manifest_with_and_without_generator(tmp_path):
    paths = _write_images(tmp_path, 2)
    full = tmp_path / "full.csv"
    full.write_text(f"path,label,generator\n{paths[0]},0,camera\n{paths[1]},1,sdv14\n")
    rows = read_manifest(full)
    assert [r["label"] for r in rows] == [0, 1]
    assert rows[1]["generator"] == "sdv14"

    bare = tmp_path / "bare.csv"
    bare.write_text(f"path,label\n{paths[0]},0\n{paths[1]},1\n")
    rows = read_manifest(bare)
    assert all(r["generator"] == "unknown" for r in rows)


def test_read_manifest_empty_raises(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("path,label,generator\n")
    with pytest.raises(ValueError, match="Empty manifest"):
        read_manifest(empty)


def test_dataset_item_shapes_and_dtypes(tmp_path):
    paths = _write_images(tmp_path, 2, size=(90, 60))
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"path,label\n{paths[0]},0\n{paths[1]},1\n")
    ds = ManifestDataset(manifest, crop_size=64, augment=False)
    assert len(ds) == 2
    item = ds[1]
    assert item["crop"].shape == (3, 64, 64)
    assert item["label"].dtype == torch.float32 and item["label"].item() == 1.0
    assert "semantic" not in item


def test_dataset_augmented_still_correct_shape(tmp_path):
    paths = _write_images(tmp_path, 1, size=(70, 70))
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"path,label\n{paths[0]},1\n")
    ds = ManifestDataset(manifest, crop_size=64, augment=True)
    assert ds[0]["crop"].shape == (3, 64, 64)
