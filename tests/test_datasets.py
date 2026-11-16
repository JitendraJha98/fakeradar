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


def test_read_manifest_validates_labels_and_columns(tmp_path):
    bad_label = tmp_path / "bad.csv"
    bad_label.write_text("path,label\na.png,2\n")
    with pytest.raises(ValueError, match="label must be 0"):
        read_manifest(bad_label)

    no_label = tmp_path / "nolabel.csv"
    no_label.write_text("path,generator\na.png,sd\n")
    with pytest.raises(ValueError, match="missing column"):
        read_manifest(no_label)

    blank_gen = tmp_path / "blank.csv"
    blank_gen.write_text("path,label,generator\na.png,1,\n")
    assert read_manifest(blank_gen)[0]["generator"] == "unknown"


def test_require_both_classes():
    from fakeradar.training.datasets import require_both_classes

    require_both_classes([{"label": 0}, {"label": 1}], "ok")
    with pytest.raises(ValueError, match="both real"):
        require_both_classes([{"label": 1}], "val manifest")


def test_corrupt_image_skipped_in_training_but_raises_in_eval(tmp_path):
    good = _write_images(tmp_path, 1, size=(80, 80))[0]
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"\x89PNG not really")
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"path,label\n{bad},1\n{good},0\n")

    train_ds = ManifestDataset(manifest, crop_size=64, augment=True)
    with pytest.warns(UserWarning, match="skipping unreadable"):
        item = train_ds[0]
    assert item["crop"].shape == (3, 64, 64)

    eval_ds = ManifestDataset(manifest, crop_size=64, augment=False)
    with pytest.raises(OSError, match="corrupt.png"):
        eval_ds[0]


def test_multi_crop_eval_matches_inference_grid(tmp_path):
    from fakeradar.preprocessing import extract_crops

    path = _write_images(tmp_path, 1, size=(300, 200))[0]
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"path,label\n{path},1\n")
    ds = ManifestDataset(manifest, crop_size=64, augment=False, max_crops=4)
    item = ds[0]
    assert "crop" not in item
    expected = extract_crops(Image.open(path).convert("RGB"), 64, 4)
    assert torch.equal(item["crops"], expected)

    with pytest.raises(ValueError, match="incompatible"):
        ManifestDataset(manifest, crop_size=64, augment=True, max_crops=4)


def test_equalizing_compression_changes_lossless_rows_only(tmp_path):
    png, jpg = tmp_path / "fake.png", tmp_path / "real.jpg"
    Image.effect_noise((96, 96), 60).convert("RGB").save(png)
    Image.effect_noise((96, 96), 60).convert("RGB").save(jpg, quality=92)
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"path,label\n{jpg},0\n{png},1\n")
    off = ManifestDataset(manifest, crop_size=64, augment=False)
    on = ManifestDataset(manifest, crop_size=64, augment=False, equalize_jpeg_quality=96)
    assert torch.equal(off[0]["crop"], on[0]["crop"])  # already a JPEG: untouched
    assert not torch.equal(off[1]["crop"], on[1]["crop"])  # PNG: one JPEG pass added
    assert on[1]["crop"].shape == (3, 64, 64)
