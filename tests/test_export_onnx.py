import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
onnxruntime = pytest.importorskip("onnxruntime")

from fakeradar.config import get_preset  # noqa: E402
from fakeradar.models import FakeRadarModel, export_onnx, verify_onnx_parity  # noqa: E402

_TORCH_OK = tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 9)


@pytest.mark.skipif(not _TORCH_OK, reason="ONNX export needs torch>=2.9 (dynamo exporter)")
def test_fast_tier_exports_with_parity(tmp_path):
    model = FakeRadarModel(get_preset("fast")).eval()
    path = export_onnx(model, tmp_path / "fast.onnx", crop_size=128)
    assert path.exists()
    diff = verify_onnx_parity(model, path, crop_size=128)  # raises if > atol
    assert diff <= 1e-4

    # dynamic batch: a different batch size must run through the same graph
    import numpy as np

    sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"crop": np.random.rand(3, 3, 128, 128).astype(np.float32)})[0]
    assert out.shape == (3,)
    assert ((out >= 0) & (out <= 1)).all()


def test_resized_branch_blocks_export():
    model = FakeRadarModel(get_preset("fast"))
    stub = torch.nn.Module()  # stand-in: any resized-input branch blocks export
    stub.input_kind = "resized"
    model.branches["stub"] = stub
    with pytest.raises(ValueError, match="crop-only"):
        export_onnx(model, "unused.onnx")


def _flat_crops(size: int):
    """Crops with large flat areas (sky, walls): where the export once broke."""
    x = torch.full((2, 3, size, size), 0.5)
    x[0, :, :, size // 2 :] = 0.8  # one vertical edge, flat on both sides
    x[1, :, size // 3 : size // 2, :] = 0.2  # a band, flat elsewhere
    return x


@pytest.mark.skipif(not _TORCH_OK, reason="ONNX export needs torch>=2.9 (dynamo exporter)")
def test_export_matches_torch_on_flat_regions(tmp_path):
    import numpy as np

    torch.manual_seed(0)
    model = FakeRadarModel(get_preset("fast")).eval()
    path = export_onnx(model, tmp_path / "fast.onnx", crop_size=128)
    x = _flat_crops(128)
    with torch.no_grad():
        ref = torch.sigmoid(model(x)["logit"] / model.fusion.temperature).numpy()
    sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"crop": x.numpy()})[0]
    assert np.abs(got - ref).max() <= 1e-4


def test_coherence_is_one_on_flat_regions_as_trained():
    from fakeradar.branches.gradient import GradientFieldBranch

    # zero padding makes artificial edges near the border, so check the interior
    stack = GradientFieldBranch().forensic_stack(torch.full((1, 3, 64, 64), 0.5))[0, :, 8:-8, 8:-8]
    assert torch.allclose(stack[16], torch.ones(48, 48), atol=1e-5)  # coherence channel
    assert torch.allclose(stack[15], torch.full((48, 48), 1e-6), atol=1e-9)  # gradient magnitude
