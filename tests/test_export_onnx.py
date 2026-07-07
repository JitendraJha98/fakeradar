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
