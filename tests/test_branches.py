import pytest

torch = pytest.importorskip("torch")

from fakeradar.branches import build_branch  # noqa: E402


def test_gradient_branch_shapes():
    b = build_branch("gradient").eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out = b(x)
    assert out.shape == (2, b.feature_dim)
    stack = b.forensic_stack(x)
    assert stack.shape[:2] == (2, 17)


def test_frequency_branch_nonsquare():
    b = build_branch("frequency", n_bins=64).eval()
    x = torch.rand(3, 3, 200, 320)
    with torch.no_grad():
        out = b(x)
    assert out.shape == (3, b.feature_dim)
    prof, stats = b.radial_profile(x)
    assert prof.shape == (3, 64) and stats.shape == (3, 4)


def test_forensic_stack_is_fp32_under_autocast():
    """Regression: under CUDA AMP the stack used to be computed in fp16, where
    the coherence ratio drifts by O(1) vs fp32 inference -> train/test skew."""
    b = build_branch("gradient").eval()
    x = torch.rand(2, 3, 64, 64)
    x[:, :, :32, :] = 0.5  # flat region: near-zero structure-tensor terms
    ref = b.forensic_stack(x)
    with torch.autocast("cpu", dtype=torch.float16):
        got = b.forensic_stack(x)
    assert got.dtype == torch.float32
    assert torch.allclose(got, ref, atol=1e-5)


def test_resized_branch_contract():
    """Crop branches have no preprocessor; the base default is None."""
    assert build_branch("gradient").preprocessor() is None
    assert build_branch("frequency").preprocessor() is None


def test_coherence_channel_can_be_switched_off_for_the_ablation():
    from fakeradar.branches.gradient import GradientFieldBranch

    x = torch.rand(2, 3, 64, 64)
    with_it, without = GradientFieldBranch(), GradientFieldBranch(use_coherence=False)
    full, reduced = with_it.forensic_stack(x), without.forensic_stack(x)
    assert full.shape[1] == 17 and reduced.shape[1] == 16
    assert torch.equal(full[:, :16], reduced)  # only the last (coherence) map is dropped
    assert without(x).shape == (2, 512)


def test_coherence_switch_is_reachable_from_a_detector_config():
    from fakeradar.config import get_preset
    from fakeradar.models import FakeRadarModel

    cfg = get_preset("fast")
    cfg.branches[0].params["use_coherence"] = False
    model = FakeRadarModel(cfg)
    assert model.branches["gradient"].encoder.conv1.in_channels == 16
