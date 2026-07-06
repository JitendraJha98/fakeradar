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
