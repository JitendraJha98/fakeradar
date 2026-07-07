import pytest

torch = pytest.importorskip("torch")

from fakeradar.fusion import FusionHead  # noqa: E402


def _head():
    return FusionHead({"a": 8, "b": 16}, embed_dim=32, hidden_dim=64)


def test_forward_shapes_and_gate_simplex():
    head = _head().eval()
    feats = {"a": torch.rand(4, 8), "b": torch.rand(4, 16)}
    out = head(feats)
    assert out["logit"].shape == (4,)
    assert set(out["branch_logits"]) == {"a", "b"}
    assert all(v.shape == (4,) for v in out["branch_logits"].values())
    gates = torch.stack(list(out["gate_weights"].values()))
    assert torch.allclose(gates.sum(), torch.tensor(1.0), atol=1e-6)
    assert (gates >= 0).all()


def test_probability_applies_temperature():
    head = _head()
    head.temperature.fill_(2.0)
    logit = torch.tensor([4.0])
    assert torch.allclose(head.probability(logit), torch.sigmoid(logit / 2.0))


def test_fit_temperature_recovers_known_miscalibration():
    torch.manual_seed(0)
    true_t = 3.0
    logits = torch.randn(4000) * 4.0
    labels = torch.bernoulli(torch.sigmoid(logits / true_t))
    head = _head()
    fitted = head.fit_temperature(logits, labels)
    assert 2.0 < fitted < 4.5, f"expected T near {true_t}, got {fitted}"
    assert float(head.temperature) == pytest.approx(fitted)


def test_fit_temperature_degenerate_input_stays_sane():
    head = _head()
    # Single-class input pushes T toward 0; fitting must not crash and the
    # result must stay inside the clamp range [0.05, 20].
    fitted = head.fit_temperature(torch.full((64,), 5.0), torch.ones(64))
    assert 0.05 <= fitted < 1.0
