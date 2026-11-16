import numpy as np
import pytest
import torch
from lenslaw.blur_net import (
    LOGVAR_RANGE,
    MAX_STD,
    STRIDE,
    BlurNet,
    net_points,
    nll_loss,
    predict_cells,
)
from lenslaw.fit import SCORE_KEYS
from lenslaw.score import score_arrays
from lenslaw.synth import make_training_patch


class StubNet(torch.nn.Module):
    """Predicts log blur 1.0 everywhere; confident on the left half, unsure on the right."""

    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.zeros(b, 2, h // STRIDE, w // STRIDE)
        out[:, 0] = 1.0
        out[:, 1] = 2 * np.log(0.1)  # std 0.1
        out[:, 1, :, out.shape[3] // 2 :] = 2 * np.log(1.0)  # std 1.0
        return out


def texture(seed=0, size=260):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 0.5, np.float32)
    for _ in range(60):
        h, w = rng.integers(16, 80, 2)
        y, x = rng.integers(0, size - 8, 2)
        img[y : y + h, x : x + w] = rng.uniform(0.05, 0.95, 3)
    return img


def test_stride_is_one_grid_cell():
    assert STRIDE == 32


def test_network_outputs_mean_and_log_variance_per_cell():
    out = BlurNet()(torch.rand(2, 3, 96, 96))
    assert out.shape == (2, 2, 3, 3)
    assert out[:, 1].min() >= LOGVAR_RANGE[0] and out[:, 1].max() <= LOGVAR_RANGE[1]


def test_loss_rewards_accuracy_and_honest_uncertainty():
    target = torch.tensor([1.0])

    def pred(mean, logvar):
        return torch.tensor([[[[mean]], [[logvar]]]])

    assert nll_loss(pred(1.0, 0.0), target) < nll_loss(pred(2.0, 0.0), target)
    assert nll_loss(pred(3.0, 2.0), target) < nll_loss(pred(3.0, -4.0), target)


def test_network_learns_to_tell_sharp_from_blurred_patches():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = [make_training_patch(texture(i % 6), rng, out=96) for i in range(48)]
    x = torch.from_numpy(np.stack([p for p, _ in data])).permute(0, 3, 1, 2)
    y = torch.tensor([label for _, label in data], dtype=torch.float32)
    model = BlurNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for step in range(30):
        idx = torch.arange(16) + 16 * (step % 3)
        loss = nll_loss(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-3:]) < 0.7 * np.mean(losses[:3])


def test_predict_cells_returns_one_value_per_full_cell():
    log_sigma, std = predict_cells(StubNet(), np.zeros((100, 140, 3), np.float32))
    assert log_sigma.shape == (3, 4) and std.shape == (3, 4)
    assert np.allclose(log_sigma, 1.0)
    assert np.allclose(std[:, :2], 0.1) and np.allclose(std[:, 2:], 1.0)


def test_points_drop_cells_the_network_is_unsure_about():
    assert MAX_STD == 0.5
    pts = net_points(
        StubNet(), np.zeros((128, 256, 3), np.float32), np.zeros((128, 256), np.float32)
    )
    assert len(pts.d) == 4 * 4  # left half only: 4 rows x 4 of 8 columns
    assert (pts.yx[:, 1] < 128).all()
    assert np.allclose(pts.b, 1.0)
    assert ((pts.w > 0) & (pts.w <= 1)).all()


def test_points_drop_cells_that_straddle_a_depth_edge():
    depth = np.zeros((128, 256), np.float32)
    depth[:, 48:] = 1.0  # step inside the second column of cells
    pts = net_points(StubNet(), np.zeros((128, 256, 3), np.float32), depth)
    assert len(pts.d) == 4 * 3
    assert not ((pts.yx[:, 1] > 32) & (pts.yx[:, 1] < 64)).any()


def test_saved_weights_reproduce_the_same_predictions(tmp_path):
    torch.manual_seed(1)
    model = BlurNet().eval()
    img = texture(3, 128)
    before = predict_cells(model, img)
    torch.save(model.state_dict(), tmp_path / "w.pt")
    clone = BlurNet()
    clone.load_state_dict(torch.load(tmp_path / "w.pt", weights_only=True))
    after = predict_cells(clone, img)
    assert np.allclose(before[0], after[0]) and np.allclose(before[1], after[1])


def test_scoring_can_use_the_network_instead_of_the_edge_estimator():
    out = score_arrays(
        np.zeros((256, 512, 3), np.float32), np.zeros((256, 512), np.float32), blur_model=StubNet()
    )
    assert set(out) == set(SCORE_KEYS) | {"applicable"}
    assert out["n"] == 8 * 8  # confident half of a 8 x 16 grid
    assert out["applicable"] is False  # constant blur: no depth of field
    assert out["blur_spread"] == pytest.approx(1.0)
