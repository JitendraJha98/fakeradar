import numpy as np
import pytest
from lenslaw.blur_classic import BlurMap
from lenslaw.patches import MIN_BLUR_SPREAD, MIN_CELLS, Points, applicable, collect_points


def full_blur(h, w, value=2.0):
    return BlurMap(np.full((h, w), value, np.float32), np.ones((h, w), np.float32))


def points_with(n, spread):
    b = np.concatenate([np.zeros(n // 2), np.full(n - n // 2, np.log(spread))])
    return Points(np.linspace(0, 1, n), b, np.ones(n), np.zeros((n, 2)))


def test_every_full_cell_with_enough_edge_pixels_gives_one_point():
    pts = collect_points(full_blur(64, 96), np.zeros((64, 96), np.float32), cell=32)
    assert len(pts.d) == 6
    assert np.allclose(pts.b, np.log(2.0))
    assert np.allclose(pts.w, 1.0)
    assert pts.yx.shape == (6, 2)


def test_cell_straddling_a_depth_step_is_excluded():
    depth = np.zeros((64, 128), np.float32)
    depth[:, 48:] = 1.0  # the step falls inside the cells spanning columns 32..63
    pts = collect_points(full_blur(64, 128), depth, cell=32)
    assert len(pts.d) == 6
    assert not ((pts.yx[:, 1] > 32) & (pts.yx[:, 1] < 64)).any()


def test_cell_with_too_few_edge_pixels_is_excluded_and_weight_follows_the_count():
    sigma = np.full((32, 64), np.nan, np.float32)
    sigma[0, :11] = 2.0  # cell 0: 11 estimates, one short of the minimum
    sigma[0, 32:44] = 2.0  # cell 1: 12 estimates
    pts = collect_points(BlurMap(sigma, np.ones_like(sigma)), np.zeros((32, 64), np.float32))
    assert len(pts.d) == 1
    assert pts.yx[0, 1] > 32
    assert pts.w[0] == pytest.approx(12 / 50)


def test_point_is_the_median_log_blur_and_median_depth_of_its_cell():
    sigma = np.full((32, 32), np.nan, np.float32)
    sigma[0, :20] = 1.0
    sigma[1, :21] = 4.0
    depth = np.full((32, 32), 0.25, np.float32)
    pts = collect_points(BlurMap(sigma, np.ones_like(sigma)), depth)
    assert pts.b[0] == pytest.approx(np.log(4.0))
    assert pts.d[0] == pytest.approx(0.25)


def test_constant_depth_map_does_not_divide_by_zero():
    pts = collect_points(full_blur(64, 64), np.full((64, 64), 0.3, np.float32))
    assert len(pts.d) == 4
    assert np.allclose(pts.d, 0.3)


def test_no_estimates_gives_empty_points_that_are_not_applicable():
    sigma = np.full((64, 64), np.nan, np.float32)
    pts = collect_points(BlurMap(sigma, np.zeros_like(sigma)), np.zeros((64, 64), np.float32))
    assert len(pts.d) == 0 and pts.yx.shape == (0, 2)
    assert applicable(pts) is False


def test_depth_and_blur_maps_of_different_shapes_are_rejected():
    with pytest.raises(ValueError):
        collect_points(full_blur(64, 64), np.zeros((32, 32), np.float32))


def test_applicability_needs_enough_cells_and_enough_blur_spread():
    assert MIN_CELLS == 60 and MIN_BLUR_SPREAD == 2.0
    assert applicable(points_with(59, 4.0)) is False
    assert applicable(points_with(60, 1.9)) is False
    assert applicable(points_with(60, 2.1)) is True
