"""Fits of log blur against inverse depth, and the lens-law score.

A thin lens gives blur size proportional to |D - D_focus|, where D is inverse
depth. With a blur floor c0 the model for log blur is

    b(D) = 0.5 * log(c0**2 + k**2 * (D - D_focus)**2)

Depth is normalised inside the fit, so the result does not depend on the
unknown scale, offset or sign of a depth network's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HUBER_DELTA = 0.25  # log-blur units; residuals beyond this count linearly in the fit
FOCUS_RANGE = (-0.5, 1.5)  # search range for the focus plane, in normalised depth
_MIN_LOG = -30.0

SCORE_KEYS = (
    "n",
    "blur_spread",
    "depth_spread",
    "resid_const",
    "resid_lens",
    "r2_lens",
    "resid_valley",
    "r2_valley",
    "c0",
    "k",
    "d_focus",
)


@dataclass(frozen=True)
class FitResult:
    resid: float  # weighted RMS residual, log-blur units
    r2: float  # share of the constant-fit variance removed (can be negative)
    params: dict[str, float] = field(default_factory=dict)


def _clean(d, b, w) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    keep = np.isfinite(d) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    return d[keep], b[keep], w[keep]


def _const_var(b: np.ndarray, w: np.ndarray) -> float:
    mean = np.sum(w * b) / np.sum(w)
    return float(np.sum(w * (b - mean) ** 2) / np.sum(w))


def _r2(resid: float, var_const: float) -> float:
    return 0.0 if var_const <= 1e-12 else float(1.0 - resid**2 / var_const)


def _nan_result(**params: float) -> FitResult:
    return FitResult(float("nan"), float("nan"), params)


def fit_constant(b, w) -> FitResult:
    b = np.asarray(b, dtype=np.float64).ravel()
    _, b, w = _clean(np.zeros_like(b), b, w)
    if b.size == 0:
        return _nan_result(b0=float("nan"))
    return FitResult(float(np.sqrt(_const_var(b, w))), 0.0, {"b0": float(np.sum(w * b) / w.sum())})


# ------------------------------------------------------------------ thin lens
def _huber(r: np.ndarray) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= HUBER_DELTA, 0.5 * r * r, HUBER_DELTA * (a - 0.5 * HUBER_DELTA))


def _solve_floor_and_slope(u, b, w, alpha, beta, iters: int = 25):
    """Robust fit of b ~ 0.5*log(exp(alpha) + exp(beta)*u), one fit per row of u.

    u: (F, N) squared distance from each candidate focus plane. Damped
    Gauss-Newton on (alpha, beta) with Huber weights. Returns alpha, beta and
    the Huber objective, each of shape (F,).
    """
    for _ in range(iters):
        a_term = np.exp(alpha)[:, None]
        total = a_term + np.exp(beta)[:, None] * u
        r = b[None, :] - 0.5 * np.log(total)
        p = a_term / total  # d(pred)/d(alpha) = p/2, d(pred)/d(beta) = (1-p)/2
        q = 1.0 - p
        omega = w[None, :] * np.minimum(1.0, HUBER_DELTA / np.maximum(np.abs(r), 1e-12))
        h11 = 0.25 * np.sum(omega * p * p, axis=1)
        h22 = 0.25 * np.sum(omega * q * q, axis=1)
        h12 = 0.25 * np.sum(omega * p * q, axis=1)
        g1 = 0.5 * np.sum(omega * p * r, axis=1)
        g2 = 0.5 * np.sum(omega * q * r, axis=1)
        damp = 1e-3 * (h11 + h22) + 1e-12
        h11 = h11 + damp
        h22 = h22 + damp
        det = h11 * h22 - h12 * h12
        alpha = np.maximum(alpha + np.clip((h22 * g1 - h12 * g2) / det, -1.5, 1.5), _MIN_LOG)
        beta = np.maximum(beta + np.clip((h11 * g2 - h12 * g1) / det, -1.5, 1.5), _MIN_LOG)
    pred = 0.5 * np.log(np.exp(alpha)[:, None] + np.exp(beta)[:, None] * u)
    objective = np.sum(w[None, :] * _huber(b[None, :] - pred), axis=1)
    return alpha, beta, objective


def fit_thin_lens(d, b, w, n_grid: int = 61) -> FitResult:
    d, b, w = _clean(d, b, w)
    if b.size == 0:
        return _nan_result(c0=float("nan"), k=float("nan"), d_focus=float("nan"))
    var_const = _const_var(b, w)
    d_min, span = float(d.min()), float(d.max() - d.min())
    if span <= 1e-12:  # one depth only: the law says nothing, blur is a constant
        params = {"c0": float(np.exp(np.sum(w * b) / w.sum())), "k": 0.0, "d_focus": d_min}
        return FitResult(float(np.sqrt(var_const)), 0.0, params)

    x = (d - d_min) / span
    lo, hi = np.percentile(b, 10), np.percentile(b, 90)

    def search(focus: np.ndarray, alpha0: float | None = None, beta0: float | None = None):
        u = (x[None, :] - focus[:, None]) ** 2
        if alpha0 is None:
            alpha = np.full(focus.size, 2.0 * lo)
            scale = np.maximum(np.percentile(u, 90, axis=1), 1e-6)
            beta = np.log(np.maximum(np.exp(2.0 * hi) - np.exp(2.0 * lo), 1e-6) / scale)
        else:
            alpha = np.full(focus.size, alpha0)
            beta = np.full(focus.size, beta0)
        alpha, beta, objective = _solve_floor_and_slope(u, b, w, alpha, beta)
        best = int(np.argmin(objective))
        return float(focus[best]), float(alpha[best]), float(beta[best])

    coarse = np.linspace(FOCUS_RANGE[0], FOCUS_RANGE[1], n_grid)
    focus, alpha, beta = search(coarse)
    step = (FOCUS_RANGE[1] - FOCUS_RANGE[0]) / (n_grid - 1)
    fine = np.clip(np.linspace(focus - step, focus + step, 21), *FOCUS_RANGE)
    focus, alpha, beta = search(fine, alpha, beta)

    pred = 0.5 * np.log(np.exp(alpha) + np.exp(beta) * (x - focus) ** 2)
    resid = float(np.sqrt(np.sum(w * (b - pred) ** 2) / w.sum()))
    params = {
        "c0": float(np.exp(0.5 * alpha)),
        "k": float(np.exp(0.5 * beta) / span),
        "d_focus": float(d_min + focus * span),
    }
    return FitResult(resid, _r2(resid, var_const), params)


# --------------------------------------------------------------------- valley
def _prefix_decreasing_sse(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sse[i] = weighted SSE of the best non-increasing fit to the first i points."""
    sse = np.zeros(y.size + 1)
    sw: list[float] = []
    swy: list[float] = []
    swyy: list[float] = []
    total = 0.0
    for i in range(y.size):
        cw, cwy, cwyy = w[i], w[i] * y[i], w[i] * y[i] * y[i]
        while sw and swy[-1] / sw[-1] < cwy / cw:  # previous block lower: pool them
            pw, pwy, pwyy = sw.pop(), swy.pop(), swyy.pop()
            total -= pwyy - pwy * pwy / pw
            cw, cwy, cwyy = cw + pw, cwy + pwy, cwyy + pwyy
        sw.append(cw)
        swy.append(cwy)
        swyy.append(cwyy)
        total += cwyy - cwy * cwy / cw
        sse[i + 1] = max(total, 0.0)
    return sse


def fit_valley(d, b, w) -> FitResult:
    """Best fit that falls and then rises with depth, of any shape."""
    d, b, w = _clean(d, b, w)
    if b.size == 0:
        return _nan_result(d_min=float("nan"))
    var_const = _const_var(b, w)
    order = np.argsort(d, kind="stable")
    d, w = d[order], w[order]
    y = b[order] - np.sum(w * b[order]) / w.sum()  # centred for numerical stability
    falling = _prefix_decreasing_sse(y, w)
    rising = _prefix_decreasing_sse(y[::-1], w[::-1])[::-1]  # rising[i]: points i..n-1
    total = falling + rising
    cut = int(np.argmin(total))
    resid = float(np.sqrt(total[cut] / w.sum()))
    return FitResult(resid, _r2(resid, var_const), {"d_min": float(d[min(cut, d.size - 1)])})


# ---------------------------------------------------------------------- score
def score_points(d, b, w) -> dict[str, float]:
    """All lens-law numbers for one image's (inverse depth, log blur) points."""
    d, b, w = _clean(d, b, w)
    out = dict.fromkeys(SCORE_KEYS, float("nan"))
    out["n"] = int(b.size)
    if b.size < 3:
        return out
    span = float(d.max() - d.min())
    x = (d - d.min()) / span if span > 1e-12 else np.zeros_like(d)
    lens = fit_thin_lens(d, b, w)
    valley = fit_valley(d, b, w)
    out.update(
        blur_spread=float(np.exp(np.percentile(b, 90) - np.percentile(b, 10))),
        depth_spread=float(np.percentile(x, 90) - np.percentile(x, 10)),
        resid_const=float(np.sqrt(_const_var(b, w))),
        resid_lens=lens.resid,
        r2_lens=lens.r2,
        resid_valley=valley.resid,
        r2_valley=valley.r2,
        c0=lens.params["c0"],
        k=lens.params["k"],
        d_focus=lens.params["d_focus"],
    )
    return out
