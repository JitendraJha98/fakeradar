import numpy as np
from fprcal.power import detection_rates


def real(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(10):
        for platform, shift in (("orig", 0.0), ("whatsapp", 2.0), ("telegram", 0.0)):
            for s in rng.normal(shift, 1.0, 200):
                out.append(
                    {
                        "phone": f"D{p}",
                        "role": "cal" if p < 6 else "audit",
                        "platform": platform,
                        "score": float(s),
                    }
                )
    return out


def fake(seed=1):
    rng = np.random.default_rng(seed)
    return [
        {"generator": g, "condition": c, "score": float(s)}
        for g, c, mu in (("a", "clean", 4.0), ("a", "s050_q70", 2.0), ("b", "clean", 0.0))
        for s in rng.normal(mu, 1.0, 300)
    ]


def test_rates_are_reported_per_generator_condition_and_threshold():
    out = detection_rates(real(), fake(), alpha=0.05)
    assert set(out) == {"a/clean", "a/s050_q70", "b/clean"}
    assert set(out["a/clean"]) == {"orig", "whatsapp", "pooled", "n"}
    assert out["a/clean"]["n"] == 300


def test_a_strong_generator_signal_is_detected_and_a_null_one_is_not():
    out = detection_rates(real(), fake(), alpha=0.05)
    assert out["a/clean"]["orig"] > 0.95
    assert out["b/clean"]["orig"] < 0.12


def test_a_stricter_stratum_threshold_detects_less():
    out = detection_rates(real(), fake(), alpha=0.05)
    assert out["a/s050_q70"]["whatsapp"] < out["a/s050_q70"]["orig"]


def test_held_out_platform_and_audit_phones_set_no_threshold():
    rows = real()
    for r in rows:
        if r["role"] == "audit" or r["platform"] == "telegram":
            r["score"] = 50.0
    out = detection_rates(rows, fake(), alpha=0.05)
    assert "telegram" not in out["a/clean"]
    assert out["a/clean"]["orig"] > 0.95
