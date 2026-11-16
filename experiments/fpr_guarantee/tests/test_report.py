import json

import numpy as np
from fprcal.gate_b import gate_b
from fprcal.report import audit_rates, load_records, render_report

PLATFORMS = ("orig", "facebook", "instagram", "telegram", "twitter", "whatsapp")


def records(seed=0, bump=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(20):
        role = "cal" if p < 12 else "audit"
        for platform in PLATFORMS:
            shift = bump if platform == "whatsapp" else 0.0
            for i, s in enumerate(rng.normal(-3 + shift, 1.0, 40)):
                out.append(
                    {
                        "id": f"{p}-{platform}-{i}",
                        "phone": f"D{p:02d}",
                        "role": role,
                        "platform": platform,
                        "score": float(s),
                    }
                )
    return out


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records()[:5]) + "\n", encoding="utf-8")
    assert len(load_records(path)) == 5


def test_fixed_threshold_rate_is_the_share_of_audit_photos_above_it():
    rows = records()
    rates = audit_rates(rows, mode="fixed", threshold=-3.0)
    assert set(rates) == set(PLATFORMS)
    assert all(0.4 < v < 0.6 for v in rates.values())


def test_pooled_calibration_ignores_platform_so_a_shifted_platform_is_over_flagged():
    rows = records(bump=1.5)
    pooled = audit_rates(rows, mode="pooled", alpha=0.05)
    stratified = audit_rates(rows, mode="stratified", alpha=0.05)
    assert pooled["whatsapp"] > 2 * stratified["whatsapp"]
    assert stratified["whatsapp"] < 0.12


def test_only_audit_phones_are_counted():
    rows = records()
    for r in rows:
        if r["role"] == "cal":
            r["score"] = 99.0
    assert max(audit_rates(rows, mode="fixed", threshold=50.0).values()) == 0.0


def test_report_states_the_verdict_and_every_check():
    rows = records()
    text = render_report(rows, gate_b(rows, n_boot=100), detector="demo detector")
    assert "Gate B" in text and "demo detector" in text
    assert text.count("@") >= 10 or text.count("| orig |") >= 1
    for platform in PLATFORMS:
        assert platform in text
    assert "PASSED" in text or "FAILED" in text
