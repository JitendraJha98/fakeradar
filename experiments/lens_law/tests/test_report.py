import json
import math

from lenslaw.report import diagnostics, load_records


def test_records_are_loaded_from_jsonl_including_nan_values(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [{"id": "a", "resid_lens": float("nan")}, {"id": "b", "resid_lens": 0.2}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    loaded = load_records(path)
    assert [r["id"] for r in loaded] == ["a", "b"]
    assert math.isnan(loaded[0]["resid_lens"]) and loaded[1]["resid_lens"] == 0.2


def test_diagnostics_group_by_source_variant_and_condition():
    def rec(variant, applicable, r2):
        return {
            "source": "pairs",
            "variant": variant,
            "condition": "clean",
            "split": "dev",
            "applicable": applicable,
            "r2_lens": r2,
            "resid_lens": 0.1,
            "blur_spread": 3.0,
            "n": 80,
        }

    records = [rec("wide_depth_self", True, 0.8), rec("wide_depth_self", True, 0.4)]
    records += [rec("wide_depth_self", False, float("nan")), rec("wide_depth_gt", True, 0.9)]
    out = diagnostics(records, "dev")
    row = out["pairs/wide_depth_self/clean"]
    assert row["n"] == 3
    assert row["applicable_share"] == 2 / 3
    assert row["median_r2_lens"] == 0.6000000000000001 or abs(row["median_r2_lens"] - 0.6) < 1e-9
    assert out["pairs/wide_depth_gt/clean"]["n"] == 1


def test_diagnostics_only_count_the_requested_split():
    records = [
        {
            "source": "render",
            "variant": "lawful",
            "condition": "clean",
            "split": s,
            "applicable": True,
            "r2_lens": 0.9,
            "resid_lens": 0.1,
            "blur_spread": 3.0,
            "n": 80,
        }
        for s in ("dev", "dev", "test")
    ]
    assert diagnostics(records, "dev")["render/lawful/clean"]["n"] == 2
    assert diagnostics(records, "test")["render/lawful/clean"]["n"] == 1
