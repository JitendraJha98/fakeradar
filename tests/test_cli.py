import json

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from fakeradar import __version__  # noqa: E402
from fakeradar.cli import app  # noqa: E402

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_scan_requires_weights_clean_error(tmp_path):
    img = tmp_path / "x.png"
    Image.new("RGB", (300, 300), (50, 60, 70)).save(img)
    result = runner.invoke(app, ["scan", str(img)])
    assert result.exit_code != 0
    assert "No weights given" in result.output
    assert "Traceback" not in result.output


def test_scan_untrained_table_and_json(tmp_path):
    img = tmp_path / "x.png"
    Image.effect_noise((300, 300), 40).convert("RGB").save(img)

    result = runner.invoke(app, ["scan", str(img), "--untrained-ok"])
    assert result.exit_code == 0, result.output
    assert "RANDOM weights" in result.output
    assert "x.png" in result.output

    result = runner.invoke(app, ["scan", str(img), "--untrained-ok", "--json"])
    assert result.exit_code == 0, result.output
    start = result.output.index("[")
    payload = json.loads(result.output[start:])
    assert len(payload) == 1
    assert 0.0 <= payload[0]["prob_ai"] <= 1.0
    assert payload[0]["trained"] is False


def test_scan_no_images_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["scan", str(tmp_path), "--untrained-ok"])
    assert result.exit_code != 0
    assert "No supported images" in result.output


def test_scan_reports_corrupt_files_and_continues(tmp_path):
    good = tmp_path / "good.png"
    Image.effect_noise((300, 300), 40).convert("RGB").save(good)
    (tmp_path / "broken.jpg").write_bytes(b"not a jpeg")

    result = runner.invoke(app, ["scan", str(tmp_path), "--untrained-ok", "--json"])
    assert result.exit_code == 1  # some files failed
    payload = json.loads(result.output[result.output.index("[") :])
    by_name = {p["source"].replace("\\", "/").rsplit("/", 1)[-1]: p for p in payload}
    assert 0.0 <= by_name["good.png"]["prob_ai"] <= 1.0
    assert "error" in by_name["broken.jpg"]


def test_benchmark_without_weights_is_a_clean_error(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("path,label\nx.png,0\n")
    result = runner.invoke(app, ["benchmark", str(manifest)])
    assert result.exit_code != 0
    assert "No weights given" in result.output and "Traceback" not in result.output


def test_non_checkpoint_file_is_a_clean_error(tmp_path):
    bogus = tmp_path / "bogus.pt"
    torch.save({"weights": torch.zeros(1)}, bogus)
    img = tmp_path / "x.png"
    Image.new("RGB", (300, 300)).save(img)
    result = runner.invoke(app, ["scan", str(img), "--checkpoint", str(bogus)])
    assert result.exit_code != 0
    assert "not a fakeradar checkpoint" in result.output


def test_equalize_jpeg_option_is_offered_where_images_are_scored():
    for command in ("benchmark", "robustness", "calibrate"):
        result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        assert "--equalize-jpeg" in result.output


def test_calibrate_fpr_command_exists_and_refuses_to_run_without_weights(tmp_path):
    result = runner.invoke(app, ["calibrate-fpr", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "--alpha" in result.output
    manifest = tmp_path / "real.csv"
    manifest.write_text("path,label\nx.jpg,0\n")
    result = runner.invoke(app, ["calibrate-fpr", str(manifest)])
    assert result.exit_code != 0


def _thresholds(tmp_path, pooled, per_stratum=None):
    path = tmp_path / "fpr.json"
    data = {
        "alpha": 0.01,
        "n": 500,
        "pooled": pooled,
        "thresholds": per_stratum or {},
        "fallback": [],
    }
    path.write_text(json.dumps(data))
    return path


def test_scan_reports_the_verdict_at_the_calibrated_level(tmp_path):
    img = tmp_path / "a.png"
    Image.effect_noise((96, 96), 40).convert("RGB").save(img)
    low = runner.invoke(
        app,
        [
            "scan",
            str(img),
            "--untrained-ok",
            "--json",
            "--fpr-thresholds",
            str(_thresholds(tmp_path, -1.0)),
        ],
    )
    assert low.exit_code == 0
    row = json.loads(low.output[low.output.index("[") :])[0]
    assert row["flagged"] is True and row["alpha"] == 0.01 and row["stratum"] == "pooled"

    high = runner.invoke(
        app,
        [
            "scan",
            str(img),
            "--untrained-ok",
            "--json",
            "--fpr-thresholds",
            str(_thresholds(tmp_path, 2.0)),
        ],
    )
    assert json.loads(high.output[high.output.index("[") :])[0]["flagged"] is False


def test_scan_uses_the_named_stratum_threshold(tmp_path):
    img = tmp_path / "a.png"
    Image.effect_noise((96, 96), 40).convert("RGB").save(img)
    path = _thresholds(tmp_path, 2.0, {"whatsapp": -1.0})
    args = ["scan", str(img), "--untrained-ok", "--json", "--fpr-thresholds", str(path)]
    named = runner.invoke(app, [*args, "--stratum", "whatsapp"])
    row = json.loads(named.output[named.output.index("[") :])[0]
    assert row["flagged"] is True and row["stratum"] == "whatsapp"
    other = runner.invoke(app, [*args, "--stratum", "never-seen"])
    row = json.loads(other.output[other.output.index("[") :])[0]
    assert row["flagged"] is False and row["stratum"] == "pooled"


def test_scan_without_thresholds_is_unchanged(tmp_path):
    img = tmp_path / "a.png"
    Image.effect_noise((96, 96), 40).convert("RGB").save(img)
    result = runner.invoke(app, ["scan", str(img), "--untrained-ok", "--json"])
    assert "flagged" not in json.loads(result.output[result.output.index("[") :])[0]
