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
