import json
import subprocess
import sys

import kaggle_jobs
import pytest


def make_pkg(tmp_path):
    pkg = tmp_path / "lenslaw"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_bytes(b'"""pkg"""\n')
    (pkg / "thing.py").write_bytes(b"VALUE = 41 + 1\n")
    (pkg / "__pycache__" / "thing.cpython-313.pyc").write_bytes(b"\x00\x01binary")
    return pkg


def test_metadata_is_always_private_and_prefixed():
    meta = kaggle_jobs.metadata("phase1-data", "someone", gpu=True)
    assert meta["is_private"] == "true"
    assert meta["id"] == "someone/fakeradar-phase1-data"
    assert meta["title"] == "fakeradar phase1 data"
    assert meta["code_file"] == "main.py"
    assert meta["kernel_type"] == "script" and meta["language"] == "python"


def test_gpu_jobs_ask_for_the_t4_machine_and_cpu_jobs_do_not():
    gpu = kaggle_jobs.metadata("a", "someone", gpu=True)
    cpu = kaggle_jobs.metadata("a", "someone", gpu=False, internet=False)
    assert gpu["enable_gpu"] == "true" and gpu["machine_shape"] == "NvidiaTeslaT4"
    assert gpu["enable_internet"] == "true"
    assert cpu["enable_gpu"] == "false" and cpu["machine_shape"] == ""
    assert cpu["enable_internet"] == "false"


def test_sources_are_passed_through_as_lists():
    meta = kaggle_jobs.metadata(
        "a", "someone", gpu=False, dataset_sources=("x/y",), kernel_sources=("someone/fakeradar-b",)
    )
    assert meta["dataset_sources"] == ["x/y"]
    assert meta["kernel_sources"] == ["someone/fakeradar-b"]


@pytest.mark.parametrize("bad", ["Phase1", "a_b", "x y", "", "-a", "a-", "a--b", "a/b"])
def test_names_outside_lowercase_letters_digits_and_single_dashes_are_rejected(bad):
    with pytest.raises(ValueError):
        kaggle_jobs.metadata(bad, "someone", gpu=False)


def test_bundle_round_trips_python_files_only(tmp_path):
    pkg = make_pkg(tmp_path)
    out = tmp_path / "out"
    kaggle_jobs.unpack_bundle(kaggle_jobs.bundle_code(pkg), out)
    assert (out / "lenslaw" / "thing.py").read_bytes() == b"VALUE = 41 + 1\n"
    assert (out / "lenslaw" / "__init__.py").read_bytes() == b'"""pkg"""\n'
    assert not list(out.rglob("*.pyc"))


def test_bundle_is_the_same_on_every_build(tmp_path):
    pkg = make_pkg(tmp_path)
    assert kaggle_jobs.bundle_code(pkg) == kaggle_jobs.bundle_code(pkg)


def test_built_job_runs_the_entry_with_the_package_importable(tmp_path):
    pkg = make_pkg(tmp_path)
    entry = tmp_path / "entry.py"
    entry.write_text(
        "from lenslaw.thing import VALUE\nprint('value is', VALUE)\n", encoding="utf-8"
    )
    job = kaggle_jobs.build_job(
        "unit-test", entry, tmp_path / "job", gpu=False, username="someone", pkg_dir=pkg
    )
    meta = json.loads((job / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/fakeradar-unit-test" and meta["is_private"] == "true"
    source = (job / "main.py").read_text(encoding="utf-8")
    assert "kaggle.json" not in source
    run = subprocess.run([sys.executable, str(job / "main.py")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "value is 42" in run.stdout


def test_status_line_is_parsed_to_its_state():
    line = 'someone/fakeradar-x has status "KernelWorkerStatus.COMPLETE"'
    assert kaggle_jobs.parse_status(line) == "COMPLETE"
    assert kaggle_jobs.parse_status("Warning: old version\n" + line) == "COMPLETE"
    with pytest.raises(ValueError):
        kaggle_jobs.parse_status("nothing useful here")


def test_username_is_parsed_from_config_view_output():
    text = "Configuration values from C:\\x\n- username: someone\n- auth_method: LEGACY_API_KEY\n"
    assert kaggle_jobs.parse_username(text) == "someone"


def test_output_can_be_limited_to_files_matching_a_pattern(tmp_path):
    everything = kaggle_jobs.output_args("someone/fakeradar-x", tmp_path)
    some = kaggle_jobs.output_args("someone/fakeradar-x", tmp_path, pattern=r".*\.json$")
    assert "--file-pattern" not in everything
    assert some[-2:] == ["--file-pattern", r".*\.json$"]
    assert some[:3] == ["kernels", "output", "someone/fakeradar-x"]


def test_wait_survives_short_network_outages(monkeypatch):
    replies = iter([RuntimeError("dns"), "RUNNING", RuntimeError("dns"), "COMPLETE"])

    def fake_status(kernel):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(kaggle_jobs, "status", fake_status)
    monkeypatch.setattr(kaggle_jobs.time, "sleep", lambda s: None)
    assert kaggle_jobs.wait("someone/fakeradar-x", max_s=999, poll_s=1) == "COMPLETE"


def test_wait_gives_up_after_many_failures_in_a_row(monkeypatch):
    def always_down(kernel):
        raise RuntimeError("dns")

    monkeypatch.setattr(kaggle_jobs, "status", always_down)
    monkeypatch.setattr(kaggle_jobs.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        kaggle_jobs.wait("someone/fakeradar-x", max_s=10**6, poll_s=1)
