"""Build, push, wait for and pull private Kaggle jobs.

Every job is a private script notebook whose slug starts with "fakeradar-".
The `lenslaw` package is embedded in the notebook source, so one push carries
the code and nothing is published anywhere else. Credentials are only ever
read by the Kaggle CLI itself.

    python kaggle_jobs.py run phase1-data jobs/phase1_data.py --timeout 3600
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PREFIX = "fakeradar-"
HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "lenslaw"
RUNS_DIR = HERE.parents[1] / "runs" / "lens_law"
_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_BOOTSTRAP = """
import base64 as _b64
import hashlib as _hashlib
import io as _io
import os as _os
import sys as _sys
import tempfile as _tempfile
import zipfile as _zipfile

_code_dir = _os.path.join(
    _tempfile.gettempdir(), "fakeradar_code_" + _hashlib.sha1(_BUNDLE.encode()).hexdigest()[:12]
)
_os.makedirs(_code_dir, exist_ok=True)
_zipfile.ZipFile(_io.BytesIO(_b64.b64decode(_BUNDLE))).extractall(_code_dir)
_sys.path.insert(0, _code_dir)

# ---- entry script ----
"""


def bundle_code(pkg_dir: Path = PKG_DIR) -> str:
    """Base64 zip of the package's .py files; identical bytes on every build."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pkg_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            arcname = (Path(pkg_dir.name) / path.relative_to(pkg_dir)).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack_bundle(bundle: str, dest: Path) -> None:
    zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle))).extractall(dest)


def metadata(
    name: str,
    username: str,
    *,
    gpu: bool,
    internet: bool = True,
    dataset_sources: tuple[str, ...] = (),
    kernel_sources: tuple[str, ...] = (),
) -> dict:
    if not _NAME.fullmatch(name):
        raise ValueError(f"job name {name!r} must be lowercase letters, digits and single dashes")
    slug = PREFIX + name
    return {
        "id": f"{username}/{slug}",
        "title": slug.replace("-", " "),  # Kaggle derives the slug from the title
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if internet else "false",
        "machine_shape": "NvidiaTeslaT4" if gpu else "",
        "dataset_sources": list(dataset_sources),
        "competition_sources": [],
        "kernel_sources": list(kernel_sources),
        "model_sources": [],
    }


def build_job(
    name: str,
    entry: Path,
    out_dir: Path,
    *,
    gpu: bool,
    internet: bool = True,
    dataset_sources: tuple[str, ...] = (),
    kernel_sources: tuple[str, ...] = (),
    username: str | None = None,
    pkg_dir: Path = PKG_DIR,
) -> Path:
    """Write main.py (embedded package + entry script) and kernel-metadata.json."""
    meta = metadata(
        name,
        username or kaggle_username(),
        gpu=gpu,
        internet=internet,
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f'_BUNDLE = "{bundle_code(pkg_dir)}"\n' + _BOOTSTRAP
    source += Path(entry).read_text(encoding="utf-8")
    (out_dir / "main.py").write_bytes(source.encode("utf-8"))
    (out_dir / "kernel-metadata.json").write_bytes(json.dumps(meta, indent=2).encode("utf-8"))
    return out_dir


# ------------------------------------------------------------------ Kaggle CLI
def _kaggle(*args: str, timeout: int = 900) -> str:
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "-W", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])} failed: {out.strip()[-600:]}")
    return out


def parse_username(config_view: str) -> str:
    match = re.search(r"^- username:\s*(\S+)", config_view, re.MULTILINE)
    if not match:
        raise ValueError("no username in `kaggle config view` output")
    return match.group(1)


def kaggle_username() -> str:
    return parse_username(_kaggle("config", "view"))


def parse_status(text: str) -> str:
    match = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', text)
    if not match:
        raise ValueError(f"no kernel status in: {text.strip()[-200:]}")
    return match.group(1)


def push(job_dir: Path, timeout_s: int) -> str:
    """Push and start the job. Returns the kernel reference."""
    out = _kaggle("kernels", "push", "-p", str(job_dir), "-t", str(timeout_s))
    if "successfully pushed" not in out:
        raise RuntimeError(f"push did not succeed: {out.strip()[-600:]}")
    return json.loads((Path(job_dir) / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]


def status(kernel: str) -> str:
    return parse_status(_kaggle("kernels", "status", kernel))


def wait(kernel: str, max_s: int, poll_s: int = 30) -> str:
    """Block until the job leaves the queued and running states."""
    start = time.time()
    while True:
        state = status(kernel)
        print(f"[{int(time.time() - start)}s] {kernel}: {state}", flush=True)
        if state in ("COMPLETE", "ERROR") or state.startswith("CANCEL"):
            return state
        if time.time() - start > max_s:
            raise TimeoutError(f"{kernel} still {state} after {max_s}s")
        time.sleep(poll_s)


def output_args(kernel: str, dest: Path, pattern: str | None = None) -> list[str]:
    args = ["kernels", "output", kernel, "-p", str(dest), "-o", "-q"]
    return args + ["--file-pattern", pattern] if pattern else args


def pull_output(kernel: str, dest: Path, pattern: str | None = None) -> None:
    """Download the job's output files, or only those whose name matches `pattern`."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    _kaggle(*output_args(kernel, dest, pattern), timeout=3600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one private fakeradar job on Kaggle.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="build, push, wait for and pull a job")
    run.add_argument("name")
    run.add_argument("entry", type=Path)
    run.add_argument("--gpu", action="store_true")
    run.add_argument("--timeout", type=int, default=3600, help="run-time cap in seconds")
    run.add_argument("--dataset-source", action="append", default=[])
    run.add_argument("--kernel-source", action="append", default=[])
    run.add_argument("--no-wait", action="store_true")
    run.add_argument("--pkg-dir", type=Path, default=PKG_DIR, help="package folder to embed")
    run.add_argument("--pull", default=None, help="regex: only pull matching output files")
    args = parser.parse_args(argv)

    job_dir = build_job(
        args.name,
        args.entry,
        RUNS_DIR / "_jobs" / args.name,
        gpu=args.gpu,
        dataset_sources=tuple(args.dataset_source),
        kernel_sources=tuple(args.kernel_source),
        pkg_dir=args.pkg_dir.resolve(),
    )
    kernel = push(job_dir, args.timeout)
    print(f"pushed {kernel}", flush=True)
    if args.no_wait:
        return 0
    state = wait(kernel, args.timeout + 900)
    pull_output(kernel, RUNS_DIR / args.name, args.pull)
    print(f"{kernel}: {state}; output in {RUNS_DIR / args.name}", flush=True)
    return 0 if state == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
