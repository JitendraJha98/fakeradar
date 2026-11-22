"""fakeradar CLI.

    fakeradar scan photo.jpg vacation/          # score files or folders
    fakeradar prepare-genimage /data/GenImage --generators stable_diffusion_v_1_4
    fakeradar train configs/train_fast.yaml     # train a tier
    fakeradar calibrate val.csv --checkpoint best.pt
    fakeradar benchmark test.csv --checkpoint best.pt
    fakeradar robustness test.csv --checkpoint best.pt
    fakeradar export-onnx best.pt fast.onnx
    fakeradar serve --checkpoint best.pt        # local REST API

Heavy imports (torch) happen inside commands so `fakeradar --help` is instant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .constants import SUPPORTED_EXTENSIONS, VERDICT_FAKE, VERDICT_REAL

app = typer.Typer(add_completion=False, help="Local-first AI-generated image detection.")
console = Console()

_DEVICE_HELP = "auto (cuda > mps > cpu), cpu, cuda, cuda:1, mps."


def _collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    if not files:
        raise typer.BadParameter("No supported images found.")
    return files


def _build_detector(
    tier: str,
    checkpoint: Path | None,
    pretrained: str | None,
    untrained_ok: bool = False,
    device: str = "auto",
):
    from .detector import Detector

    try:
        det = Detector(
            tier=tier,
            checkpoint=checkpoint,
            pretrained=pretrained,
            device=device,
            allow_random_init=untrained_ok and checkpoint is None and pretrained is None,
        )
    except (ValueError, KeyError) as e:  # no weights / not a checkpoint / unknown zoo name
        raise typer.BadParameter(str(e).strip("'\"")) from e
    if not det.trained:
        console.print(
            "[bold yellow]⚠ running with RANDOM weights (smoke-test mode) — scores are meaningless.[/]"
        )
    return det


@app.command()
def scan(
    paths: list[Path] = typer.Argument(..., help="Image files and/or directories."),
    tier: str = typer.Option("fast", help="fast (offline, CPU) or accurate (adds CLIP branch)."),
    checkpoint: Path = typer.Option(None, help="Path to a trained .pt checkpoint."),
    pretrained: str = typer.Option(None, help="Model-zoo name, e.g. fast-v0."),
    device: str = typer.Option("auto", help=_DEVICE_HELP),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    untrained_ok: bool = typer.Option(False, help="Allow random-init weights (testing only)."),
    fpr_thresholds: Path = typer.Option(
        None, help="Thresholds from `fakeradar calibrate-fpr`: adds a verdict at that level."
    ),
    stratum: str = typer.Option(
        None, help="Transport state of these images (e.g. whatsapp), to pick its threshold."
    ),
):
    """Score images: probability that each image is AI-generated.

    Unreadable files are reported and skipped; the exit code is 1 if any failed.
    """
    from PIL import Image

    det = _build_detector(tier, checkpoint, pretrained, untrained_ok, device)
    cal = None
    if fpr_thresholds is not None:
        from .evaluation.guarantee import load_calibration

        cal = load_calibration(fpr_thresholds)
    results, errors = [], []
    for f in _collect(paths):
        try:
            result = det.predict(f)
            if cal is not None:  # verdict at the calibrated false-accusation level
                known = stratum in cal.thresholds
                limit = cal.thresholds[stratum] if known else cal.pooled
                result.extra.update(
                    flagged=bool(result.prob_ai > limit),
                    alpha=cal.alpha,
                    stratum=stratum if known else "pooled",
                )
            results.append(result)
        except (OSError, Image.DecompressionBombError) as e:  # corrupt/truncated/huge image
            errors.append({"source": str(f), "error": str(e)})

    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in results] + errors, indent=2))
    else:
        table = Table(title=f"fakeradar · tier={det.config.tier}")
        table.add_column("image")
        table.add_column("P(AI)", justify="right")
        table.add_column("verdict")
        table.add_column("branches")
        if cal is not None:
            table.add_column(f"flagged at {cal.alpha:g}")
        for r in results:
            color = (
                "red"
                if r.verdict == VERDICT_FAKE
                else "green"
                if r.verdict == VERDICT_REAL
                else "yellow"
            )
            branches = "  ".join(f"{k}:{v:.2f}" for k, v in r.per_branch.items())
            cells = [
                escape(Path(r.source).name),
                f"{r.prob_ai:.3f}",
                f"[{color}]{r.verdict}[/]",
                branches,
            ]
            if cal is not None:
                cells.append("[red]yes[/]" if r.extra["flagged"] else "no")
            table.add_row(*cells)
        for e in errors:
            table.add_row(
                escape(Path(e["source"]).name), "–", "[magenta]error[/]", escape(e["error"])
            )
        console.print(table)
        console.print("[dim]Probabilistic signal, not proof. See README §Limitations.[/]")
    if errors:
        raise typer.Exit(code=1)


@app.command("prepare-genimage")
def prepare_genimage(
    root: Path = typer.Argument(
        ..., help="GenImage root: ROOT/<generator>/[<subset>/]{train,val}/"
    ),
    generators: list[str] = typer.Option(
        ..., "--generators", "-g", help="Training generator dir(s); repeat or comma-separate."
    ),
    test_generators: list[str] = typer.Option(
        None, "--test-generators", "-t", help="Test generator(s), or 'all'. Default: --generators."
    ),
    out: Path = typer.Option(Path("data/manifests"), help="Output directory."),
    name: str = typer.Option(None, help="Manifest prefix (default genimage_<generators>)."),
    val_fraction: float = typer.Option(
        0.05, help="Hold-out of the train split for val/calibration."
    ),
    seed: int = typer.Option(0),
):
    """Build train / val (hold-out) / test CSV manifests from a GenImage download."""
    from .training.genimage import build_manifests

    def split(values: list[str] | None) -> list[str] | None:
        return [v for item in values for v in item.split(",") if v] if values else None

    try:
        result = build_manifests(
            root, split(generators), split(test_generators), out, name, val_fraction, seed
        )
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    for split_name, (path, n) in result.items():
        console.print(f"{split_name:>5}: [bold]{path}[/] ({n} rows)")


@app.command()
def train(
    config: Path = typer.Argument(..., help="Training YAML, e.g. configs/train_fast.yaml"),
    resume: Path = typer.Option(None, help="Resume from a run's last.pt (optimizer state incl.)."),
):
    """Train a detector tier from a manifest dataset."""
    from .training.train import train as _train

    _train(config, resume=resume)


@app.command()
def calibrate(
    manifest: Path = typer.Argument(..., help="Held-out CSV with path,label[,generator]."),
    checkpoint: Path = typer.Option(..., help="Trained checkpoint to calibrate."),
    out: Path = typer.Option(
        None, help="Write the calibrated checkpoint here (default: in place)."
    ),
    num_workers: int = typer.Option(0, help="DataLoader workers for image decoding."),
    device: str = typer.Option("auto", help=_DEVICE_HELP),
    equalize_jpeg: int = typer.Option(
        None, help="JPEG-encode non-JPEG images once at this quality before scoring."
    ),
):
    """Fit the output temperature on held-out data (calibrated P(AI), lower ECE)."""
    from .detector import auto_device
    from .evaluation.calibrate import calibrate_checkpoint

    stats = calibrate_checkpoint(
        checkpoint,
        manifest,
        out=out,
        num_workers=num_workers,
        device=auto_device() if device == "auto" else device,
        equalize_jpeg_quality=equalize_jpeg,
    )
    console.print(
        f"temperature=[bold]{stats['temperature']:.3f}[/]  "
        f"ECE {stats['ece_before']:.4f} -> [bold]{stats['ece_after']:.4f}[/]  (n={stats['n']})"
    )
    console.print(f"saved -> [bold]{out or checkpoint}[/]")


@app.command()
def benchmark(
    manifest: Path = typer.Argument(..., help="CSV with path,label[,generator]."),
    checkpoint: Path = typer.Option(None, help="Trained checkpoint."),
    pretrained: str = typer.Option(None, help="Model-zoo name, e.g. fast-v0."),
    out: Path = typer.Option(None, help="Write markdown table here."),
    scores_csv: Path = typer.Option(None, help="Write raw per-image scores here."),
    device: str = typer.Option("auto", help=_DEVICE_HELP),
    equalize_jpeg: int = typer.Option(
        None, help="JPEG-encode non-JPEG images once at this quality before scoring."
    ),
):
    """Per-generator AUROC/AP/acc/TPR@5%FPR/ECE table (cross-generator generalization)."""
    from .evaluation.benchmark import run_benchmark

    det = _build_detector("fast", checkpoint, pretrained, device=device)
    console.print(run_benchmark(det, manifest, out, scores_csv, equalize_jpeg))


@app.command()
def robustness(
    manifest: Path = typer.Argument(..., help="CSV with path,label[,generator]."),
    checkpoint: Path = typer.Option(None, help="Trained checkpoint."),
    pretrained: str = typer.Option(None, help="Model-zoo name, e.g. fast-v0."),
    out: Path = typer.Option(None, help="Write markdown table here."),
    scores_csv: Path = typer.Option(
        None, help="Write raw per-image, per-perturbation scores here."
    ),
    device: str = typer.Option("auto", help=_DEVICE_HELP),
    equalize_jpeg: int = typer.Option(
        None, help="JPEG-encode non-JPEG images once at this quality before scoring."
    ),
):
    """Re-evaluate under JPEG/resize/blur — the 'will it survive WhatsApp' table."""
    from .evaluation.benchmark import run_robustness

    det = _build_detector("fast", checkpoint, pretrained, device=device)
    console.print(run_robustness(det, manifest, out, scores_csv, equalize_jpeg))


@app.command("calibrate-fpr")
def calibrate_fpr(
    manifest: Path = typer.Argument(
        ..., help="CSV of REAL photos: path,label(0),generator (= transport state, e.g. whatsapp)."
    ),
    checkpoint: Path = typer.Option(None, help="Trained checkpoint."),
    pretrained: str = typer.Option(None, help="Model-zoo name, e.g. fast-v0."),
    alpha: float = typer.Option(0.01, help="Target rate of real photos flagged as AI."),
    out: Path = typer.Option(None, help="Write thresholds here (default: <manifest>.fpr.json)."),
    device: str = typer.Option("auto", help=_DEVICE_HELP),
    equalize_jpeg: int = typer.Option(
        None, help="JPEG-encode non-JPEG images once at this quality before scoring."
    ),
):
    """Set thresholds from real photos only, so at most ALPHA of real photos are flagged."""
    from .evaluation.guarantee import calibrate_real_manifest, save_calibration

    if not 0.0 < alpha < 1.0:
        raise typer.BadParameter("alpha must be between 0 and 1")
    det = _build_detector("fast", checkpoint, pretrained, device=device)
    try:
        cal, n = calibrate_real_manifest(det, manifest, alpha, equalize_jpeg)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    out = out or manifest.with_suffix(".fpr.json")
    save_calibration(cal, out, n)
    for name, value in sorted(cal.thresholds.items()):
        console.print(f"{name}: flag when P(AI) > [bold]{value:.4f}[/]")
    console.print(f"pooled (unknown or small strata): [bold]{cal.pooled:.4f}[/]")
    if cal.fallback:
        console.print(f"[yellow]too few photos for their own threshold:[/] {sorted(cal.fallback)}")
    console.print(f"calibrated on {n} real photos at alpha={alpha:g} -> [bold]{out}[/]")


@app.command("export-onnx")
def export_onnx_cmd(
    checkpoint: Path = typer.Argument(...),
    out: Path = typer.Argument(Path("fakeradar_fast.onnx")),
    crop_size: int = typer.Option(256),
):
    """Export the fast tier to ONNX for CPU/edge deployment (torch>=2.9)."""
    # torch's export progress logging prints emoji; on legacy Windows consoles
    # (cp1252) that raises UnicodeEncodeError and kills the export. Degrade
    # unencodable characters instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from .models import export_onnx, load_checkpoint, verify_onnx_parity

    model = load_checkpoint(checkpoint)
    path = export_onnx(model, out, crop_size)
    console.print(f"exported -> [bold]{path}[/]")
    try:
        diff = verify_onnx_parity(model, path, crop_size)
        console.print(f"parity vs torch: max abs prob diff = {diff:.2e} — OK")
    except ImportError:
        console.print("[dim]install onnxruntime ('fakeradar[onnx]') to run the parity check[/]")


@app.command()
def serve(
    checkpoint: Path = typer.Option(None, help="Trained checkpoint."),
    pretrained: str = typer.Option(None, help="Model-zoo name, e.g. fast-v0."),
    tier: str = typer.Option("fast"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    device: str = typer.Option("auto", help=_DEVICE_HELP),
    max_upload_mb: float = typer.Option(25.0, help="Reject larger uploads with HTTP 413."),
    untrained_ok: bool = typer.Option(False),
):
    """Run a local REST API (POST /v1/detect with an image file)."""
    try:
        import uvicorn
    except ImportError as e:
        raise typer.BadParameter("Install serve extras: pip install 'fakeradar[serve]'") from e
    from .server import create_app

    try:
        api = create_app(tier, checkpoint, untrained_ok, pretrained, device, max_upload_mb)
    except (ValueError, KeyError) as e:
        raise typer.BadParameter(str(e).strip("'\"")) from e
    uvicorn.run(api, host=host, port=port)


@app.command()
def version():
    """Print the installed fakeradar version."""
    from . import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
