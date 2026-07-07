"""fakeradar CLI.

    fakeradar scan photo.jpg vacation/          # score files or folders
    fakeradar train configs/train_fast.yaml     # train a tier
    fakeradar calibrate val.csv --checkpoint best.pt
    fakeradar benchmark val.csv --checkpoint best.pt
    fakeradar robustness val.csv --checkpoint best.pt
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
from rich.table import Table

from .constants import SUPPORTED_EXTENSIONS, VERDICT_FAKE, VERDICT_REAL

app = typer.Typer(add_completion=False, help="Local-first AI-generated image detection.")
console = Console()


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


def _build_detector(tier: str, checkpoint: Path | None, pretrained: str | None, untrained_ok: bool):
    from .detector import Detector

    try:
        det = Detector(
            tier=tier,
            checkpoint=checkpoint,
            pretrained=pretrained,
            allow_random_init=untrained_ok and checkpoint is None and pretrained is None,
        )
    except ValueError as e:  # e.g. no weights given -> clean message, no traceback
        raise typer.BadParameter(str(e)) from e
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
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    untrained_ok: bool = typer.Option(False, help="Allow random-init weights (testing only)."),
):
    """Score images: probability that each image is AI-generated."""
    det = _build_detector(tier, checkpoint, pretrained, untrained_ok)
    results = [det.predict(f) for f in _collect(paths)]

    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
        return

    table = Table(title=f"fakeradar · tier={det.config.tier}")
    table.add_column("image")
    table.add_column("P(AI)", justify="right")
    table.add_column("verdict")
    table.add_column("branches")
    for r in results:
        color = (
            "red"
            if r.verdict == VERDICT_FAKE
            else "green"
            if r.verdict == VERDICT_REAL
            else "yellow"
        )
        branches = "  ".join(f"{k}:{v:.2f}" for k, v in r.per_branch.items())
        table.add_row(Path(r.source).name, f"{r.prob_ai:.3f}", f"[{color}]{r.verdict}[/]", branches)
    console.print(table)
    console.print("[dim]Probabilistic signal, not proof. See README §Limitations.[/]")


@app.command()
def train(config: Path = typer.Argument(..., help="Training YAML, e.g. configs/train_fast.yaml")):
    """Train a detector tier from a manifest dataset."""
    from .training.train import train as _train

    _train(config)


@app.command()
def calibrate(
    manifest: Path = typer.Argument(..., help="Held-out CSV with path,label[,generator]."),
    checkpoint: Path = typer.Option(..., help="Trained checkpoint to calibrate."),
    out: Path = typer.Option(
        None, help="Write the calibrated checkpoint here (default: in place)."
    ),
    batch_size: int = typer.Option(32),
):
    """Fit the output temperature on held-out data (calibrated P(AI), lower ECE)."""
    from .evaluation.calibrate import calibrate_checkpoint

    stats = calibrate_checkpoint(checkpoint, manifest, out=out, batch_size=batch_size)
    console.print(
        f"temperature=[bold]{stats['temperature']:.3f}[/]  "
        f"ECE {stats['ece_before']:.4f} -> [bold]{stats['ece_after']:.4f}[/]  (n={stats['n']})"
    )
    console.print(f"saved -> [bold]{out or checkpoint}[/]")


@app.command()
def benchmark(
    manifest: Path = typer.Argument(..., help="CSV with path,label[,generator]."),
    checkpoint: Path = typer.Option(..., help="Trained checkpoint."),
    out: Path = typer.Option(None, help="Write markdown table here."),
):
    """Per-generator AUROC/AP/accuracy table (cross-generator generalization)."""
    from .detector import Detector
    from .evaluation.benchmark import run_benchmark

    console.print(run_benchmark(Detector(checkpoint=checkpoint), manifest, out))


@app.command()
def robustness(
    manifest: Path = typer.Argument(...),
    checkpoint: Path = typer.Option(...),
    out: Path = typer.Option(None),
):
    """Re-evaluate under JPEG/resize/blur — the 'will it survive WhatsApp' table."""
    from .detector import Detector
    from .evaluation.benchmark import run_robustness

    console.print(run_robustness(Detector(checkpoint=checkpoint), manifest, out))


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
    checkpoint: Path = typer.Option(None),
    tier: str = typer.Option("fast"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    untrained_ok: bool = typer.Option(False),
):
    """Run a local REST API (POST /v1/detect with an image file)."""
    try:
        import uvicorn
    except ImportError as e:
        raise typer.BadParameter("Install serve extras: pip install 'fakeradar[serve]'") from e
    from .server import create_app

    uvicorn.run(create_app(tier, checkpoint, untrained_ok), host=host, port=port)


@app.command()
def version():
    from . import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
