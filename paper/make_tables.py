"""Write paper/generated_tables.tex from the saved result files.

    python paper/make_tables.py

Every number in the paper's result tables comes from here; none is typed by
hand (paper/INTEGRITY_CHECKLIST.md). Sources:
  experiments/lens_law/figures/gate_m_v0_dev.json, gate_m_v1_dev.json
  experiments/fpr_guarantee/GATE_B_REPORT.json
  runs/fpr_guarantee/detection_rates.json
  runs/lens_law/phase4-*/  (via experiments/phase4/summarize.py)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "phase4"))
from summarize import (  # noqa: E402
    ABLATION,
    CONDITIONS,
    GENERATORS,
    MAIN,
    ROOT,
    available,
    benchmark,
    robustness,
)

PRETTY = {
    "stable_diffusion_v_1_4": "SD v1.4",
    "stable_diffusion_v_1_5": "SD v1.5",
    "wukong": "Wukong",
    "VQDM": "VQDM",
    "Midjourney": "Midjourney",
    "glide": "GLIDE",
    "BigGAN": "BigGAN",
    "ADM": "ADM",
}
COND = {
    "clean": "clean",
    "jpeg90": "JPEG 90",
    "jpeg70": "JPEG 70",
    "jpeg50": "JPEG 50",
    "jpeg30": "JPEG 30",
    "down0.75": "down $\\times$0.75",
    "down0.5": "down $\\times$0.5",
    "blur1.0": "blur $\\sigma{=}1$",
}


def ms(values: list[float], digits: int = 3) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f}$\\pm${statistics.stdev(values):.{digits}f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def gate_m_table() -> str:
    v0 = json.loads((REPO / "experiments/lens_law/figures/gate_m_v0_dev.json").read_text())["dev"][
        "gate"
    ]
    v1 = json.loads((REPO / "experiments/lens_law/figures/gate_m_v1_dev.json").read_text())["dev"][
        "gate"
    ]

    def row(g):
        m3 = g["M3"]["value"]
        m5 = g["M5"]["value"]["auroc"]
        return {
            "M1": f"{pct(g['M1']['value']['depth_narrow'])} / {pct(g['M1']['value']['depth_self'])}",
            "M2": pct(g["M2"]["value"]),
            "M3": " / ".join(f"{m3[k]:.2f}" for k in ("step", "position", "missing_foreground")),
            "M4": f"{pct(g['M4']['value']['unlawful_flagged'])} / {pct(g['M4']['value']['lawful_flagged'])}",
            "M5": " / ".join(f"{m5[k]:.2f}" for k in ("step", "position", "missing_foreground")),
        }

    a, b = row(v0), row(v1)
    need = {
        "M1": "$\\ge$80\\% / 70\\%",
        "M2": "$\\ge$90\\%",
        "M3": "$\\ge$0.85 each",
        "M4": "$\\ge$50\\% / $\\le$10\\%",
        "M5": "drop $\\le$0.10",
    }
    what = {
        "M1": "real pairs fit the V",
        "M2": "wide shot has more blur spread",
        "M3": "AUROC, lawful vs.\\ broken",
        "M4": "broken / lawful flagged at 5\\%",
        "M5": "AUROC after 0.5$\\times$+JPEG70",
    }
    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Gate M (development split). Both blur estimators fail; the test split was never scored.}",
        "\\label{tab:gatem}",
        "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{l l l l l}\\hline",
        "Check & What & Needed & Edge ratio (v0) & Learned (v1)\\\\\\hline",
    ]
    for k in ("M1", "M2", "M3", "M4", "M5"):
        lines.append(f"{k} & {what[k]} & {need[k]} & {a[k]} & {b[k]}\\\\")
    lines += ["\\hline\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def gate_b_table() -> str:
    g = json.loads((REPO / "experiments/fpr_guarantee/GATE_B_REPORT.json").read_text())
    names = {
        "orig": "camera original",
        "facebook": "Facebook",
        "instagram": "Instagram",
        "twitter": "Twitter",
        "whatsapp": "WhatsApp",
    }
    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Gate B: share of real photos flagged on 11 audit phones never used for calibration "
        f"(23{{,}}105 FODB photos). {'Passed' if g['passed'] else 'Failed'}: "
        f"{g['n_failed']} of {len(g['checks'])} checks missed.}}",
        "\\label{tab:gateb}",
        "\\begin{tabular}{l c c}\\hline",
        "Version & target 5\\% & target 1\\%\\\\\\hline",
    ]
    for p in ("orig", "facebook", "instagram", "twitter", "whatsapp"):
        lines.append(
            f"{names[p]} & {pct(g['checks'][f'{p}@0.05']['rate'])} & {pct(g['checks'][f'{p}@0.01']['rate'])}\\\\"
        )
    h = g["held_out"]["rates"]
    lines.append(f"Telegram (held out) & {pct(h['0.05'])} & {pct(h['0.01'])}\\\\")
    lines += ["\\hline\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def detection_table() -> str:
    d = json.loads((REPO / "runs/fpr_guarantee/detection_rates.json").read_text())
    platforms = ("orig", "facebook", "instagram", "twitter", "whatsapp")
    cond = {
        "clean": "as generated",
        "s075_q85": "0.75$\\times$ + JPEG 85",
        "s050_q70": "0.5$\\times$ + JPEG 70",
    }

    def rng(level, key):
        v = [d[level][key][p] for p in platforms]
        lo, hi = min(v), max(v)
        return pct(lo) if abs(hi - lo) < 1e-9 else f"{pct(lo)}--{pct(hi)}"

    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Share of 200 SDXL images flagged by the reference detector: its own 0.5 cut vs.\\ "
        "thresholds calibrated on real photos (range over the five platform versions).}",
        "\\label{tab:power}",
        "\\begin{tabular}{l c c c}\\hline",
        "Condition & 0.5 cut & calibrated 5\\% & calibrated 1\\%\\\\\\hline",
    ]
    for c, name in cond.items():
        k = f"sdxl/{c}"
        lines.append(f"{name} & {pct(d['fixed_cut'][k])} & {rng('0.05', k)} & {rng('0.01', k)}\\\\")
    lines += ["\\hline\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def benchmark_tables() -> str:
    main, abl = available(MAIN, ROOT), available(ABLATION, ROOT)
    eq = {s: benchmark(p / "benchmark_equalized.md") for s, p in main.items()}
    raw = {s: benchmark(p / "benchmark_raw.md") for s, p in main.items()}
    ab = {s: benchmark(p / "benchmark_equalized.md") for s, p in abl.items()}
    rob = {s: robustness(p / "robustness_equalized.md") for s, p in main.items()}
    rob_ab = {s: robustness(p / "robustness_equalized.md") for s, p in abl.items()}
    n = len(main)
    lines = [
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{Cross-generator AUROC, trained on SD v1.4 only (mean$\\pm$std over {n} seeds). "
        "``Equalized'': non-JPEG images JPEG-encoded once at quality 96, closing the JPEG/PNG shortcut.}",
        "\\label{tab:main}",
        "\\begin{tabular}{l c c c}\\hline",
        "Test generator & equalized & as stored & w/o coherence\\\\\\hline",
    ]
    for g in (*GENERATORS, "MEAN"):
        name = "\\textbf{Mean of 8}" if g == "MEAN" else PRETTY[g]
        no = ms([d[g]["auroc"] for d in ab.values()]) if ab else "--"
        lines.append(
            f"{name} & {ms([d[g]['auroc'] for d in eq.values()])} & {ms([d[g]['auroc'] for d in raw.values()])} & {no}\\\\"
        )
    lines += ["\\hline\\end{tabular}", "\\end{table}", ""]
    lines += [
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{Transport robustness: AUROC on 500 real + 500 generated test images (equalized; {n} seeds).}}",
        "\\label{tab:robust}",
        "\\begin{tabular}{l c c}\\hline",
        "Condition & full model & w/o coherence\\\\\\hline",
    ]
    for c in CONDITIONS:
        no = ms([r[c] for r in rob_ab.values()]) if rob_ab else "--"
        lines.append(f"{COND[c]} & {ms([r[c] for r in rob.values()])} & {no}\\\\")
    lines += ["\\hline\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> int:
    parts = {
        "gatem": gate_m_table(),
        "gateb": gate_b_table(),
        "power": detection_table(),
        "bench": benchmark_tables(),
    }
    for name, body in parts.items():
        out = REPO / "paper" / f"generated_tables_{name}.tex"
        text = "% Generated by paper/make_tables.py -- do not edit by hand.\n" + body + "\n"
        out.write_bytes(text.encode("utf-8"))
        print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
