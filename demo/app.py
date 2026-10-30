"""Gradio demo (also deployable as a Hugging Face Space).

    pip install 'fakeradar[demo]'
    python demo/app.py --checkpoint runs/fast_v0/best.pt
    python demo/app.py --pretrained fast-v0          # needs 'fakeradar[hub]'

A Space runs `app.py` with no arguments: configure it with the
FAKERADAR_CHECKPOINT or FAKERADAR_PRETRAINED environment variables instead.
"""

import argparse
import os

import gradio as gr

from fakeradar import Detector

VERDICT_TEXT = {"likely_ai": "Likely AI-generated", "likely_real": "Likely real", "uncertain": "Uncertain"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("FAKERADAR_CHECKPOINT"))
    ap.add_argument("--pretrained", default=os.environ.get("FAKERADAR_PRETRAINED"))
    ap.add_argument("--untrained-ok", action="store_true")
    args = ap.parse_args()

    det = Detector(
        checkpoint=args.checkpoint,
        pretrained=None if args.checkpoint else args.pretrained,
        allow_random_init=args.untrained_ok,
    )

    def score(img):
        if img is None:
            return {}, ""
        r = det.predict(img)
        lines = [f"verdict: {VERDICT_TEXT[r.verdict]}"]
        lines += [f"{k}: {v:.3f}" for k, v in r.per_branch.items()]
        if not r.trained:
            lines.append("WARNING: random weights — scores are meaningless.")
        return {"AI-generated": r.prob_ai, "Real": 1 - r.prob_ai}, "\n".join(lines)

    gr.Interface(
        fn=score,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=2, label="P(AI)"), gr.Textbox(label="details")],
        title="fakeradar — local AI-image detection",
        description="Probabilistic signal, not proof. Gradient-field + frequency + semantic fusion.",
    ).launch()


if __name__ == "__main__":
    main()
