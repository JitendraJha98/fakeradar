"""Gradio demo (also deployable as a Hugging Face Space).

    pip install 'fakeradar[demo]'
    python demo/app.py --checkpoint runs/fast_v0/best.pt
"""

import argparse

import gradio as gr

from fakeradar import Detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--untrained-ok", action="store_true")
    args = ap.parse_args()

    det = Detector(checkpoint=args.checkpoint, allow_random_init=args.untrained_ok)

    def score(img):
        r = det.predict(img)
        label = {"likely_ai": "AI-generated", "likely_real": "Real", "uncertain": "Uncertain"}[r.verdict]
        detail = "\n".join(f"{k}: {v:.3f}" for k, v in r.per_branch.items())
        return {label: r.prob_ai if r.verdict == "likely_ai" else 1 - r.prob_ai}, detail

    gr.Interface(
        fn=score,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=1), gr.Textbox(label="per-branch scores")],
        title="fakeradar — local AI-image detection",
        description="Probabilistic signal, not proof. Gradient-field + frequency + semantic fusion.",
    ).launch()


if __name__ == "__main__":
    main()
