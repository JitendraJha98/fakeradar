"""fakeradar demo for Hugging Face Spaces (CPU).

Loads the published weights from the Hub on first use of each tier and scores
an uploaded image. The page states the limits plainly: one training
generator, a probabilistic signal, and the fixed 0.5 cut is not a guarantee.
"""

from __future__ import annotations

import gradio as gr
from huggingface_hub import hf_hub_download

from fakeradar import Detector

TIERS = {
    "fast (47 MB, gradient + frequency)": ("JitendraJha98/fakeradar-fast", "fast_v0.pt"),
    "accurate (393 MB, adds frozen CLIP; more robust to compression)": (
        "JitendraJha98/fakeradar-accurate",
        "accurate_v0.pt",
    ),
}
VERDICT = {
    "likely_ai": "Likely AI-generated",
    "likely_real": "Likely real",
    "uncertain": "Uncertain",
}
_loaded: dict[str, Detector] = {}


def detector(tier: str) -> Detector:
    if tier not in _loaded:
        repo, filename = TIERS[tier]
        _loaded[tier] = Detector(checkpoint=hf_hub_download(repo, filename), device="cpu")
    return _loaded[tier]


def score(image, tier):
    if image is None:
        return {}, "Upload an image to score it."
    r = detector(tier).predict(image)
    branches = ", ".join(f"{k} {v:.2f}" for k, v in r.per_branch.items())
    details = (
        f"**{VERDICT[r.verdict]}** (P(AI) = {r.prob_ai:.3f}; 'uncertain' between 0.35 and 0.65)\n\n"
        f"Per-branch P(AI): {branches}\n\n"
        f"{r.n_crops} native-resolution crops scored in {r.latency_ms:.0f} ms."
    )
    return {"AI-generated": r.prob_ai, "Real": 1 - r.prob_ai}, details


INTRO = """
# fakeradar

Upload an image to get the probability that it was made by an AI image generator.

**Read before trusting a score.** Both models were trained on one generator (Stable
Diffusion v1.4). They are strong on that family and weaker on others, and were never
tested on 2025-2026 commercial generators. Real photos can score high: in a spot check
one real photo scored 0.997. A score is a probabilistic signal, never proof about a
person. For real decisions, set a threshold on your own real photos with
`fakeradar calibrate-fpr` ([code](https://github.com/JitendraJha98/fakeradar)).
Uploaded images are processed in memory and not stored by this app.
"""

with gr.Blocks(title="fakeradar") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Image")
            tier = gr.Radio(list(TIERS), value=list(TIERS)[0], label="Model")
            run = gr.Button("Score image", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=2, label="P(AI)")
            details = gr.Markdown()
    run.click(score, inputs=[image, tier], outputs=[label, details])
    gr.Markdown(
        "Models: [fakeradar-fast](https://huggingface.co/JitendraJha98/fakeradar-fast) · "
        "[fakeradar-accurate](https://huggingface.co/JitendraJha98/fakeradar-accurate) · "
        "weights CC-BY-NC-SA-4.0 (non-commercial), code Apache-2.0."
    )

if __name__ == "__main__":
    demo.launch()
