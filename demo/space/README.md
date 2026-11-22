---
title: fakeradar
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Detect AI-generated images (research demo, CPU)
---

Gradio version of the demo. Hosting Gradio Spaces on Hugging Face's free CPU now needs a PRO subscription; the free, published demo is the static one in `demo/space-static/`.

Demo of [fakeradar](https://github.com/JitendraJha98/fakeradar): upload an image and
get P(AI) from the published `fast-v0` or `accurate-v0` weights. Trained on one
generator (GenImage Stable Diffusion v1.4); read the limits on the page before
trusting a score. Source of this Space: `demo/space/` in the repository.
