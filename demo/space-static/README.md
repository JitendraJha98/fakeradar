---
title: fakeradar
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-sa-4.0
short_description: Detect AI-generated images in your browser (research demo)
---

Demo of [fakeradar](https://github.com/JitendraJha98/fakeradar). The `fast-v0`
model runs entirely in your browser (ONNX Runtime Web); uploaded images never
leave your computer. Trained on one generator (GenImage Stable Diffusion v1.4):
read the limits on the page before trusting a score.

Source: `demo/space-static/` in the repository. `model.onnx` is
`fakeradar export-onnx` of `fast_v0.pt`, merged into a single file.
