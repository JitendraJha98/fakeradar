// Scoring logic shared by the browser page and its Node test.
// Mirrors fakeradar.preprocessing.crop_positions / extract_crops and
// Detector.predict: up to 4 native-resolution 256x256 crops (centre, then
// corners, then edge midpoints), mean of logits, sigmoid. The ONNX graph
// already applies the calibration temperature and returns P(AI) per crop.
(function (root) {
  const SIZE = 256;
  const MAX_CROPS = 4;
  const THRESHOLD_REAL = 0.35;
  const THRESHOLD_FAKE = 0.65;

  function cropPositions(w, h, size, maxCrops) {
    const x0 = 0, x1 = w - size, xc = Math.floor((w - size) / 2);
    const y0 = 0, y1 = h - size, yc = Math.floor((h - size) / 2);
    const grid = [[xc, yc], [x0, y0], [x1, y0], [x0, y1], [x1, y1], [xc, y0], [xc, y1], [x0, yc], [x1, yc]];
    const out = [];
    for (const p of grid) {
      if (!out.some((q) => q[0] === p[0] && q[1] === p[1])) out.push(p);
    }
    return out.slice(0, Math.max(1, maxCrops));
  }

  // rgba: Uint8ClampedArray of width*height*4. Returns Float32Array (n,3,S,S) in [0,1].
  function cropsToTensor(rgba, width, positions, size) {
    const n = positions.length, plane = size * size;
    const data = new Float32Array(n * 3 * plane);
    positions.forEach(([x, y], k) => {
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const src = ((y + r) * width + (x + c)) * 4;
          const dst = k * 3 * plane + r * size + c;
          data[dst] = rgba[src] / 255;
          data[dst + plane] = rgba[src + 1] / 255;
          data[dst + 2 * plane] = rgba[src + 2] / 255;
        }
      }
    });
    return data;
  }

  function meanLogitProbability(probs) {
    let z = 0;
    for (const p0 of probs) {
      const p = Math.min(Math.max(p0, 1e-7), 1 - 1e-7);
      z += Math.log(p / (1 - p));
    }
    z /= probs.length;
    return 1 / (1 + Math.exp(-z));
  }

  function verdict(p) {
    if (p >= THRESHOLD_FAKE) return { text: "Likely AI-generated", css: "ai" };
    if (p <= THRESHOLD_REAL) return { text: "Likely real", css: "real" };
    return { text: "Uncertain", css: "uncertain" };
  }

  async function runCrops(session, ort, data, n) {
    const input = new ort.Tensor("float32", data, [n, 3, SIZE, SIZE]);
    const out = await session.run({ crop: input });
    return Array.from(out.prob_ai.data);
  }

  // Browser path: draw the bitmap (upscaling only if smaller than a crop), crop, score.
  async function score(session, ort, bitmap) {
    let w = bitmap.width, h = bitmap.height;
    if (Math.min(w, h) < SIZE) {
      const s = SIZE / Math.min(w, h);
      w = Math.round(w * s);
      h = Math.round(h * s);
    }
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const positions = cropPositions(w, h, SIZE, MAX_CROPS);
    const probs = await runCrops(session, ort, cropsToTensor(rgba, w, positions, SIZE), positions.length);
    return { probAi: meanLogitProbability(probs), nCrops: positions.length };
  }

  const api = { SIZE, MAX_CROPS, cropPositions, cropsToTensor, meanLogitProbability, verdict, runCrops, score };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FakeradarScoring = api;
})(typeof window !== "undefined" ? window : globalThis);
