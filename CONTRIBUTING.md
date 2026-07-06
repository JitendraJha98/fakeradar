# Contributing

## Dev setup
```bash
git clone https://github.com/JitendraJha98/fakeradar && cd fakeradar
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
pytest -q && ruff check src tests
```

## High-value contributions
1. **Failure cases** — an image fakeradar gets wrong + its score (use the
   new-generator issue template). These directly drive eval refreshes.
2. **New generator datasets** — a CSV manifest (`path,label,generator`) +
   provenance/license notes. See below.
3. **New forensic branches** — subclass `DetectionBranch`, register it,
   add a config + a shape test. The fusion head adapts automatically:

```python
from fakeradar.branches.base import DetectionBranch, register_branch

@register_branch("mybranch")
class MyBranch(DetectionBranch):
    feature_dim = 128
    input_kind = "crop"          # or "resized"
    def forward(self, x): ...    # (B,3,H,W) in [0,1] -> (B,128)
```
4. **Perturbations** — add to `PERTURBATIONS` in `evaluation/benchmark.py`.

## Adding a new generator to the benchmark
- Collect ≥1k images with clear licensing; note generator name+version+date
- Write a manifest CSV; run `fakeradar benchmark`
- PR the manifest script + resulting table row to `docs/benchmarks.md`

## Rules of the road
- Keep the `fast` tier offline-capable: no new mandatory downloads
- Any change touching detection quality must include benchmark numbers
- Honest reporting only: clean AND perturbed numbers, never just the good table
