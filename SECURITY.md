# Security Policy

## Reporting a vulnerability
Please report security issues privately via GitHub Security Advisories
("Report a vulnerability" on the repo Security tab). Do not open public issues
for exploitable bugs.

## Scope notes
- `fakeradar serve` is designed for **local / trusted-network** use. It ships
  with no authentication. Do not expose it to the public internet without a
  reverse proxy + auth.
- Detection outputs are **probabilistic signals**, not security guarantees.
  Do not build authentication or legal-evidence systems on a single score.
- Model weights are loaded with `weights_only=True` where possible. Only load
  checkpoints from sources you trust.
