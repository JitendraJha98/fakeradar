# Launch plan — how this becomes the go-to open detector

## Positioning
"**The detector you can actually run.** Local-first, robustness-audited,
honest about uncertainty." Nobody else in open source combines: maintained +
deployable + cross-generator benchmarked + runs on a laptop.

## Pre-launch gate (do NOT launch before these)
1. v0.1 weights on HF Hub with model card
2. Cross-generator + robustness tables in README (real numbers)
3. HF Space demo live
4. 15-min quickstart verified on a clean machine (Linux + Mac CPU)

## Launch week
- Show HN: "fakeradar — open-source AI-image detection that runs locally"
- r/MachineLearning (technical post: robustness tables front and center)
- X/LinkedIn thread: 5 images, 5 verdicts, 1 failure case (honesty = credibility)
- HF community post + model card cross-links

## Content engine (1 post / 2 weeks)
- "Why AI-image detectors die on WhatsApp" (robustness study)
- "What gradient fields see that you can't" (visual explainer)
- "We tested fakeradar on [new generator] the week it launched" (repeatable format)

## Community flywheel
- `new_generator_request` issue template -> monthly eval refresh -> changelog
- Leaderboard page = reason to return; failure-case gallery = reason to trust
- Good-first-issues: new perturbations, dataset adapters, docs

## Integration targets (post v0.2)
- Fact-checking orgs & OSINT communities (Bellingcat-style workflows)
- Moderation pipelines (REST API), CMS plugins
- GRAFT paper (paper/) for research credibility -> repo for adoption; each feeds the other

## Metrics that matter
GitHub stars are vanity; track: weekly PyPI installs, HF weight downloads,
failure cases reported, external PRs merged.
