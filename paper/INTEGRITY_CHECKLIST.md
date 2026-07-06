# GRAFT — Pre-submission Integrity Checklist

This draft is **not submittable yet**. Submitting it with placeholder or
invented numbers would be research misconduct and would end the project's
credibility. Work through every item below, in order.

## 1. Experiments (the paper does not exist without these)
- [ ] Train GRAFT-fast and GRAFT-accurate under the exact protocol in §4
      (GenImage SD1.4-only; configs are in `configs/`)
- [ ] Re-run/re-evaluate every baseline (CNNDetection, LGrad, NPR, UnivFD,
      DIRE, FatFormer, AIDE) with their released code under the SAME
      protocol and SAME perturbations — never copy numbers across papers
      with different protocols
- [ ] Fill Tables 1–2 and the ablation table ONLY from `fakeradar benchmark`
      / `fakeradar robustness` outputs; commit the raw CSVs alongside
- [ ] Run each headline number with ≥3 seeds; report mean (±std in appendix)
- [ ] If a claimed advantage doesn't hold, report it anyway and revise the
      claims — a narrow honest win beats a broad fake one

## 2. Novelty verification (do this the week before submission)
- [ ] Fresh arXiv sweep (cs.CV, last 6 months) for: "gradient field detection",
      "structure tensor forensic", "coherence synthetic image", "CLIP fusion
      fake detection", "frozen backbone AI-generated"
- [ ] If someone published the coherence idea or the same fusion: cite them,
      reposition the contribution (e.g., robustness protocol + system), do
      not pretend they don't exist
- [ ] The "to our knowledge" claims in §1 and §3.1 must be re-dated

## 3. Citations
- [ ] Verify EVERY entry in references.bib against arXiv/DBLP/the PDF:
      authors, venue, year, title. Fix or delete anything unverifiable
- [ ] Read (at minimum skim) every paper you cite — reviewers ask questions

## 4. Writing integrity
- [ ] Rewrite the full text in your own voice; you must be able to defend
      every sentence in a rebuttal without assistance
- [ ] Check the target venue's policy on AI writing assistance and comply
      (most allow assistance, forbid AI authorship, and hold YOU responsible
      for all content)
- [ ] Remove the red draft box and all \tofill markers (grep for "TO FILL")

## 5. Venue mechanics
- [ ] Retarget to the venue template (NeurIPS/CVPR/WACV .sty), check page
      limits, anonymize for double-blind, move code link to supplementary
- [ ] Reproducibility checklist + ethics statement per venue requirements
- [ ] Licenses of all datasets used permit this use; cite dataset papers

## Suggested targets (verify current deadlines yourself)
- Workshop first (fast feedback): NeurIPS/CVPR workshops on synthetic media,
  media forensics, or trustworthy ML
- Main venues: WACV, then CVPR/ICCV depending on strength of results
- Post to arXiv only once Tables 1–2 are real
