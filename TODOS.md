# TODOS

## Post-Hackathon

### Investigate SFREQ=128 vs SFREQ=500 impact on XGBoost
- **What:** Compare XGBoost classification accuracy with SFREQ=128 (original, incorrect) vs SFREQ=500 (correct) to quantify the impact of the sampling rate mismatch.
- **Why:** The original XGBoost code used SFREQ=128 on 500Hz data without downsampling, meaning all frequency band labels were incorrect. Despite this, the model scored 12/15. Understanding whether the model learned from genuine EEG patterns or sampling artifacts has scientific value and could inform future EEG classification work.
- **Context:** Found during eng review on 2026-03-21. The bug means "alpha power (8-13Hz)" was actually measuring ~31-51Hz. The SFREQ was fixed to 500 for the hackathon submission, but the before/after comparison was not rigorously documented.
- **Depends on:** Hackathon submission complete.

### Add unit tests for PLV features and ensemble logic
- **What:** Create pytest unit tests for: PLV values in [0,1], feature output shapes (30,5,19), multi-seed ensemble averaging, meta-ensemble CSV join correctness.
- **Why:** Current validation relies solely on the label-shuffle sanity check script. No unit-level tests exist for individual functions. During eng review, chose inline assertions over pytest due to hackathon deadline.
- **Pros:** Catches regressions if code is reused or features are modified.
- **Cons:** Only valuable if the project continues past the hackathon.
- **Context:** Found during eng review on 2026-03-21. Inline assertions (in predict.py / ensemble scripts) serve as the interim safety net.
- **Depends on:** Hackathon submission complete.

### Extract shared feature precomputation helper (DRY)
- **What:** Pull the load→normalize→segment→extract→save loop from `cache.py:precompute_all()` and `inference.py:precompute_test_features()` into a shared function.
- **Why:** Both functions implement nearly identical pipelines (~30 lines). If feature extraction logic changes, both must be updated in sync or train/test features silently drift apart.
- **Pros:** Single source of truth for the feature pipeline.
- **Cons:** Only matters if features are modified again.
- **Context:** Found during eng review on 2026-03-21. Features are locked for the hackathon, so risk of drift is near-zero in the short term.
- **Depends on:** Hackathon submission complete.
