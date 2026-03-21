# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

EEG-based neurological condition classification challenge (AD / CN / FTD). The full task definition, data format, and evaluation criteria are in `Challenge_EEG.pdf` — read it before writing any code.

## Data Layout

```
training/
  train_label_mapping.csv   # anonymized_id → label (A=AD, C=CN, F=FTD)
  AD/   # 25 .npy files, Alzheimer's Disease
  CN/   # 28 .npy files, Control Normal
  FTD/  # 16 .npy files, Frontotemporal Dementia
```

Each `.npy` file is an EEG recording for one subject (~8–24 MB). Labels map anonymized subject IDs to one of three classes.

## Stack

No tooling is established yet. The expected Python stack:
- `numpy`, `scipy` — signal processing
- `mne` — EEG-specific preprocessing
- `scikit-learn`, `pytorch`, or `tensorflow` — modeling

Update this file once `requirements.txt` / `pyproject.toml` and project structure exist.
