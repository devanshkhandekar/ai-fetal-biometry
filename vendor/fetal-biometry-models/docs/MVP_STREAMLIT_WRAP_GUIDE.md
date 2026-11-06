# Streamlit MVP Wrap Guide

The dashboard wraps the complete Core Biometry MVP flow:

```text
scan input → preprocessing/validity gate → static segmentation → post-processing → ellipse fitting → deterministic measurements → GA lookup → quality/confidence → accept/review/reject → reports
```

## Required checkpoint setup

```text
models/checkpoints/4a_best_pruned_ft_v10.pth
models/checkpoints/phase0_model.pth
```

## Run backend once

```bash
python scripts/03_run_inference_devansh.py \
  --image data/sample_cases/example.png \
  --checkpoint models/checkpoints/4a_best_pruned_ft_v10.pth \
  --variant phase4a \
  --pixel-spacing-mm 0.1
```

## Run dashboard

```bash
streamlit run app/streamlit_app.py
```

## Suggested demo cases

- Good-quality head scan
- Medium-quality review case
- Poor-quality/reject case
- Missing-pixel-spacing case
- DICOM metadata case, if available
