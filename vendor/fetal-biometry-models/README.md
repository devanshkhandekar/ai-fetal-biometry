# Fetal Head Biometry Core MVP

This package implements the complete one-week fetal head biometry MVP:

```text
scan/image input → preprocessing → validity gate → pretrained segmentation → mask cleanup → ellipse fitting → HC/BPD/OFD/head area → GA lookup → quality/confidence → overlay → JSON/CSV/PDF-ready outputs → Streamlit dashboard
```

## Main features

- PNG/JPG/TIFF/BMP/DICOM upload
- Static pretrained fetal-head segmentation backend
- Model variant selector: `phase4a` or `phase0`
- Rule-based input validity gate
- Mask post-processing: thresholding, largest component, hole cleanup
- Ellipse fitting and geometry extraction
- Measurements: HC, BPD, OFD, head area
- Pixel-to-mm conversion when DICOM/manual spacing exists
- HC-based GA lookup through configurable chart file
- Quality score, confidence score, accept/review/reject triage
- Mask + ellipse overlay
- Patient, doctor, and technical report previews
- JSON, CSV, overlay PNG, and simple PDF downloads
- Runtime/device/checksum audit metadata

## Setup

```bash
cd fetal_biometry_devansh_mvp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Prepare model weights

Place your downloaded pretrained checkpoint files here:

```text
models/checkpoints/4a_best_pruned_ft_v10.pth
models/checkpoints/phase0_model.pth
```

Or copy them into place with:

```bash
python scripts/prepare_devansh_weights.py \
  --source /path/to/4a_best_pruned_ft_v10.pth \
  --variant phase4a
```

Recommended MVP default:

```text
variant: phase4a
checkpoint: models/checkpoints/4a_best_pruned_ft_v10.pth
```

## Run command-line inference

```bash
python scripts/03_run_inference_devansh.py \
  --image /path/to/image.png \
  --checkpoint models/checkpoints/4a_best_pruned_ft_v10.pth \
  --variant phase4a \
  --pixel-spacing-mm 0.1 \
  --output-dir outputs
```

## Run Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard supports:

- Single-scan inference
- Batch demo folder processing
- Doctor review tab
- Patient summary tab
- GA/growth-ready tab
- Technical audit tab
- JSON/CSV/PDF/overlay downloads

## Expected outputs

```text
outputs/predictions/<case>_devansh_<variant>_mask.png
outputs/overlays/<case>_devansh_<variant>_overlay.png
outputs/reports/<case>_devansh_<variant>_report.json
```

## HC18 utilities

Audit HC18 dataset:

```bash
python scripts/01_audit_hc18.py --dataset-root /path/to/HC18 --out outputs/logs/hc18_audit.csv
```

Optional scratch training baseline:

```bash
python scripts/02_train_segmentation.py \
  --dataset-root /path/to/HC18 \
  --config configs/model_config.yaml \
  --output models/checkpoints/best_unet.pt
```

## Clinical-use note

This is an MVP/research demonstration. AI result must be reviewed by a licensed doctor. It is not a certified clinical device.
