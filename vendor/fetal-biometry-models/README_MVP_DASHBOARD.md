# Complete MVP Dashboard Quick Start

## Minimal commands

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD
streamlit run app/streamlit_app.py
```

## Recommended model

```text
models/checkpoints/4a_best_pruned_ft_v10.pth
variant: phase4a
```

## Dashboard sections

- Single Scan
- Batch Demo
- MVP Architecture

After a scan is processed, the result area shows:

- Doctor Review
- Patient Summary
- GA / Growth
- Technical Audit
- JSON / Downloads

## What it demonstrates

- Input loading for image/DICOM
- Validity gating
- Pretrained segmentation
- Mask cleanup
- Ellipse fitting
- HC/BPD/OFD/head-area calculation
- Optional mm conversion using pixel spacing
- GA lookup through configurable chart
- Quality/confidence scoring
- Accept/review/reject triage
- Overlay visualization
- JSON, CSV, overlay PNG, and simple PDF downloads

Clinical-use note: MVP/research demo only. AI result must be reviewed by a licensed doctor.
