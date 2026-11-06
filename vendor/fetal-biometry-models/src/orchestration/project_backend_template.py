"""
Copy this file to:
    src/orchestration/project_backend.py

Then replace each NotImplementedError with a call into your already frozen
Phase-0/1/2/4/5 code. Do not retrain models here.
"""
from src.biometry.estimated_fetal_weight import estimate_fetal_weight_safe

class ProjectBackend:
    def __init__(self):
        # Load/cache all real model objects here once.
        pass

    def detect_plane(self,image):
        # Phase 1 -> dict with predicted_plane, application_plane, status,
        # top_score, margin
        raise NotImplementedError

    def assess_quality(self,image):
        # Phase 2 -> dict with label good/review/poor, score, margin, mode
        raise NotImplementedError

    def measure_head(self,image):
        # Phase 0 -> dict with status, confidence, hc_mm, bpd_mm, ofd_mm, reasons
        raise NotImplementedError

    def measure_abdomen(self,image):
        # Phase 4 -> dict with status, confidence, tad_mm, apad_mm, ac_mm, reasons
        raise NotImplementedError

    def measure_femur(self,image):
        # Phase 5 -> dict with status, confidence, fl_mm, reasons
        raise NotImplementedError

    def calculate_efw(self,hc_mm,ac_mm,fl_mm):
        return estimate_fetal_weight_safe(hc_mm,ac_mm,fl_mm)
