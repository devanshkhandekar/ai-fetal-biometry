from src.biometry.estimated_fetal_weight import estimate_fetal_weight_safe

class DemoBackend:
    def __init__(self, anatomy="head", quality="good"):
        self.anatomy=anatomy; self.quality=quality

    def detect_plane(self,image):
        m={
            "head":("brain","head"),
            "abdomen":("abdomen","abdomen"),
            "femur":("femur","femur"),
            "thorax":("thorax","unsupported"),
            "other":("other","unknown")
        }
        pred,app=m[self.anatomy]
        return {"predicted_plane":pred,"application_plane":app,"status":"accept","top_score":0.98,"margin":0.8}

    def assess_quality(self,image):
        return {"label":self.quality,"score":0.91,"mode":"advisory"}

    def measure_head(self,image):
        return {"status":"valid","confidence":0.91,"hc_mm":320.0,"bpd_mm":82.0,"ofd_mm":102.0}

    def measure_abdomen(self,image):
        return {"status":"valid","confidence":0.90,"tad_mm":95.0,"apad_mm":96.0,"ac_mm":300.0}

    def measure_femur(self,image):
        return {"status":"valid","confidence":0.93,"fl_mm":60.0}

    def calculate_efw(self,hc_mm,ac_mm,fl_mm):
        return estimate_fetal_weight_safe(hc_mm,ac_mm,fl_mm)
