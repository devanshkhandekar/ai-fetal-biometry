"""
Prompt ensemble for Phase-1 fetal plane detection.

Important:
- FetalCLIP's published zero-shot standard-plane benchmark excludes the broad
  FETAL_PLANES_DB "Other" class.
- We therefore classify FIVE standard views here and treat "Other" as an
  open-set/rejection test during validation.
- These prompts are caption-style prompts modeled on the prompt strategy
  described by the FetalCLIP paper/repository. They are intentionally kept in
  one file so you can replace them with the exact prompt list from your cloned
  external/FetalCLIP/zero_shot_planes_db code if you want strict reproduction.
"""

STANDARD_CLASSES = ("abdomen", "brain", "femur", "thorax", "cervix")

PROMPT_ENSEMBLES = {
    "abdomen": [
        "Fetal ultrasound image focusing on the abdomen and its anatomical structures.",
        "Ultrasound scan showing a standard fetal abdominal plane.",
        "Sonographic image highlighting the fetal abdominal area.",
        "Prenatal ultrasound view centered on the fetal abdomen.",
        "Obstetric ultrasound image showing the fetal abdomen in a standard view.",
    ],
    "brain": [
        "Fetal ultrasound image focusing on the brain and cranial structures.",
        "Ultrasound scan showing a standard fetal brain plane.",
        "Sonographic image highlighting fetal brain anatomy.",
        "Prenatal ultrasound view centered on the fetal head and brain.",
        "Obstetric ultrasound image showing the fetal brain in a standard view.",
    ],
    "femur": [
        "Fetal ultrasound image focusing on the femur.",
        "Ultrasound scan showing the fetal femur and long bone.",
        "Sonographic image highlighting the fetal femoral shaft.",
        "Prenatal ultrasound view centered on the fetal femur.",
        "Obstetric ultrasound image showing the fetal femur in a standard view.",
    ],
    "thorax": [
        "Fetal ultrasound image focusing on the thorax and chest structures.",
        "Ultrasound scan showing a standard fetal thoracic plane.",
        "Sonographic image highlighting the fetal chest and thorax.",
        "Prenatal ultrasound view centered on the fetal thorax.",
        "Obstetric ultrasound image showing the fetal thorax in a standard view.",
    ],
    "cervix": [
        "Obstetric ultrasound image focusing on the maternal cervix.",
        "Ultrasound scan showing the maternal cervical region.",
        "Sonographic image highlighting the cervix during pregnancy.",
        "Prenatal ultrasound view centered on the maternal cervix.",
        "Obstetric ultrasound image showing the cervix in a standard view.",
    ],
}

APPLICATION_MAPPING = {
    "brain": "head",
    "abdomen": "abdomen",
    "femur": "femur",
    "thorax": "unsupported",
    "cervix": "unsupported",
}
