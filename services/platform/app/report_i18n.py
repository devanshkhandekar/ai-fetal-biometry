from __future__ import annotations


LABELS = {
    "en": {
        "measurements": "Measurements",
        "measurement": "Measurement",
        "value": "Value",
        "unit": "Unit",
        "status": "Status",
        "warnings": "Warnings / Review Notes",
        "estimated_weight": "Estimated Fetal Weight",
    },
    "hi": {
        "measurements": "माप",
        "measurement": "माप",
        "value": "मान",
        "unit": "इकाई",
        "status": "स्थिति",
        "warnings": "चेतावनी / समीक्षा नोट्स",
        "estimated_weight": "अनुमानित भ्रूण वजन",
    },
    "gu": {
        "measurements": "માપ",
        "measurement": "માપ",
        "value": "મૂલ્ય",
        "unit": "એકમ",
        "status": "સ્થિતિ",
        "warnings": "ચેતવણીઓ / સમીક્ષા નોંધો",
        "estimated_weight": "અંદાજિત ભ્રૂણ વજન",
    },
    "ta": {
        "measurements": "அளவீடுகள்",
        "measurement": "அளவீடு",
        "value": "மதிப்பு",
        "unit": "அலகு",
        "status": "நிலை",
        "warnings": "எச்சரிக்கைகள் / மதிப்பாய்வு குறிப்புகள்",
        "estimated_weight": "மதிப்பிடப்பட்ட கருவின் எடை",
    },
    "te": {
        "measurements": "కొలతలు",
        "measurement": "కొలత",
        "value": "విలువ",
        "unit": "యూనిట్",
        "status": "స్థితి",
        "warnings": "హెచ్చరికలు / సమీక్ష గమనికలు",
        "estimated_weight": "అంచనా భ్రూణ బరువు",
    },
    "ar": {
        "measurements": "القياسات",
        "measurement": "القياس",
        "value": "القيمة",
        "unit": "الوحدة",
        "status": "الحالة",
        "warnings": "التحذيرات / ملاحظات المراجعة",
        "estimated_weight": "الوزن التقديري للجنين",
    },
    "fr": {
        "measurements": "Mesures",
        "measurement": "Mesure",
        "value": "Valeur",
        "unit": "Unité",
        "status": "Statut",
        "warnings": "Avertissements / notes de révision",
        "estimated_weight": "Poids fœtal estimé",
    },
    "es": {
        "measurements": "Mediciones",
        "measurement": "Medición",
        "value": "Valor",
        "unit": "Unidad",
        "status": "Estado",
        "warnings": "Advertencias / notas de revisión",
        "estimated_weight": "Peso fetal estimado",
    },
}


ALIASES = {
    "english": "en",
    "en": "en",
    "hindi": "hi",
    "hi": "hi",
    "gujarati": "gu",
    "gu": "gu",
    "tamil": "ta",
    "ta": "ta",
    "telugu": "te",
    "te": "te",
    "arabic": "ar",
    "ar": "ar",
    "french": "fr",
    "fr": "fr",
    "spanish": "es",
    "es": "es",
}


def get_report_labels(language: str | None) -> dict[str, str]:
    key = (language or "English").strip().lower()
    key = ALIASES.get(key, "en")
    return LABELS[key]
