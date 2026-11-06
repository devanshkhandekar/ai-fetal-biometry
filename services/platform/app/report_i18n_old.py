from __future__ import annotations


LABELS = {

    "en": {
        "measurements":
            "Measurements",

        "measurement":
            "Measurement",

        "value":
            "Value",

        "unit":
            "Unit",

        "status":
            "Status",

        "estimated_weight":
            "Estimated Fetal Weight",

        "warnings":
            "Warnings / Review Notes",
    },

    "hi": {
        "measurements":
            "माप",

        "measurement":
            "माप",

        "value":
            "मान",

        "unit":
            "इकाई",

        "status":
            "स्थिति",

        "estimated_weight":
            "अनुमानित भ्रूण वजन",

        "warnings":
            "चेतावनी / समीक्षा नोट्स",
    },

    "gu": {
        "measurements":
            "માપ",

        "measurement":
            "માપ",

        "value":
            "મૂલ્ય",

        "unit":
            "એકમ",

        "status":
            "સ્થિતિ",

        "estimated_weight":
            "અંદાજિત ભ્રૂણ વજન",

        "warnings":
            "ચેતવણીઓ / સમીક્ષા નોંધો",
    },

    "fr": {
        "measurements":
            "Mesures",

        "measurement":
            "Mesure",

        "value":
            "Valeur",

        "unit":
            "Unité",

        "status":
            "Statut",

        "estimated_weight":
            "Poids fœtal estimé",

        "warnings":
            "Avertissements / notes",
    },

    "es": {
        "measurements":
            "Mediciones",

        "measurement":
            "Medición",

        "value":
            "Valor",

        "unit":
            "Unidad",

        "status":
            "Estado",

        "estimated_weight":
            "Peso fetal estimado",

        "warnings":
            "Advertencias / notas",
    },

}


def get_report_labels(
    language,
):

    key = (
        language
        or
        "en"
    ).lower()

    aliases = {
        "english": "en",
        "hindi": "hi",
        "gujarati": "gu",
        "french": "fr",
        "spanish": "es",
    }

    key = aliases.get(
        key,
        key,
    )

    return LABELS.get(
        key,
        LABELS["en"],
    )
