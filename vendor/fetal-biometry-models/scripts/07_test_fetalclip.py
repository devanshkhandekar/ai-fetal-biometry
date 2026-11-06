import json
import torch
import open_clip

CONFIG = "models/checkpoints/fetalclip/FetalCLIP_config.json"
WEIGHTS = "models/checkpoints/fetalclip/FetalCLIP_weights.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

with open(CONFIG) as f:
    config = json.load(f)

open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config

model, preprocess_train, preprocess_test = \
    open_clip.create_model_and_transforms(
        "FetalCLIP",
        pretrained=WEIGHTS
    )

tokenizer = open_clip.get_tokenizer("FetalCLIP")

model.eval()
model.to(device)

print("FetalCLIP loaded successfully")
print("Device:", device)