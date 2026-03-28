from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import json

app = FastAPI()

checkpoint = torch.load("../model/haircut_model.pth", map_location="cpu")
class_to_idx = checkpoint["classes"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

with open("../model/popularity.json") as f:
    popularity = json.load(f)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(checkpoint["model"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/classify")
async def classify(file: UploadFile):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        max_prob, idx = probs.max(dim=1)

    label = idx_to_class[idx.item()]
    pop = popularity.get(label, 0)

    if pop < 0.5:
        return {"status": "bad", "message": "YOU'RE CHOPPED"}

    return {"status": "good", "haircut": label}
