import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json

# -----------------------------
# Load Model + Class Mapping
# -----------------------------
checkpoint = torch.load("haircut_model.pth", map_location="cpu")
class_to_idx = checkpoint["classes"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

# -----------------------------
# Load Popularity Scores
# -----------------------------
with open("popularity.json") as f:
    popularity = json.load(f)

# -----------------------------
# Build Model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(checkpoint["model"])
model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Classification Function
# -----------------------------
def classify_image(image_path, conf_threshold=0.6, pop_threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        max_prob, idx = probs.max(dim=1)

    max_prob = max_prob.item()
    idx = idx.item()
    label = idx_to_class[idx]

    # Unknown haircut (low confidence)
    if max_prob < conf_threshold:
        return {
            "status": "bad",
            "message": "YOU'RE CHOPPED",
            "reason": "Unknown haircut",
            "confidence": max_prob
        }

    # Known haircut — check popularity
    pop_score = popularity.get(label, 0)

    if pop_score < pop_threshold:
        return {
            "status": "bad",
            "message": "YOU'RE CHOPPED",
            "haircut": label,
            "popularity": pop_score,
            "confidence": max_prob
        }

    return {
        "status": "good",
        "haircut": label,
        "popularity": pop_score,
        "confidence": max_prob
    }


# -----------------------------
# Manual Test
# -----------------------------
if __name__ == "__main__":
    result = classify_image("Wavy.png")
    print(result)
