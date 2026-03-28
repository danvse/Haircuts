import torch
import clip
from PIL import Image
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open("popularity.json") as f:
    popularity = json.load(f)

labels = list(popularity.keys())
text_tokens = clip.tokenize(labels).to(device)

def classify(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        logits = (image_features @ text_features.T).softmax(dim=-1)
        prob, idx = logits[0].max(dim=0)

    label = labels[idx]
    pop = popularity[label]

    if pop < 0.5:
        return "YOU'RE CHOPPED", label, pop

    return "GOOD HAIRCUT", label, pop
