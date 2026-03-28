import torch
import clip
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# -----------------------------
# Load CLIP Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# Load Haircut Classes
# -----------------------------
with open("classes.json") as f:
    classes = json.load(f)

# Convert to CLIP text tokens
text_tokens = clip.tokenize(classes).to(device)

# -----------------------------
# Load Popularity Scores
# -----------------------------
with open("popularity.json") as f:
    popularity = json.load(f)

# -----------------------------
# Classification Function
# -----------------------------
def classify_image(image_path, conf_threshold=0.6, pop_threshold=0.5):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    max_prob = float(probs.max())
    idx = int(probs.argmax())
    label = classes[idx]

    # Unknown haircut
    if max_prob < conf_threshold:
        return {
            "status": "bad",
            "message": "YOU'RE CHOPPED",
            "reason": "Unknown haircut",
            "confidence": max_prob
        }

    # Popularity check
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
# Flask API
# -----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "uploaded_image.png"
    file.save(filepath)

    result = classify_image(filepath)
    return jsonify(result)

if __name__ == "__main__":
    print("CLIP model loaded. Server running at http://127.0.0.1:5000")
    app.run(debug=True)
