from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes


# Load fine-tuned model & processor
model = BlipForConditionalGeneration.from_pretrained("./fine_tuned_blip").to("cpu")
processor = BlipProcessor.from_pretrained("./fine_tuned_blip")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load image
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Process & predict
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs)

    # Decode caption
    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)
