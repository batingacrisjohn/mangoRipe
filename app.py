import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── Load model once at startup ──────────────────────────────────────────────
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "EfficientNetB0_mango_ripeness.h5")

print(f"[startup] Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()
print("[startup] Model ready ✓")

# ── Class labels — must match the order your model was trained on ───────────
CLASS_LABELS = ["partially_ripe", "ripe", "rotten", "unripe"]

# EfficientNetB0 expects 224×224 RGB images
IMG_SIZE = (224, 224)

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes → normalised (1, 224, 224, 3) float32 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # EfficientNetB0 preprocessing: scale to [-1, 1]
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # add batch dim


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "EfficientNetB0 ready ✓"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        tensor = preprocess(image_bytes)
        preds = model.predict(tensor, verbose=0)[0]          # shape (4,)

        results = {
            label: float(round(float(score) * 100, 2))
            for label, score in zip(CLASS_LABELS, preds)
        }

        top_label = CLASS_LABELS[int(np.argmax(preds))]
        top_confidence = float(round(float(np.max(preds)) * 100, 2))

        return jsonify({
            "prediction": top_label,
            "confidence": top_confidence,
            "scores": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
