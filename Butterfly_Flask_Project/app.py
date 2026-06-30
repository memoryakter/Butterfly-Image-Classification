import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__, template_folder="template")

IMG_SIZE = (224, 224)

# -----------------------------
# CLASS LABELS (75 classes)
# -----------------------------
import json

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# reverse mapping: index → class name
class_names = {v: k for k, v in class_indices.items()}

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("butterfly_model.keras")


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        # Load image
        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)

        # Convert to array
        img = np.array(img)

        # IMPORTANT: EfficientNet preprocessing
        img = preprocess_input(img)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result = class_names[index]

        return render_template(
            "index.html",
            predicted_text=f"Predicted: {result} ({confidence:.2f})"
        )

    except Exception as e:
        return render_template(
            "index.html",
            predicted_text=f"Error: {str(e)}"
        )


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)