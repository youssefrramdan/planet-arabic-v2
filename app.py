from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import json
import uuid
from PIL import Image
import os

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/plant_disease_detect_model_pwp_quantized.tflite")
interpreter.allocate_tensors()

# Load the labels
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# Function to preprocess the image
def extract_features(image):
    image = image.resize((160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)
    return feature

# Prediction function
def model_predict(image):
    img = extract_features(image)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    prediction_label = plant_disease[np.argmax(prediction)]
    return prediction_label

# Serve uploaded image (optional)
@app.route('/uploadimages/<path:filename>', methods=['GET'])
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

# API health check
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Plant disease prediction API is running."})

# Prediction endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['img']
    temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
    os.makedirs("uploadimages", exist_ok=True)
    image_path = os.path.join(temp_name)

    image.save(image_path)

    try:
        img = Image.open(image_path)
        prediction = model_predict(img)
        return jsonify({
            'prediction': prediction,
            'image_url': f'/uploadimages/{os.path.basename(image_path)}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)  # Remove file after prediction

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
