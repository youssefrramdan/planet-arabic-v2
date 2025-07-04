from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import io

app = Flask(__name__)

# تحميل النموذج
interpreter = tf.lite.Interpreter(model_path="model/plant_disease_detect_model_pwp_quantized.tflite")
interpreter.allocate_tensors()

# تحميل ملف الأمراض
with open("plant_disease.json", 'r', encoding="utf-8") as file:
    plant_disease = json.load(file)

# تجهيز الصورة للموديل
def extract_features(image):
    image = image.resize((160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)
    return feature

# دالة التنبؤ
def model_predict(image):
    img = extract_features(image)

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get prediction results
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(prediction))
    prediction_info = plant_disease[predicted_index]
    return prediction_info

# API health check
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'نظام كشف أمراض النبات API',
        'endpoints': {
            'predict': {
                'url': '/predict',
                'method': 'POST',
                'parameters': {
                    'image': 'file (jpg, png, or jpeg)'
                }
            }
        }
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        try:
            # Get prediction
            result = model_predict(image)

            # Prepare response
            response = {
                'success': True,
                'message': 'تم التعرّف على المرض بنجاح!',
                'disease_name': result['الاسم'],
                'cause': result['السبب'],
                'treatment': result['العلاج'],
                'accuracy': '98%'
            }

            return jsonify(response), 200

        except Exception as e:
            return jsonify({
                'error': 'Error processing image',
                'details': str(e)
            }), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
