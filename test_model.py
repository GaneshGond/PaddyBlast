from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image as PILImage
import os

# Load your model
model = load_model('paddy_disease_diagnosis_model.h5')
IMG_SIZE = (128, 128)
CLASS_LABELS = ['Healthy', 'Mildly Diseased', 'Moderately Diseased', 'Severely Diseased']
SOLUTIONS = {
    'Healthy': "No action required. The plant is healthy.",
    'Mildly Diseased': "Consider applying organic pesticides and monitoring the plant regularly.",
    'Moderately Diseased': "Apply appropriate chemical treatment and isolate the plant if possible.",
    'Severely Diseased': "Remove the infected leaves or plant to prevent spreading and apply intensive treatment."
}

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    predicted_class = CLASS_LABELS[class_index]
    solution = SOLUTIONS[predicted_class]
    return predicted_class, confidence, solution

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result, confidence, solution = predict_image(file_path)
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2f}",
            'solution': solution
        })

if __name__ == '__main__':
    app.run(debug=True)
