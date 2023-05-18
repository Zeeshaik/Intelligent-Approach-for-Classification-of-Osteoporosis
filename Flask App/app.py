from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

img_size = 256
model = tf.keras.models.load_model('C:/Users/zeesh/OneDrive/Documents/Projects/Intelligent-Approach-for-Classification-of-Osteoporosis/best_model.h5')
categories = ['Normal', 'Doubtful', 'Moderate', 'Mild', 'Severe']

def predict(img_path):
    img = Image.open(img_path).convert('L')  # convert to grayscale
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0  # normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    predictions_single = model.predict(img)
    predicted_category = categories[np.argmax(predictions_single)]
    return predicted_category

def get_precaution(prediction):
    precautions = {
        'Normal': 'No specific precautions needed.',
        'Doubtful': 'Consult a healthcare professional for further evaluation.',
        'Moderate': 'Take necessary precautions and consider medical advice.',
        'Mild': 'Take necessary precautions and consider medical advice.',
        'Severe': 'Seek immediate medical attention.'
    }
    return precautions.get(prediction)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file found"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            file_path = "C:/Users/zeesh/OneDrive/Documents/Projects/Intelligent-Approach-for-Classification-of-Osteoporosis/Flask App/static/uploads/" + file.filename
            file.save(file_path)
            prediction = predict(file_path)
            precaution = get_precaution(prediction)
            res = prediction + " detected " + precaution
            return render_template("result.html", prediction = prediction, precaution=precaution)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)
