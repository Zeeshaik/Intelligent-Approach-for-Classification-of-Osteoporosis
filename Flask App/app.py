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
    # Prediction 
    predictions_single = model.predict(img)
    predicted_category = categories[np.argmax(predictions_single)]
    return predicted_category

def get_precaution(prediction):
    precautions = {
        'Normal': ["Maintain a balanced and nutritious diet to support bone health.",
                   " Engage in regular weight-bearing exercises like walking, jogging, or dancing to strengthen bones.",
                   " Avoid excessive alcohol consumption and smoking, as they can contribute to bone loss.", 
                   " Get regular check-ups and bone density tests as recommended by your healthcare provider."],
        'Doubtful': [" Consult a healthcare professional for further evaluation and diagnosis."," Follow any additional tests or screenings recommended by your healthcare provider."," Maintain a healthy lifestyle with a focus on nutrition, exercise, and overall well-being."],
        'Moderate': [" Take necessary precautions to prevent falls, such as removing hazards at home, using assistive devices, and ensuring proper lighting."," Follow the recommendations of your healthcare provider regarding medication, supplements, and physical therapy."," Engage in exercises that focus on balance, strength, and flexibility to reduce the risk of fractures."," Consider modifications in daily activities to prevent excessive strain on the bones."],
        'Mild': [" Take necessary precautions similar to those for the moderate category."," Follow the advice of your healthcare provider regarding lifestyle modifications, medication, and therapeutic interventions."," Engage in exercises that are appropriate for your condition and focus on improving bone strength and flexibility."," Ensure an adequate intake of calcium, vitamin D, and other essential nutrients for bone health."],
        'Severe': [" Seek immediate medical attention and follow the guidance of healthcare professionals.",
                   " Adhere strictly to the prescribed treatment plan, including medication, therapy, and lifestyle modifications.",
                   " Take precautions to minimize the risk of falls and fractures, such as using assistive devices and making necessary home modifications.",
                   " Engage in physical activities as recommended by your healthcare provider, considering the limitations of your condition."]
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
            # res = prediction + " detected " + precaution
            return render_template("result.html", prediction = prediction, precaution=precaution)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)
