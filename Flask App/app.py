from flask import Flask, render_template, request
from flask import Flask, render_template, request, send_from_directory
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2,os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print(os.getcwd())

app = Flask(__name__)

img_size = 256
model = tf.keras.models.load_model('C:/Users/zeesh/OneDrive/Documents/Projects/Intelligent-Approach-for-Classification-of-Osteoporosis/Flask App/best_model.h5')
categories = ['Normal', 'Doubtful', 'Moderate', 'Mild', 'Severe']
data_path = "C:/Users/zeesh/OneDrive/Documents/Projects/Intelligent-Approach-for-Classification-of-Osteoporosis/DataSet"
categories_test=os.listdir(data_path)
labels=[i for i in range(len(categories_test))]

label_dict=dict(zip(categories_test,labels)) #empty dictionary
print(label_dict)
print(categories_test)
print(labels)
prediction = None   
data=[]
label=[]
for category in categories_test:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
                
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the image  into 256 x 256, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            label.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)
        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image
                    
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
label=np.array(label)
from keras.utils import np_utils
new_label=np_utils.to_categorical(label)







def predict(img_path):
    img = Image.open(img_path).convert('L')  # convert to grayscale
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0  # normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    # Prediction 
    predictions_test_single = model.predict(img)
    predicted_category = categories[np.argmax(predictions_test_single)]
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
    global prediction
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
            return render_template("result.html", prediction = prediction, precaution=precaution, image_file=file.filename)
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

@app.route("/about")
def about():
    return render_template("about.html")

# Route for statistics page
@app.route('/statistics')
def statistics():
    global prediction
    x_train,x_test,y_train,y_test=train_test_split(data,new_label,test_size=0.1)
    # Calculate the statistics for each uploaded image
    statistics = {}
    
    
    # Calculate accuracy
    test_labels = np.argmax(y_test, axis=1)
    predictions_test = model.predict(x_test)
    predictions_test = np.argmax(predictions_test, axis=-1)
    cm = confusion_matrix(test_labels, predictions_test)

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(5), ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe'], fontsize=16)
    plt.yticks(range(5), ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe'], fontsize=16)
    plt.savefig('C:/Users/zeesh/OneDrive/Documents/Projects/Intelligent-Approach-for-Classification-of-Osteoporosis/Flask App/static/uploads/confusion_matrix.png')

     # Calculate statistics for each uploaded image
    statistics = {
        'Accuracy': accuracy_score(test_labels, predictions_test),
        'Precision': precision_score(test_labels, predictions_test, average='macro'),
        'Recall': recall_score(test_labels, predictions_test, average='macro'),
        'F1 Score': f1_score(test_labels, predictions_test, average='macro'),
        'Confusion Matrix': cm.tolist()
    }

    # Render the statistics.html template and pass the statistics dictionary
    return render_template('statistics.html', statistics=statistics, prediction=prediction)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port='5000', debug=True)

