import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

app = Flask(__name__)

model = load_model('Updated-Xception-diabetic-retinopathy.h5')
# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Register page
@app.route('/register')
def register():
    return render_template('register.html')

# Login page
@app.route('/login')
def login():
    return render_template('login.html')

# Prediction page (UI)
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# 🔴 IMPORTANT: Image upload route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return f"Image uploaded successfully: {filename}"

# Logout page
@app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    labels = {
        0: "No Diabetic Retinopathy",
        1: "Mild Diabetic Retinopathy",
        2: "Moderate Diabetic Retinopathy",
        3: "Severe Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }

    result = labels[class_index]

    return render_template(
        'prediction.html',
        prediction_result=result,
        image_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True)
