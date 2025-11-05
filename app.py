import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os

# SỬA LỖI 1: Tách import ra đúng vị trí
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# Khởi tạo Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# SỬA LỖI 2: Dùng đường dẫn tương đối (chỉ cần tên file)
model_path = 'deepweeds_effnet_final.keras'
model = load_model(model_path)


# Danh sách các lớp (dựa trên bảng bạn cung cấp)
class_names_en = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", 
    "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative"
]
class_names_vi = [
    "Táo dại châu Phi", "Cỏ Ngũ sắc", "Cây Parkinsonia", "Cỏ ngải dại", 
    "Keo gai", "Dây cao su", "Cỏ Xiêm", "Cỏ đuôi rắn", "Ảnh nền tự nhiên"
]

# Hàm dự đoán ảnh
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    return predicted_class, confidence

# Route cho trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        predicted_class, confidence = predict_image(filepath)
        
        en_name = class_names_en[predicted_class]
        vi_name = class_names_vi[predicted_class]
        
        os.remove(filepath)
        
        return render_template('result.html', 
                               en_name=en_name, 
                               vi_name=vi_name, 
                               confidence=round(confidence * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)