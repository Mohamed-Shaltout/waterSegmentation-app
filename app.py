from flask import Flask, request, render_template, send_from_directory
import os
import torch
import numpy as np
import cv2
from utils import preprocess_image, postprocess_mask, load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('best_model.pth', device)
model.eval()

@app.route('/')
def index():
    return '''
    <h1>Upload Satellite Image</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess input
    image_tensor = preprocess_image(filepath).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Postprocess and save
    mask_path = os.path.join(OUTPUT_FOLDER, 'mask_' + file.filename)
    postprocess_mask(pred_mask, mask_path)

    return f'''
    <h2>Prediction Complete</h2>
    <p>Original:</p><img src="/static/uploads/{file.filename}" width=300><br>
    <p>Predicted Mask:</p><img src="/static/outputs/mask_{file.filename}" width=300><br>
    <a href="/">Back</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)
