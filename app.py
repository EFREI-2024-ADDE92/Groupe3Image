import torch
from torchvision import transforms
from flask import Flask, request, redirect, url_for, render_template
from PIL import Image
from ResNet import ResNet9Lighting
import io
import base64

model = ResNet9Lighting(3,6, 0.01, 0.01)
model.load_state_dict(torch.load('model.pth'))

app = Flask(__name__)


# Define the transformation to be applied to the input image
labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
def predict_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)
 
    # Apply the transformation to the image
    img_tensor = transform(img)
 
    # Add an extra batch dimension to the image tensor
    img_tensor = img_tensor.unsqueeze(0)
 
    # Perform the prediction
    with torch.no_grad():
        output = model(img_tensor)
 
    # Get the predicted class index
    _, predicted_index = torch.max(output, 1)
 
    # You can map the predicted index to a human-readable label based on your model
    # For simplicity, let's just return the predicted index in this example
    return {'prediction': labels[int(predicted_index)]}


@app.route('/', methods=['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
       if 'file' not in request.files:
           return 'No file part'
       file = request.files['file']
       if file.filename == '':
           return 'No selected file'
       if file:
           prediction = predict_image(file)
           image = Image.open(file.stream)
           buffered = io.BytesIO()
           image.save(buffered, format="JPEG")
           img_str = base64.b64encode(buffered.getvalue()).decode()
           return render_template('show_image_prediction.html', img_str=img_str, prediction=prediction)
   return render_template('show_image_prediction.html')

@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
   if request.method == 'POST':
       if 'file' not in request.files:
           return 'No file part'
       file = request.files['file']
       if file.filename == '':
           return 'No selected file'
       if file:
           prediction = predict_image(file)
           return prediction

if __name__ == '__main__':
    # Assuming your Flask app is named 'app'
    app.run(host = '0.0.0.0', port=5000)