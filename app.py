import torch
from torchvision import transforms
from flask import Flask, flash, request, render_template, Response
from PIL import Image
import io
import base64
from prometheus_client import Counter, generate_latest, REGISTRY
from ResNet import ResNet9Lighting

model = ResNet9Lighting.load_from_checkpoint('model.ckpt', map_location= {'cuda:0':'cpu'})

model.eval()
model.freeze()


app = Flask(__name__)



api_call_counter = Counter('api_calls_total', 'Total number of API calls')
api_call_counter_sea = Counter('api_predictions_sea', 'Total number of Sea predictions')
api_call_counter_building = Counter('api_predictions_building', 'Total number of Building predictions')
api_call_counter_forest = Counter('api_predictions_forest', 'Total number of Forest predictions')
api_call_counter_glacier = Counter('api_predictions_glacier', 'Total number of Glacier predictions')
api_call_counter_mountain = Counter('api_predictions_mountain', 'Total number of Mountain predictions')
api_call_counter_street = Counter('api_predictions_street', 'Total number of Street predictions')


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the transformation to be applied to the input image
labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

stats = ((0.4302, 0.4575, 0.4540), (0.2481, 0.2467, 0.2807))

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace = True)
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
    
    prediction = labels[int(predicted_index)]
    
    if prediction == 'buildings':
        api_call_counter_building.inc()
    elif prediction == 'forest':
        api_call_counter_forest.inc()
    elif prediction == 'glacier':
        api_call_counter_glacier.inc()
    elif prediction == 'mountain':
        api_call_counter_mountain.inc()
    elif prediction == 'sea':
        api_call_counter_sea.inc()
    else:
        api_call_counter_street.inc()

    probabilities = torch.nn.functional.softmax(output, dim=1)

    # You can map the predicted index to a human-readable label based on your model
    # For simplicity, let's just return the predicted index in this example
    return {'prediction': prediction, 'probability': probabilities[0][predicted_index].item()}


@app.route('/', methods=['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
       if 'file' not in request.files:
           flash('No file part')
       file = request.files['file']
       if file.filename == '':
           flash('No selected file')
       if file and allowed_file(file.filename):
           prediction = predict_image(file)
           choose_prediction = prediction['prediction'].capitalize()
           confidence_level = prediction['probability']
           api_call_counter.inc()
           image = Image.open(file.stream)
           buffered = io.BytesIO()
           image.save(buffered, format="JPEG")
           img_str = base64.b64encode(buffered.getvalue()).decode()
           return render_template('show_image_prediction.html', img_str=img_str, prediction=choose_prediction, confidence_level = confidence_level)
       else:
           flash('No file format allowed')
   return render_template('show_image_prediction.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        prediction = predict_image(file)
        api_call_counter.inc()
        return prediction

@app.route('/metrics', methods=['GET'])
def metrics():
    # Expose the metrics in Prometheus format
    return Response(generate_latest(REGISTRY), content_type='text/plain; version=0.0.4')

if __name__ == '__main__':
    # Assuming your Flask app is named 'app'
    app.run(host = '0.0.0.0', port=5000)