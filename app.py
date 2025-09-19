import os
from flask import Flask, request, render_template, flash
import torch
import numpy as np
from PIL import Image
import io
import joblib
from model_def import ACEPnet
from sklearn.preprocessing import RobustScaler
import rasterio
from rasterio.io import MemoryFile
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Get the absolute path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Load model and scalers
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ACEPnet()
    
    # Load checkpoint with weights_only=False since we trust our own files
    checkpoint = torch.load(
        os.path.join(MODELS_DIR, 'best_model.pth'),
        map_location=device,
        weights_only=False  # This is safer than using pickle_module
    )
    
    # Extract model state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scalers from separate files
    feature_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.joblib'))
    target_scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.joblib'))
    
    print("Model and scalers loaded successfully!")
except Exception as e:
    print(f"Error loading model or scalers: {str(e)}")
    raise

def process_image(file):
    try:
        # Read the image file into memory
        image_data = file.read()
        memfile = MemoryFile(image_data)
        
        # Open as rasterio dataset
        with memfile.open() as dataset:
            # Read the first band
            image = dataset.read(1)
            
            # Resize using OpenCV
            import cv2
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            eps = 1e-8
            img_min = image.min()
            img_max = image.max()
            if abs(img_max - img_min) < eps:
                image_array = np.zeros((64, 64))
            else:
                image_array = (image - img_min) / (img_max - img_min + eps)
            
            # Add channel dimension and convert to tensor
            image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
            return image_tensor
            
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def is_valid_image_name(filename):
    """Validate image filename format (e.g., 'China_2013_06.tif')"""
    if not filename.lower().endswith('.tif'):
        return False
    try:
        name = filename[:-4]  # Remove .tif extension
        country, year, month = name.split('_')
        return (len(year) == 4 and len(month) == 2 and 
                year.isdigit() and month.isdigit() and
                1 <= int(month) <= 12)
    except:
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            population = float(request.form['population'])
            area = float(request.form['area'])
            image_file = request.files['image']

            if not image_file:
                flash('No image uploaded')
                return render_template('index.html')
                
            if not is_valid_image_name(image_file.filename):
                flash('Invalid image filename format. Expected format: Country_YYYY_MM.tif')
                return render_template('index.html')

            # Extract country and date from filename
            filename = image_file.filename
            country, year, month = filename.rsplit('.', 1)[0].split('_')

            # Process image
            image_tensor = process_image(image_file)
            
            # Process features (log transform and scale)
            features = np.array([[np.log1p(population), np.log1p(area)]])
            features_scaled = feature_scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled)

            # Make prediction
            with torch.no_grad():
                model.eval()
                prediction = model(image_tensor.to(device), features_tensor.to(device))
                
            # Transform prediction back (inverse scale and exp)
            prediction = target_scaler.inverse_transform(prediction.cpu().numpy())
            prediction = np.expm1(prediction)[0][0]

            return render_template('result.html', 
                                prediction=f"{prediction:.2f}",
                                population=f"{population:,.0f}",
                                area=f"{area:,.2f}",
                                country=country,
                                date=f"{month}/{year}",
                                year=datetime.now().year)

        except Exception as e:
            flash(f'Error: {str(e)}')
            return render_template('index.html')

    return render_template('index.html', year=datetime.now().year)

@app.route('/result')
def result():
    return render_template('result.html', year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True)
