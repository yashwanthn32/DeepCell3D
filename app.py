from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)

# Load the trained model
# IMPORTANT: Ensure 'model.h5' is in the same directory as this app.py file.
try:
    model = load_model('model.h5')
    logging.info("Keras model 'model.h5' loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Keras model 'model.h5': {e}")
    # It's crucial that the model loads. If it fails, the app won't function.
    # sys.exit(1) # Uncomment this if you want the app to exit on model load failure

# Define folder paths
# UPLOAD_FOLDER is where images sent from the frontend will be saved.
UPLOAD_FOLDER = 'static/uploads'
# MODEL_DIRECTORY is where your .glb 3D models are located.
MODEL_DIRECTORY = 'static/models' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
# This creates the 'static/uploads' directory if it doesn't already exist.
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created upload folder: {UPLOAD_FOLDER}")

# Set up logging
# This helps in debugging by printing informative messages to the console.
logging.basicConfig(level=logging.INFO)

# Define class labels
# These labels must match the output classes of your trained model.
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Preprocess function
# This function prepares the uploaded image for prediction by your Keras model.
def preprocess_image(image_path):
    logging.info(f'Preprocessing image: {image_path}')
    # Load the image, resizing it to the target size expected by your model (224x224 for MobileNetV2).
    image = load_img(image_path, target_size=(224, 224))  
    # Convert the image to a NumPy array.
    image = img_to_array(image)
    # Apply MobileNetV2-specific preprocessing (e.g., scaling pixel values).
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  
    # Add a batch dimension (models expect input in batches, even if it's a single image).
    image = np.expand_dims(image, axis=0)  
    return image

# Route for the home page
# When a user accesses the root URL ('/'), render index.html.
@app.route('/')
def home():
    logging.info("Serving index.html")
    return render_template('index.html')

# Route for the project details page
# When a user accesses '/project_details.html', render project_details.html.
@app.route('/project_details.html')
def project_details():
    logging.info("Serving project_details.html")
    return render_template('project_details.html')

@app.route('/about_us.html')
def about_us():
    logging.info("Serving about_us.html")
    return render_template('about_us.html')
 
 #/contact_us.html
@app.route('/contact_us.html')
def contact_us():
    logging.info("Serving contact_us.html")
    
    return render_template('contact_us.html')


# Prediction route
# This route handles POST requests for image prediction.
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received prediction request.")
    # Check if 'image' key is in the request files.
    if 'image' not in request.files:
        logging.warning("No image file provided in request.")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    
    # Check if the filename is empty (no file selected).
    if file.filename == '':
        logging.warning("No image selected for uploading.")
        return jsonify({'error': 'No image selected for uploading'}), 400

    # If a file is present and has a valid filename
    if file:
        # Construct the full path to save the uploaded image.
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        # Save the file.
        file.save(filepath)
        logging.info(f'Image uploaded to: {filepath}')

        # Preprocess the image and get a prediction from the model.
        try:
            image = preprocess_image(filepath)
            prediction = model.predict(image)
            # Get the predicted class name using argmax on the prediction probabilities.
            predicted_class = class_labels[np.argmax(prediction)]
            logging.info(f'Predicted class: {predicted_class}')
            # Return the predicted class as a JSON response.
            return jsonify({'prediction': predicted_class})
        except Exception as e:
            logging.error(f"Error during image preprocessing or prediction: {e}")
            return jsonify({'error': f'Error during prediction: {e}'}), 500
    
    # Fallback for unexpected scenarios.
    logging.error("Unknown error occurred during file upload or prediction.")
    return jsonify({'error': 'Unknown error occurred'}), 500

# Serve 3D models from the static/models directory
# This route allows your <model-viewer> to load .glb files dynamically.
@app.route('/static/models/<path:filename>') # Changed from /models/<path:filename>
def serve_models(filename):
    logging.info(f"Serving 3D model: {filename}")
    # send_from_directory securely serves files from the specified directory.
    return send_from_directory(MODEL_DIRECTORY, filename)


# Main entry point for running the Flask application
if __name__ == '__main__':
    # Run the app in debug mode.
    # debug=True enables automatic reloading on code changes and provides a debugger.
    # port=5000 specifies the port the app will listen on.
    app.run(debug=True, port=5000)