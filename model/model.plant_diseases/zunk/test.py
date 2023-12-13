from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# Load the trained model
model_path = './keras-model/model.h5'
model = load_model(model_path)

# Class names
CLASS_NAMES = [
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_class(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1) 

    # Get the predicted class label
    pd_class = CLASS_NAMES[predicted_class_index[0]]

    return pd_class

# Example usage
base_path = '/home/amir/Downloads/zip/ai-model/valid/Strawberry___healthy/'
image_path = base_path + '0f79c593-bcf2-4a5b-baac-6433f3037a89___RS_HL 2022_new30degFlipLR.JPG'

predicted_class = predict_class(image_path)
print(f'Predicted Class: { predicted_class }')