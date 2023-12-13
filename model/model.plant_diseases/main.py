from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os  # Add this line to import the 'os' module

app = FastAPI()

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
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class label
    predicted_class = CLASS_NAMES[predicted_class_index]

    return predicted_class

@app.post("/predict-plant-diseases")
async def predict_plant_diseases(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    with open('temp_image.jpg', 'wb') as temp_image:
        temp_image.write(file.file.read())

    # Preprocess the image
    img_array = preprocess_image('temp_image.jpg')

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class label
    predicted_class = CLASS_NAMES[predicted_class_index]

    # Delete the temporary image file
    os.remove('temp_image.jpg')

    return JSONResponse(content={'predicted_class': predicted_class})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=5171)

print('Running...')
