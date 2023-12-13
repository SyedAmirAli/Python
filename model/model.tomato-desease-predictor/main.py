import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os 

# Load the model with model build folder.
model = tf.keras.models.load_model('./model-v2/tomato_disease.h5')

# Defines The Class Names
# class_names = [
#   "Tomato___Bacterial_spot",
#   "Tomato___Early_blight",
#   "Tomato___Late_blight",
#   "Tomato___Leaf_Mold",
#   "Tomato___Septoria_leaf_spot",
#   "Tomato___Spider_mites",
#   "Tomato___Target_Spot",
#   "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#   "Tomato___Tomato_mosaic_virus",
#   "Tomato___healthy",
#   "Two-spotted_spider_mite",
# ]
class_names = sorted(os.listdir('./train'))
diseases_names = []

# make a Disease names Array
for class_name in class_names:
    part = class_name.split("___")
    disease_name = part[-1]
    disease_name = disease_name.replace("_", " ")
    diseases_names.append(disease_name)

# make a image preprocess function with a image path Load and preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    return img_array


# make a function to find out the exact diseases's class from `class_names` Array!
def predict_class(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class label
    predicted_class = class_names[predicted_class_index]
    disease_name = diseases_names[predicted_class_index]
    return disease_name, predicted_class, predictions[0][predicted_class_index]

 
""" 
 # Define a demo image's path
 image_path = '/home/amir/Desktop/Python/model.tomato-from-chrome/train/Tomato___Leaf_Mold/ 5c58dd13-93e0-4e77-b653-ba1f6833a67e___Crnl_L.Mold 8712.JPG'
 
 # get the confidence and predicted class to call the `predict_class` function with a image  pictures
 disease_name, predicted_class, confidence = predict_class(image_path)
 
 # print the diseases's name and accuracy level from the imputed image's.
 print(f'\nPredicted Class: {predicted_class} with confidence: {confidence * 100:.2f}%\n') 
 
"""
 
