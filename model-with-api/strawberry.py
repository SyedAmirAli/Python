import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os 

model = tf.keras.models.load_model('strawberry-model/model.h5')

class_names = [
  "Strawberry___Angular_Leaf_Spot",
  "Strawberry___Anthracnose_Fruit",
  "Strawberry___Blossom",
  "Strawberry___Gray mold",
  "Strawberry___Leaf_Spot",
  "Strawberry___Leaf_scorch",
  "Strawberry___Powdery_Mildew_Fruit",
  "Strawberry___Powdery_Mildew_Leaf",
  "Strawberry___healthy",
];

diseases_names = []

for class_name in class_names:
    part = class_name.split("___")
    disease_name = part[-1]
    disease_name = disease_name.replace("_", " ")
    diseases_names.append(disease_name)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array


def strawberry_predict_class(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    predicted_class = class_names[predicted_class_index]
    disease_name = diseases_names[predicted_class_index]
    return disease_name, predicted_class, predictions[0][predicted_class_index]

 
