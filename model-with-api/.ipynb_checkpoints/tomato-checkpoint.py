import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os 

model = tf.keras.models.load_model('tomato-model/model.h5')

class_names = [
  "Tomato___Bacterial_spot",
  "Tomato___Early_blight",
  "Tomato___Late_blight",
  "Tomato___Leaf_Mold",
  "Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites",
  "Tomato___Target_Spot",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato___Tomato_mosaic_virus",
  "Tomato___healthy",
  "Two-spotted_spider_mite",
]

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


def tomato_predict_class(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    predicted_class = class_names[predicted_class_index]
    disease_name = diseases_names[predicted_class_index]
    return disease_name, predicted_class, predictions[0][predicted_class_index]

 
