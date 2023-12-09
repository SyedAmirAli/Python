import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('/home/amir/Desktop/Python/model.tomato-model/tomato-plant-diseases-detector/model.h5')

# Class names
class_names = sorted(os.listdir('/home/amir/Desktop/Python/model.tomato-model/plant-dataset/tomato-dataset/train/'))

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    return img_array

def predict_class(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class label
    predicted_class = class_names[predicted_class_index]

    return predicted_class, predictions[0][predicted_class_index]

image_path = '/home/amir/Desktop/Python/model.tomato-from-chrome/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/d0be6c81-9b1e-43e2-a468-373cd0987885___UF.GRC_YLCV_Lab 01907.JPG'

predicted_class, confidence = predict_class(image_path)
print(f'\nPredicted Class: {predicted_class} with confidence: {confidence * 100:.2f}%\n')
# print(class_names)

# Example usage
# image_path_1 = './plant-dataset/tomato-dataset/val/Tomato___healthy/01c1da17-8d9f-4d69-8a1e-58d37453d3c3___RS_HL 9641.JPG'
# image_path_2 = './plant-dataset/tomato-dataset/val/Tomato___Bacterial_spot/01a3cf3f-94c1-44d5-8972-8c509d62558e___GCREC_Bact.Sp 3396.JPG'
# image_path_3 = './plant-dataset/tomato-dataset/val/Tomato___Early_blight/00c5c908-fc25-4710-a109-db143da23112___RS_Erly.B 7778.JPG'
# image_path_4 = './plant-dataset/tomato-dataset/val/Tomato___Late_blight/00ce4c63-9913-4b16-898c-29f99acf0dc3___RS_Late.B 4982.JPG'
# image_path_5 = './plant-dataset/tomato-dataset/val/Tomato___Leaf_Mold/02a29ab9-8cba-47a0-bc2f-e7af7dbae149___Crnl_L.Mold 7165.JPG'
# image_path_6 = './plant-dataset/tomato-dataset/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/1af07f2b-027b-4792-80c5-2c20a4ed538c___YLCV_NREC 0179.JPG'
# predicted_class_1, confidence_1 = predict_class(image_path_1)
# print(f'\nPredicted Class: {predicted_class_1} with confidence: {confidence_1 * 100:.2f}%\n')
    
# predicted_class_2, confidence_2 = predict_class(image_path_2)
# print(f'Predicted Class: {predicted_class_2} with confidence: {confidence_2 * 100:.2f}%\n')
    
# predicted_class_3, confidence_3 = predict_class(image_path_3)
# print(f'Predicted Class: {predicted_class_3} with confidence: {confidence_3 * 100:.2f}%\n')
    
# predicted_class_4, confidence_4 = predict_class(image_path_4)
# print(f'Predicted Class: {predicted_class_4} with confidence: {confidence_4 * 100:.2f}%\n')
    
# predicted_class_5, confidence_5 = predict_class(image_path_5)
# print(f'Predicted Class: {predicted_class_5} with confidence: {confidence_5 * 100:.2f}%\n')
    
# predicted_class_6, confidence_6 = predict_class(image_path_6)
# print(f'Predicted Class: {predicted_class_6} with confidence: {confidence_6 * 100:.2f}%\n')
