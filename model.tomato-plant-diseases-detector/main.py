import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('./model/tomato_disease_model.h5')

# Class names
class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Two-spotted_spider_mite']

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

# image_path = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/d0d6f6ab-9a4e-41ed-90eb-171c7f7d6bc7___UF.GRC_YLCV_Lab 02692.JPG'

# image_path = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/train/Two-spotted_spider_mite/e136c0bb-0587-431b-b412-0abe8eea294a___Com.G_SpM_FL 8919.JPG'

image_path = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/val/Tomato___healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG'

predicted_class, confidence = predict_class(image_path)
print(f'\nPredicted Class: {predicted_class} with confidence: {confidence * 100:.2f}%\n')


# print(class_names)

# Example usage
# image_path_1 = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/val/Tomato___Bacterial_spot/0ce4dee0-2d0c-4c25-a9b2-4abc5f9083db___GCREC_Bact.Sp 3708.JPG'

# image_path_2 = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/val/Tomato___Early_blight/00c5c908-fc25-4710-a109-db143da23112___RS_Erly.B 7778.JPG'

# image_path_3 = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/val/Tomato___healthy/0a334ae6-bea3-4453-b200-85e082794d56___GH_HL Leaf 310.1.JPG'

# image_path_4 = 'dataset/val/Tomato___Septoria_leaf_spot/0b4886e5-a065-44c1-aa19-bd166922d3de___JR_Sept.L.S 8498.JPG'

# image_path_5 = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/val/Tomato___Target_Spot/0a3b6099-c254-4bc3-8360-53a9f558a0c4___Com.G_TgS_FL 8259.JPG'

# image_path_6 = '/home/amir/Desktop/Python/model.tomato-plant-diseases-detector/dataset/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/1af287e6-9d1e-4501-a113-d1a2b7c54b62___YLCV_GCREC 5321.JPG'




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
