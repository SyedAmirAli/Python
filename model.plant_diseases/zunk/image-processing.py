from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

model_path = './keras-model/model.h5'
model = load_model(model_path)

base_path = '/home/amir/Downloads/zip/ai-model/valid/Strawberry___healthy/'
image_path = base_path + '0f79c593-bcf2-4a5b-baac-6433f3037a89___RS_HL 2022_new30degFlipLR.JPG'

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

print('\n\n')
print(model.predict(preprocess_image(image_path)))
print('\n\n')
 