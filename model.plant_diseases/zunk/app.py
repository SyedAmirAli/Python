# # app.py
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# from predictor import predict_class

# app = Flask(__name__)

# @app.route('/predict-plant-diseases', methods=['POST'])
# def predict_plant_diseases():
#     if 'image' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     image_file = request.files['image']

#     if image_file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         predicted_class = predict_class(image_file)
#         result = {'predicted_class': predicted_class}
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


import requests

url = 'http://localhost:5174/predict-plant-diseases'

# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = './test-images/1cee26b4-0ba1-4371-8c7e-c641e6ca311d___PSU_CG 2283_newPixel25.JPG'

# Create a dictionary with the file to upload
files = {'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')}

# Send the POST request
response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print(f'Predicted Class: {result["predicted_class"]}')
else:
    print(f'Error: {response.text}')