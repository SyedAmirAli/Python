import requests

url = "http://localhost:5171/predict-plant-diseases"

files = {'file': ('filename.jpg', open('./test-images/3.png', 'rb'), 'image/jpeg')}

response = requests.post(url, files=files)
print(response.text)
