from fastapi import FastAPI, UploadFile, File
from main import predict_class
import uvicorn
import nest_asyncio
from fastapi.responses import JSONResponse
import datetime
import pytz

our_timezone = 'Asia/Dhaka'
current_time = datetime.datetime.now(pytz.timezone(our_timezone))
iso_string_timezone = current_time.isoformat()

nest_asyncio.apply()
app = FastAPI()

main_api_endpoint = '/predict-tomato-plant-diseases'
hostname='localhost'
port=5172

print(f'\nRunning Localhost Server On (http://{hostname}:{port}{main_api_endpoint})\n')

allowed_image_extensions = ["jpg", "png", "svg", "jpeg", "webp", "JPG", "PNG", "JPEG", "SVG", "WEBP"]
def is_valid_image(image_ext):
  return image_ext in allowed_image_extensions

def wrong_api_call():
  return JSONResponse(
    content = { 
        'message': 'Server is Running...', 
        'detail': f'Please send a `POST` request into: `http://{hostname}:{port}{main_api_endpoint}` with a `Tomato Plant Leaf` image.' 
      }
    ) 

@app.get('/')
async def root():
  return wrong_api_call();
  
# show a error message when the user send get request
@app.get(main_api_endpoint)
async def main_api_endpoint_get_request():
  return wrong_api_call()
  
# Define a post route to make api of image processing.
@app.post(main_api_endpoint)
async def get_file_from_main_api_endpoint(image: UploadFile = File(...)):
  file_ext = image.filename.split(".")[-1]
  
  # Check the validate file extension 
  if not is_valid_image(file_ext):
    return JSONResponse(
      content = {
        'status_code': 400,
        'detail': f"Invalid image format. Allowed formats: {', '.join(allowed_image_extensions)}",
      }
    )

  # save the validate image in our temporary folder
  image_name = f'./temp/Tomato_Leaf___{iso_string_timezone}.{file_ext}'
  with open(image_name, 'wb') as get_image:
    get_image.write(image.file.read())
  
  # call the model prediction function
  disease_name, predicted_class, confidence = predict_class(image_name)
  
  return JSONResponse( 
    content = {
      'confidence': f'{confidence:.2f}',
      'predicted_class': predicted_class,
      'leaf_condition': disease_name.capitalize(),
      'confidence_percentage': f'{confidence * 100:.2f}%',
    }
  )

uvicorn.run(app, host=hostname, port=port)