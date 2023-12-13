from fastapi import FastAPI, UploadFile, File
from lettuce import lettuce_predict_class
from tomato import tomato_predict_class
from strawberry import strawberry_predict_class
import uvicorn
import nest_asyncio
from fastapi.responses import JSONResponse
import datetime
import pytz
import os
from fastapi.middleware.cors import CORSMiddleware

our_timezone = 'Asia/Dhaka'
current_time = datetime.datetime.now(pytz.timezone(our_timezone))
iso_string_timezone = current_time.isoformat()

nest_asyncio.apply()
app = FastAPI()

# block cors origin requests
# Allow all origins in this example
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

main_api_endpoint = '/predict-tomato-plant-diseases'
hostname='localhost'
port=5172


print(f'\nRunning Localhost Server On (http://{hostname}:{port}{main_api_endpoint})\n\nTomato:\t\thttp://{hostname}:{port}/tomato-plant-disease\nLettuce:\thttp://{hostname}:{port}/lettuce-plant-disease\nStrawberry:\thttp://{hostname}:{port}/strawberry-plant-disease\n')

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

# Define a post route to make an API for tomato plant disease detector
@app.post('/tomato-plant-disease')
async def tomato_plant_disease(image: UploadFile = File(...)):
  file_ext = image.filename.split(".")[-1]
  
  # Check the validate file extension 
  if not is_valid_image(file_ext):
    return JSONResponse(
      content = {
        'status_code': 400,
        'detail': f"Invalid image format. Allowed formats: {', '.join(allowed_image_extensions)}",
      }
    )

  with open('temp/image.webp', 'wb') as get_image:
    get_image.write(image.file.read())
  
  # call the model prediction function
  disease_name, predicted_class, confidence = tomato_predict_class('temp/image.webp')
  
  return JSONResponse( 
    content = {
      'confidence': f'{confidence:.2f}',
      'predicted_class': predicted_class,
      'leaf_condition': disease_name.capitalize(),
      'confidence_percentage': f'{confidence * 100:.2f}%',
    }
  )

# Define a post route to make an API for lettuce plant disease detector
@app.post('/lettuce-plant-disease')
async def lettuce_plant_disease(image: UploadFile = File(...)):
  file_ext = image.filename.split(".")[-1]
  
  # Check the validate file extension 
  if not is_valid_image(file_ext):
    return JSONResponse(
      content = {
        'status_code': 400,
        'detail': f"Invalid image format. Allowed formats: {', '.join(allowed_image_extensions)}",
      }
    )

  with open('temp/image.webp', 'wb') as get_image:
    get_image.write(image.file.read())
  
  # call the model prediction function
  disease_name, predicted_class, confidence = lettuce_predict_class('temp/image.webp')
  
  return JSONResponse( 
    content = {
      'confidence': f'{confidence:.2f}',
      'predicted_class': predicted_class,
      'leaf_condition': disease_name.capitalize(),
      'confidence_percentage': f'{confidence * 100:.2f}%',
    }
  )

# Define a post route to make an API for strawberry plant disease detector
@app.post('/strawberry-plant-disease')
async def strawberry_plant_disease(image: UploadFile = File(...)):
  file_ext = image.filename.split(".")[-1]
  
  # Check the validate file extension 
  if not is_valid_image(file_ext):
    return JSONResponse(
      content = {
        'status_code': 400,
        'detail': f"Invalid image format. Allowed formats: {', '.join(allowed_image_extensions)}",
      }
    )

  with open('temp/image.webp', 'wb') as get_image:
    get_image.write(image.file.read())
  
  # call the model prediction function
  disease_name, predicted_class, confidence = strawberry_predict_class('temp/image.webp')
  
  return JSONResponse( 
    content = {
      'confidence': f'{confidence:.2f}',
      'predicted_class': predicted_class,
      'leaf_condition': disease_name.capitalize(),
      'confidence_percentage': f'{confidence * 100:.2f}%',
    }
  )

uvicorn.run(app, host=hostname, port=port)