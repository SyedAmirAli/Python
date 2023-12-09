from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def root():
    return { 'message': 'Hello, How are you?' }

print('Running Port: http://localhost:8000/')
