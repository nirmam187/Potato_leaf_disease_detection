from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras  # ✅ Use Keras directly
from keras.saving import load_model

app = FastAPI()

# CORS setup
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Model Path
MODEL_PATH = "/Users/nirmamparikh/Desktop/machinelearning/potato_disease_project/saved_models/potatoes.h5"

# ✅ Load Model Correctly
try:
    MODEL = load_model(MODEL_PATH,compile=False, safe_mode=False)  # ✅ Use keras.models.load_model()
    print("✅ Model loaded successfully!")
    print(MODEL.summary())
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL = None

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

def read_file_as_image(data) -> np.ndarray:
    """Convert uploaded image bytes to a numpy array."""
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # Resize for model
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image uploads and return predictions."""
    if MODEL is None:
        return {"error": "Model not loaded properly. Check logs for details."}

    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
# from keras.saving import load_model

# model_path = "/Users/nirmamparikh/Desktop/machinelearning/potato_disease_project/saved_models/potatoes.h5"
# loaded_model = load_model(model_path,compile=False, safe_mode=False)
