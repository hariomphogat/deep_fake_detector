import os
import shutil
import tempfile
import uvicorn
import warnings
from fastapi import FastAPI, UploadFile, File, HTTPException
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

# --- Suppress TensorFlow & MTCNN Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# --- Imports for config and prediction functions ---
try:
    from . import config
    from .predict import get_image_prediction
    from .predict_video_model import get_video_prediction
    from .video_model import build_video_model
except ImportError:
    # This fallback lets us run the file directly if needed
    import config
    from predict import get_image_prediction
    from predict_video_model import get_video_prediction

# --- 1. Create the FastAPI app ---
app = FastAPI(
    title="Deepfake Detector API",
    description="An API to detect deepfake images and videos using advanced ML models.",
    version="1.0.0"
)

# --- 2. Load Models at Startup (Best Practice) ---
# This dictionary will hold our models, loaded ONCE.
# This is far more efficient than loading them for every request.
models = {}

@app.on_event("startup")
async def load_models():
    """
    Load all ML models into memory when the API server starts.
    """
    print("--- Loading models into memory... ---")
    
    # Load Image Model (finetuned_model.h5)
    image_model_path = os.path.join(config.MODEL_DIR, "finetuned_model.h5")
    if os.path.exists(image_model_path):
        models["image_model"] = tf.keras.models.load_model(image_model_path, compile=False)
        print("Image model (finetuned_model.h5) loaded successfully.")
    else:
        print(f"WARNING: Image model not found at {image_model_path}")

    # # Load Video Model (video_model_v2.keras)
    # video_model_path = os.path.join(config.MODEL_DIR, "video_model_v2.keras")
    # if os.path.exists(video_model_path):
    #     models["video_model"] = tf.keras.models.load_model(video_model_path)
    #     print("Video model (video_model_v2.keras) loaded successfully.")
    # else:
    #     print(f"WARNING: Video model not found at {video_model_path}")
    # --- START: ULTIMATE FIX FOR VIDEO MODEL ---
    # Load Video Model (video_model_v2.keras)
    video_model_path = os.path.join(config.MODEL_DIR, "video_model_v2.keras")
    if os.path.exists(video_model_path):
        try:
            # 1. Build the "empty" model architecture from your code
            print("Building video model architecture from video_model.py...")
            video_model = build_video_model()
            
            # 2. Load *only the weights* from your saved file
            print(f"Loading weights from {video_model_path}...")
            video_model.load_weights(video_model_path)
            
            # 3. Assign the working model
            models["video_model"] = video_model
            print("Video model (video_model_v2.keras) loaded successfully from weights.")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load video model from weights: {e}")
            print("This might be due to a mismatch between video_model.py and your saved file.")
    else:
        print(f"WARNING: Video model not found at {video_model_path}")
    # --- END: ULTIMATE FIX FOR VIDEO MODEL ---
    # Load MTCNN Detector
    models["mtcnn_detector"] = MTCNN()
    print("MTCNN detector initialized.")
    print("--- All models loaded. API is ready. ---")
    
# --- 3. Define API Endpoints ---

@app.get("/")
def read_root():
    """A simple 'health check' endpoint to see if the server is running."""
    return {"status": "Deepfake Detector API is online and running."}

@app.post("/predict_image")
async def predict_image_api(file: UploadFile = File(...)):
    """
    Endpoint for predicting a single deepfake image.
    Accepts an uploaded image file.
    """
    if "image_model" not in models:
        raise HTTPException(status_code=500, detail="Image model is not loaded.")
    
    # We must save the uploaded file to a temporary path
    # because our prediction function expects a file path.
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        print(f"Processing image: {temp_file_path}")
        
        # Call our prediction function and pass it the pre-loaded model
        result = get_image_prediction(
            image_path=temp_file_path,
            model=models["image_model"]
        )
        return result
        
    except Exception as e:
        # If anything goes wrong, return an error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # CRITICAL: Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/predict_video")
async def predict_video_api(file: UploadFile = File(...)):
    """
    Endpoint for predicting a single deepfake video.
    Accepts an uploaded video file.
    """
    if "video_model" not in models or "mtcnn_detector" not in models:
        raise HTTPException(status_code=500, detail="Video models are not loaded.")

    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            
        print(f"Processing video: {temp_file_path}")

        # Call the video prediction function
        result = get_video_prediction(
            video_path=temp_file_path,
            video_model=models["video_model"],
            detector=models["mtcnn_detector"]
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # CRITICAL: Always clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- 4. How to run this file for development ---
if __name__ == "__main__":
    print("--- Starting FastAPI server directly (for development) ---")
    print("--- Go to http://127.0.0.1:8000 for the API ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir="src")