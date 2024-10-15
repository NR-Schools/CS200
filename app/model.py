import app.optimizer_def
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

CONVERT_CLASS_PRED_TO_NAME = ["Common Rust", "Gray Leaf Spot", "Leaf Blight"]

def _fetch_model(opt_name: str):
    return load_model(f'models/{opt_name}-model-001.h5', safe_mode=False)

def classify_image(model_name, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    model = _fetch_model(model_name)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return CONVERT_CLASS_PRED_TO_NAME[predicted_class]
