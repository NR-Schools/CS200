from keras.models import load_model
from keras.preprocessing import image
import numpy as np

CONVERT_CLASS_PRED_TO_NAME = ["Common Rust", "Gray Leaf Spot", "Leaf Blight"]

model = load_model('vgnet_model/model-001.h5', safe_mode=False)

def classify_image(img):
    img = img.resize((224, 224))  # Adjust the target size based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the shape of model input
    img_array /= 255.0  # Normalize the image if the model requires normalization

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    

    return {"class": CONVERT_CLASS_PRED_TO_NAME[predicted_class], "confidence": confidence}
