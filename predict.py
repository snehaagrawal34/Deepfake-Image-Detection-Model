import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load BEST model
model = load_model('best_model.keras')

def predict_image_from_array(image):

    # Resize
    image = cv2.resize(image, (96, 96))

    # Normalize
    image = image / 255.0

    # Reshape
    image = np.reshape(image, (1, 96, 96, 3))

    # Predict
    pred = model.predict(image)[0][0]

    # SAME LOGIC AS test.py
    if pred > 0.4:
        return "Fake", float(pred)
    else:
        return "Real", float(pred)