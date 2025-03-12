import tensorflow as tf
import cv2
import numpy as np

# Load trained waste classification model
model = tf.keras.models.load_model("waste_classifier.h5")

def classify_waste(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    categories = ["Recyclable", "Non-Recyclable"]
    return categories[np.argmax(prediction)]

# Test Case
print(classify_waste("plastic_bottle.jpg"))
