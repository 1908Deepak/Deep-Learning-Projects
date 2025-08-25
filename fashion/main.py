import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


class FashionClassifier:
    def __init__(self, model_path: str):
        """Initialize and load the trained model"""
        self.model = load_model('fashion/fashion_mnist.h5')
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize image for model prediction"""
        image = image.convert("L")              # Convert to grayscale
        image = image.resize((28, 28))          # Resize to 28x28
        img_array = np.array(image) / 255.0     # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)  # Add batch & channel dims
        return img_array

    def predict(self, image: Image.Image):
        """Predict class and probability for input image"""
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        return self.class_names[predicted_class], confidence, predictions[0]
