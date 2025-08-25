import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from main import FashionClassifier

# Initialize classifier with your new model
classifier = FashionClassifier("fashion/fashion_mnist.h5")

# App title
st.title("👕👟 Fashion MNIST Image Classification")
st.write("Upload a grayscale fashion item image (28x28 or larger) and the model will predict its category.")

# Sidebar for project info
st.sidebar.header("📌 Project Details")
st.sidebar.markdown("""
**Dataset**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)  
**Algorithm**: Convolutional Neural Network (CNN)  
**Frameworks**: TensorFlow, Keras, Streamlit  
**Author**: Deepak Singh
""")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        label, confidence, all_probs = classifier.predict(image)
        st.success(f"Predicted Class: **{label}** ({confidence*10:.2f}%)")

        # Show probability distribution as bar chart
        st.subheader("Prediction Confidence per Class")
        fig, ax = plt.subplots()
        ax.bar(classifier.class_names, all_probs)
        ax.set_xticklabels(classifier.class_names, rotation=45, ha="right")
        ax.set_ylabel("Probability")
        st.pyplot(fig)
