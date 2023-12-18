import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Streamlit app
def main():
    menu = ["Home", "Upload Image", "About", "Contact Us"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.title("Seeds Biospeckle Image  Classification App - Home")
        st.write("Welcome to the Seeds Biospeckle Image  Classification App!")

    elif choice == "Upload Image":
        st.title("Seeds Biospeckle Image  Classification App - Upload Image")

        # Image upload and prediction
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            image = Image.open(uploaded_file).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            predicted_class = class_names[index]
            confidence_score = prediction[0][index]

            # Display prediction and confidence score
            st.write(f"Predicted Class: {predicted_class[2:]}")
            st.write(f"Confidence Score: {confidence_score}")

    elif choice == "About":
        st.title("Seeds Biospeckle Image  Classification App - About")
        st.write("This app uses a deep learning model to classify seeds biospeckle image based on uploaded images.")

    elif choice == "Contact Us":
        st.title("Seeds Biospeckle Image  Classification App - Contact Us")
        st.write("For any inquiries or support, please contact us at example@email.com.")

if __name__ == "__main__":
    main()
