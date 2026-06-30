import streamlit as st
import joblib
import numpy as np
import librosa
from PIL import Image
import tensorflow as tf
import cv2
import tempfile
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# =====================================================
# CREATE NEWS MODEL
# =====================================================

texts = [
    "Government launches new satellite",
    "Aliens landed on Earth yesterday",
    "Education policy announced",
    "Magic medicine cures all diseases"
]

labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

news_model = LogisticRegression()

news_model.fit(X, labels)

# =====================================================
# CREATE IMAGE MODEL
# =====================================================

image_model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

image_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# STREAMLIT PAGE
# =====================================================

st.set_page_config(
    page_title="AI Fake/Real Detector",
    layout="centered"
)

st.title("AI Fake News, Image & Video Detector")

st.write("Detect Fake News, Fake Images and Fake Videos using AI")

# =====================================================
# MENU
# =====================================================

menu = st.sidebar.selectbox(
    "Choose Detection Module",
    [
        "News Detector(text,image,video)",
        "Image Detector",
        "Video Detector",
        "Voice Detector"
    ]
)

# =====================================================
# NEWS DETECTOR
# =====================================================

if menu == "News Detector(text,image,video)":

    st.header("Fake or Real News Detection")

    input_type = st.radio(
        "Choose Input Type",
        ["Text", "Image", "Video"]
    )

    # ---------------------------------
    # TEXT INPUT
    # ---------------------------------
    if input_type == "Text":

        news = st.text_area(
            "Enter News Article"
        )

        if st.button("Analyze News"):

            if news.strip() == "":
                st.warning("Please enter news text")

            else:

                vector_input = vectorizer.transform(
                    [news]
                )

                prediction = news_model.predict(
                    vector_input
                )

                if prediction[0] == 1:
                    st.success(
                        "✅ This news appears REAL"
                    )

                else:
                    st.error(
                        "❌ This news appears FAKE"
                    )

    # ---------------------------------
    # IMAGE INPUT
    # ---------------------------------
    elif input_type == "Image":

        uploaded_image = st.file_uploader(
            "Upload News Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:

            image = Image.open(
                uploaded_image
            )

            st.image(
                image,
                caption="Uploaded News Image",
                use_container_width=True
            )

            st.info(
                "OCR-based news analysis can be added here."
            )

    # ---------------------------------
    # VIDEO INPUT
    # ---------------------------------
    elif input_type == "Video":

        uploaded_video = st.file_uploader(
            "Upload News Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video is not None:

            st.video(uploaded_video)

            st.info(
                "Speech-to-text or OCR analysis can be added here."
            )
# =====================================================
# IMAGE DETECTOR
# =====================================================

elif menu == "Image Detector":

    st.header("Fake or Real Image Detection")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        try:
            # Open image
            image = Image.open(
                uploaded_image
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            # Resize image
            image = image.resize((128, 128))

            # Convert to NumPy
            image_array = np.array(image)

            # Normalize
            image_array = image_array.astype(
                np.float32
            ) / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Predict
            prediction = image_model.predict(
                image_array,
                verbose=0
            )

            score = float(prediction[0][0])

            st.write(
                "Prediction Score:",
                round(score, 4)
            )

            if score > 0.5:
                st.success(
                    "✅ Image appears REAL"
                )

            else:
                st.error(
                    "❌ Image appears FAKE or AI Generated"
                )

        except Exception as e:
            st.error(
                f"Error processing image: {str(e)}"
            )
# =====================================================
# VIDEO DETECTOR
# =====================================================

elif menu == "Video Detector":

    st.header("Fake or Real Video Detection")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_file.write(uploaded_video.read())
        temp_file.close()

        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        cap.release()

        st.write("Frame Count:", frame_count)
        st.write("Resolution:", f"{width} x {height}")

        if frame_count < 30:
            st.error("Suspicious or Fake Video")
        else:
            st.success("Likely Real Video")

        if os.path.exists(video_path):
            os.remove(video_path)
# =====================================================
# VOICE DETECTOR
# =====================================================

elif menu == "Voice Detector":

    st.header("Fake or Real Voice Detection")

    uploaded_audio = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3"]
    )

    if uploaded_audio is not None:

        st.audio(uploaded_audio)

        # Preserve original extension
        extension = os.path.splitext(
            uploaded_audio.name
        )[1]

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        )

        temp_audio.write(uploaded_audio.read())
        temp_audio.close()

        audio_path = temp_audio.name

        try:

            y, sr = librosa.load(
                audio_path,
                sr=16000,
                mono=True
            )

            duration = librosa.get_duration(
                y=y,
                sr=sr
            )

            st.write(
                "Duration:",
                round(duration, 2),
                "seconds"
            )

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=40
            )

            score = np.mean(np.abs(mfcc))

            st.write(
                "Detection Score:",
                round(score, 2)
            )

            if score > 50:
                st.success(
                    "✅ Voice appears REAL"
                )
            else:
                st.error(
                    "❌ Voice may be AI Generated"
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)