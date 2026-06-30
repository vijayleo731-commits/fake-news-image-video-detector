import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    LSTM,
    TimeDistributed
)

# ==========================================
# NEWS MODEL
# ==========================================

texts = [
    "Government launches satellite",
    "Aliens landed on Earth",
    "Education policy announced",
    "Magic medicine cures all diseases"
]

labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

news_model = LogisticRegression()

news_model.fit(X, labels)

joblib.dump(news_model, "news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ News model created")

# ==========================================
# IMAGE MODEL
# ==========================================

image_model = Sequential([
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 128, 3)
    ),

    MaxPooling2D((2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(1, activation='sigmoid')
])

image_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

image_model.save("image_model.h5")

print("✅ Image model created")

# ==========================================
# VOICE MODEL
# ==========================================

voice_model = Sequential([
    Dense(
        64,
        activation='relu',
        input_shape=(40,)
    ),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

voice_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

voice_model.save("voice_model.h5")

print("✅ Voice model created")

# ==========================================
# VIDEO MODEL
# ==========================================

video_model = Sequential([

    TimeDistributed(
        Flatten(),
        input_shape=(10, 128, 128, 3)
    ),

    LSTM(64),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

video_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

video_model.save("video_model.h5")

print("✅ Video model created")

print("\nAll models saved successfully.")