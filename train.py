import os
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import tensorflow as tf

# =======================
# CONFIGURATION
# =======================

PDF_YES_PATH = "test.pdf"
PDF_NO_PATH = "drawing.pdf"
TFJS_MODEL_DIR = "tfjs_model_dir"

# Fixed input image size
IMAGE_SIZE = (224, 224)

NORMALIZE_MEAN = np.array([0.87857046, 0.88497418, 0.88770625])
NORMALIZE_STD = np.array([0.24927001, 0.23154211, 0.22728177])

# =======================
# FUNCTIONS
# =======================

def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    img = images[0]
    img = img.resize(IMAGE_SIZE)
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img

def preprocess(img):
    img = img / 255.0
    img = (img - NORMALIZE_MEAN) / NORMALIZE_STD
    return img

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 32 filters
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filters
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), # 128 filters
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),           # 64 Dense units
        tf.keras.layers.Dense(1, activation='sigmoid')          # Binary output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =======================
# MAIN PIPELINE
# =======================

def main():
    print("Loading and preprocessing images...")
    img_yes = preprocess(pdf_to_image(PDF_YES_PATH))
    img_no = preprocess(pdf_to_image(PDF_NO_PATH))

    X = np.stack([img_yes, img_no], axis=0)
    y = np.array([[1], [0]])

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X, y, epochs=20, verbose=2)

    print("Training complete.")

    print("Saving model as tiny_model.h5...")
    model.save("tiny_model.h5")

    print("\nModel saved!")
    print(f"Now run:")
    print(f"tensorflowjs_converter --input_format=keras --quantize_float16 tiny_model.h5 {TFJS_MODEL_DIR}")

if __name__ == "__main__":
    main()
