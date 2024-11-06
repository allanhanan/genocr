import tensorflow as tf
import numpy as np
import cv2

# Define the expected image dimensions
IMG_WIDTH = 1024
IMG_HEIGHT = 64

def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, resizes to match model expectations,
    normalizes pixel values, and reshapes to (1, IMG_HEIGHT, IMG_WIDTH, 1).
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Resize to match model's expected width and height
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    
    # Reshape to (1, 64, 1024, 1)
    img = img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 1))
    
    return img

def predict_text_from_image(image_path):
    """
    Loads the trained model, preprocesses the input image, and predicts the text, including blanks.
    """
    # Load the trained model
    model = tf.keras.models.load_model('ocr_checkpoint.keras', compile=False)
    
    # Preprocess the input image
    img = preprocess_image(image_path)
    
    # Perform prediction
    preds = model.predict(img)
    
    # Decode the prediction using CTC decoding
    decoded_text = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1])[0][0]
    decoded_text = tf.keras.backend.get_value(decoded_text)
    
    # Convert array to text string, treating -1 as a space
    predicted_text = ''.join([chr(int(c)) if c != -1 else ' ' for c in decoded_text[0]])
    
    return predicted_text


# Test the function with an image path
if __name__ == "__main__":
    image_path = "./tes2.jpg"  # Replace with the actual image path
    predicted_text = predict_text_from_image(image_path)
    print("Predicted Text:", predicted_text)
