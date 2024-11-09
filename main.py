import tensorflow as tf
from google.generativeai import palm
import cv2
import numpy as np

#google PaLM API (substitute with your API key)
palm.configure(api_key="YOUR_API_KEY")

#load the OCR model
model = tf.keras.models.load_model("ocr_checkpoint1.keras")

def preprocess_image(image_path):
    """
    Preprocesses the input image to match the model's input requirements.
    Converts to grayscale, resizes, normalizes, and adds necessary dimensions.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = image / 255.0  #normalize
    image = np.expand_dims(image, axis=-1)  #add channel dimension
    return np.expand_dims(image, axis=0)  #add batch dimension

def detect_text_from_image(image_path):
    """
    Detects text from the input image using the OCR model.
    """
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    
    #decode predictions into text
    detected_text = decode_predictions(predictions)
    print(f"Detected incomplete text: {detected_text}")
    return detected_text

def decode_predictions(predictions):
    """
    Decodes model predictions by converting ASCII-encoded integers back to characters.
    """
    indices = np.argmax(predictions, axis=-1)  #get the highest probability index for each position
    decoded_text = []
    for index in indices[0]:  #decode for the first item in batch
        if index == 0:
            continue
        decoded_text.append(chr(index))  #convert ASCII code to character
    return ''.join(decoded_text)

def generate_completion(text):
    """
    Uses Google PaLM API to complete the detected text.
    """
    response = palm.generate_text(
        prompt=f"Complete the sentence: {text}",
        temperature=0.7,
        max_output_tokens=50
    )
    if response and response.candidates:
        return response.candidates[0]['output']
    return "No completion generated."

def main(image_path):
    """
    Main function to detect incomplete text and generate a completion.
    """
    #detect incomplete text
    detected_text = detect_text_from_image(image_path)
    
    #generate and print the completion
    completed_text = generate_completion(detected_text)
    print("Completed sentence:", completed_text)


main("path_to_your_image.jpg")
