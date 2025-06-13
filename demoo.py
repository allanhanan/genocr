import cv2
import pytesseract
from pytesseract import Output
from ultralytics import YOLO
import google.generativeai as genai
import PIL.Image
import numpy as np
import os
import time
import uuid
import logging
from datetime import datetime

# setting up logging, just in case something breaks and we need to debug
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# set the Gemini API key (make sure to put your own)
genai.configure(api_key="YOUR_API_KEY")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# config stuff, could move this to a file later maybe
CONFIG = {
    "ocr_engine": "tesseract",
    "yolo_model": "yolov8n.pt",
    "tmp_dir": "./tmp",
    "log_each_step": True
}

# make temp dir if it doesn’t exist
if not os.path.exists(CONFIG["tmp_dir"]):
    os.makedirs(CONFIG["tmp_dir"])


# this makes a random ID for files, not really needed but might be useful
def get_temp_id():
    return str(uuid.uuid4())


# gets image size and last modified time, might help later if we need metadata
def log_image_info(image_path):
    if not os.path.exists(image_path):
        return
    size = os.path.getsize(image_path)
    mtime = datetime.fromtimestamp(os.path.getmtime(image_path))
    logging.info(f"Image size: {size} bytes | Last modified: {mtime}")


# not actually doing anything complex, just pretending for now
def fake_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    return blurred  # not used but could help later


# run OCR and return just the words
def extract_text(image):
    if CONFIG["log_each_step"]:
        logging.info("running OCR...")

    # just using pytesseract directly
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words = []

    for word in data["text"]:
        clean = word.strip()
        if clean:
            words.append(clean)

    if CONFIG["log_each_step"]:
        logging.info(f"OCR done, found {len(words)} words")

    return words


# use YOLO to detect stuff in the image
def detect_objects(image):
    if CONFIG["log_each_step"]:
        logging.info("running object detection...")

    model = YOLO(CONFIG["yolo_model"])
    results = model(image)
    boxes = results[0].boxes

    # collecting just labels for now, no coords
    labels = list({model.names[int(box.cls[0])] for box in boxes})

    if CONFIG["log_each_step"]:
        logging.info(f"detected {len(labels)} objects")

    return labels


# send stuff to Gemini and get back only the final text
def send_to_gemini(image_path, words, labels):
    if CONFIG["log_each_step"]:
        logging.info("sending to Gemini...")

    img = PIL.Image.open(image_path)

    prompt = (
        "Extract and reconstruct all text from this image.\n"
        f"OCR extracted words: {' '.join(words)}\n"
        f"Objects in image: {', '.join(labels)}\n"
        "Do not include any explanation — return only the final reconstructed text."
    )

    # send the prompt and the image
    response = gemini_model.generate_content([prompt, img])

    return response.text.strip()


# main thing that runs everything
def process_image(image_path):
    logging.info(f"starting on {image_path}")

    if not os.path.exists(image_path):
        logging.error("image file not found")
        raise FileNotFoundError("image not found")

    log_image_info(image_path)

    image = cv2.imread(image_path)
    if image is None:
        logging.error("cv2 couldn't load the image")
        raise ValueError("couldn't read image")

    # might use this later to save intermediate files
    temp_id = get_temp_id()

    # OCR part
    text = extract_text(image)

    # object detection part
    objects = detect_objects(image)

    # send to Gemini and get result
    result = send_to_gemini(image_path, text, objects)

    return result


# not used, maybe for batch processing later
def process_multiple_images(image_paths):
    results = {}
    for path in image_paths:
        try:
            output = process_image(path)
            results[path] = output
        except Exception as e:
            logging.warning(f"failed to process {path}: {e}")
            results[path] = None
    return results


# runs if script is executed directly
if _name_ == "_main_":
    path = "tes.jpg"  # change this to your image
    final = process_image(path)
    print(final)