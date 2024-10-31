import tensorflow as tf
import cv2
import os
import pytesseract
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from concurrent.futures import ThreadPoolExecutor
import datetime


IMAGE_DIR = './dataset'
WEIGHTS_PATH = 'ocr_model_weights.keras'
LOG_DIR = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#use GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used.")
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("No GPU found. Please ensure TensorFlow is installed with GPU support.")

def preprocess_image(image_path):
    #preprocess image by loading, cropping, resizing, and normalizing
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    handwritten_image = image[int(h * 0.5):, :]  #crop handwritten section
    handwritten_image = cv2.resize(handwritten_image, (128, 32))
    handwritten_image = handwritten_image / 255.0  #normalize
    return handwritten_image

def extract_text_from_image(image_path, index, total_images):
    #extract printed groundtruth text from upper section of image using OCR
    print(f"Processing text for image {index}/{total_images}: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    printed_text_image = image[:int(h * 0.5), :]  #crop printed section
    printed_text = pytesseract.image_to_string(printed_text_image)
    print("Extraction success for:", image_path)
    return printed_text.strip()

def process_images_multithreaded(image_paths, max_workers=None):
    total_images = len(image_paths)
    extracted_texts = []
    
    #set up the ThreadPoolExecutor with a specified or default max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_text_from_image, image_path, i + 1, total_images)
            for i, image_path in enumerate(image_paths)
        ]
        
        for future in futures:
            extracted_texts.append(future.result())
    
    return extracted_texts

#model definition
def build_model(input_shape, num_classes):
    #CNN-LSTM model
    inputs = Input(shape=input_shape, name='input_image')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    #reshape for RNN
    new_shape = (input_shape[0] // 4, input_shape[1] // 4 * 64)
    x = Reshape(target_shape=new_shape)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    #LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

#CTC loss function
def ctc_loss(y_true, y_pred):
    return tf.nn.ctc_loss(y_true, y_pred, logits_time_major=False, blank_index=-1)

#training pipeline
def train_model():
    input_shape = (32, 128, 1)
    num_classes = 80  #ASCII space or character count based on dataset
    model = build_model(input_shape, num_classes)
    
    #Load previous weights if exists
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
    
    model.compile(optimizer='adam', loss=ctc_loss)
    
    #define callbacks
    checkpoint_callback = ModelCheckpoint(WEIGHTS_PATH, save_best_only=True, monitor='val_loss', mode='min')
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True)

    #data loading
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR)]
    
    #process images with multithreading for OCR
    train_labels = process_images_multithreaded(image_paths, max_workers=500)
    train_data = [preprocess_image(image_path) for image_path in image_paths]
    
    train_data = np.array(train_data).reshape(-1, 32, 128, 1)
    train_labels = np.array(train_labels)

    model.fit(train_data, train_labels, batch_size=16, epochs=10, validation_split=0.2, callbacks=[checkpoint_callback, tensorboard_callback])

#train fr fr
train_model()
