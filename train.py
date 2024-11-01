import tensorflow as tf
import cv2
import os
import pytesseract
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import datetime
import concurrent.futures

IMAGE_DIR = './dataset'
WEIGHTS_PATH = 'ocr_model_weights.keras'
LOG_DIR = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
NUM_CLASSES = 80
BATCH_SIZE = 16
EPOCHS = 10

#character mapping
characters = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
max_label_length = 32

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

def extract_text_from_image(image_path):
    #extract printed groundtruth text from upper section of image using OCR
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    printed_text_image = image[:int(h * 0.5), :]  #crop printed section
    printed_text = pytesseract.image_to_string(printed_text_image)
    return printed_text.strip()

def encode_label(text):
    return [char_to_num.get(char, 0) for char in text]

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
    y_true = tf.cast(y_true, tf.int32)
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
    logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    return tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=-1
    )

class ModelImprovementLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        if self.model.stop_training:
            print(f"Epoch {epoch + 1}: Training stopped early")
        elif current_val_loss is not None and current_val_loss < self.best_val_loss:
            print(f"Epoch {epoch + 1}: New best val_loss {current_val_loss:.4f} achieved!")
            self.best_val_loss = current_val_loss
        else:
            print(f"Epoch {epoch + 1}: val_loss did not improve")

#training pipeline
def train_model():
    input_shape = (32, 128, 1)
    model = build_model(input_shape, NUM_CLASSES)
    
    #Load previous weights if exists
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
    
    model.compile(optimizer='adam', loss=ctc_loss)
    
    #define callbacks
    checkpoint_callback = ModelCheckpoint(WEIGHTS_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=True)
    improvement_logger = ModelImprovementLogger()

    #data loading
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR)]
    train_data = []
    train_labels = []

    #process images with multithreading for OCR
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_text_from_image, path): path for path in image_paths}
        for i, (future, path) in enumerate(futures.items(), start=1):
            printed_text = future.result()
            print(f"Processing text for image {i}/{len(image_paths)}: {path}")
            processed_image = preprocess_image(path)
            train_data.append(processed_image)
            train_labels.append(encode_label(printed_text))

    train_data = np.array(train_data).reshape(-1, 32, 128, 1)
    train_labels = pad_sequences(train_labels, maxlen=max_label_length, padding='post')
    train_labels = np.array(train_labels)

    model.fit(
        train_data, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[checkpoint_callback, tensorboard_callback, improvement_logger]
    )

#train fr fr
train_model()
