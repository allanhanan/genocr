import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import os
import pytesseract

#CNN-BiLSTM Model
def create_ocr_model(input_shape=(128, 32, 1), vocab_size=80):
    inputs = layers.Input(shape=input_shape)
    
    #CNN feature extractor
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)
    
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)
    
    #reshape for LSTM layers
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    #bidirectional LSTM layers for sequence modeling
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    
    #output layer
    x = layers.Dense(vocab_size + 1, activation="softmax")(x)
    
    model = Model(inputs, x)
    return model

#function to extract text labels using Tesseract OCR
def extract_text_labels(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(img, config="--psm 7")  # PSM 7 for single-line text
    return text.strip()

#data generator for images with extracted labels
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_paths, batch_size, img_width, img_height, vocab, max_text_length):
        self.img_paths = img_paths
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.vocab = vocab
        self.max_text_length = max_text_length
        self.indices = np.arange(len(img_paths))
        
        #extract labels from images
        self.labels = [extract_text_labels(path) for path in img_paths]

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images, texts = [], []
        
        for i in batch_indices:
            img = cv2.imread(self.img_paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img / 255.0  #normalize
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            
            text = self.labels[i]
            text = [self.vocab.get(char, 0) for char in text]  #handles missing characters
            texts.append(text)
        
        images = np.array(images)
        texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=self.max_text_length, padding='post')
        input_lengths = np.ones(len(images)) * (self.img_width // 4)
        label_lengths = np.array([len(t) for t in texts])
        
        return [images, texts, input_lengths, label_lengths], np.zeros(len(images))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

#CTC Loss Function for spaces as well
def ctc_loss_lambda_func(y_true, y_pred):
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

#define character set and load data paths
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\"' "
vocab = {char: i+1 for i, char in enumerate(char_list)}  # Mapping each character to an integer
vocab_size = len(vocab)

#image dataset paths from a folder
img_dir = "/path/to/your/images"  # Replace with the actual directory path
img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

#parameters
input_shape = (128, 32, 1)
max_text_length = 100  # Adjust based on your data
batch_size = 16
model_weights_path = "ocr_model_weights.h5"  # Path to save/load model weights

#model instantiation
model = create_ocr_model(input_shape=input_shape, vocab_size=vocab_size)
model.compile(optimizer="adam", loss=ctc_loss_lambda_func)

#load existing weights if available
if os.path.exists(model_weights_path):
    print("Loading existing weights from:", model_weights_path)
    model.load_weights(model_weights_path)
else:
    print("No existing weights found, starting from scratch.")

#prepare data generator
train_data = DataGenerator(img_paths, batch_size, input_shape[1], input_shape[0], vocab, max_text_length)

#callback to save weights after each epoch
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    model_weights_path,
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

#train fr fr
model.fit(train_data, epochs=50, callbacks=[checkpoint_callback])
