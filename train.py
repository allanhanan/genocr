import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def load_GT(file_path):
    image_text_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_name, text = parts
                image_text_map[image_name] = text.strip('"')
    return image_text_map


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_text_map, image_dir, batch_size=32, img_size=(1024, 64), max_text_len=128, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!'\" "):
        self.image_text_map = image_text_map
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.charset = charset
        self.char_to_num = {char: idx for idx, char in enumerate(charset)}
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}
        self.indices = list(self.image_text_map.keys())
    
    def __len__(self):
        return len(self.indices) // self.batch_size

    def encode_text(self, text):
        encoded = [self.char_to_num.get(char, 0) for char in text]
        return encoded + [0] * (self.max_text_len - len(encoded))

    def __getitem__(self, idx):
        batch_images = []
        batch_texts = []
        
        for i in range(self.batch_size):
            image_name = self.indices[idx * self.batch_size + i]
            image_path = os.path.join(self.image_dir, image_name)
            text = self.image_text_map[image_name]


            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.img_size)
            image = image.astype("float32") / 255.0
            batch_images.append(image)
            
            #encode ext
            encoded_text = self.encode_text(text)
            batch_texts.append(encoded_text)
        
        batch_images = np.array(batch_images).reshape(-1, self.img_size[0], self.img_size[1], 1)
        batch_texts = np.array(batch_texts)
        
        return batch_images, batch_texts


def model_arch(input_shape, vocab_size):
    input_img = layers.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    conv_output = layers.Reshape(target_shape=(-1, x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(conv_output)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    output = layers.Dense(vocab_size + 1, activation="softmax")(x)  # +1 for CTC blank label

    model = models.Model(inputs=input_img, outputs=output)
    return model


def ctc_loss(y_true, y_pred):
    #tf.shape() to get dynamic shapes for batch and sequence length
    batch_size = tf.shape(y_pred)[0]
    sequence_length = tf.shape(y_pred)[1]
    
    input_length = tf.ones(shape=(batch_size, 1)) * tf.cast(sequence_length, dtype="float32")
    label_length = tf.ones(shape=(tf.shape(y_true)[0], 1)) * tf.cast(tf.shape(y_true)[1], dtype="float32")
    
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)



def train_model(data_generator, model, checkpoint_path="ocr_checkpoint.keras"):
    
    
    model.compile(optimizer="adam", loss=ctc_loss)
    

    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="loss", mode="min")
    early_stopping = callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    
    #load weights if available
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded previous model")
    

    model.fit(data_generator, epochs=100, callbacks=[checkpoint, early_stopping])



IMG_SIZE = (1024, 64)  
BATCH_SIZE = 32
MAX_TEXT_LEN = 128 
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!'\" "


ground_truth_file = "./archive/IAM/gt_test.txt"
image_dir = "./archive/IAM/image"
image_text_map = load_GT(ground_truth_file)
data_generator = DataGenerator(image_text_map, image_dir, BATCH_SIZE, IMG_SIZE, MAX_TEXT_LEN, CHARSET)


vocab_size = len(CHARSET)
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
model = model_arch(input_shape, vocab_size)


train_model(data_generator, model)
