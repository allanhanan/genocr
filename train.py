import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

#load the ground truth text mappings
def load_GT(file_path):
    image_text_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_name, text = parts
                image_text_map[image_name] = text.strip('"')
    return image_text_map

#data generator to handle image strips of 512x512
class StripDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_text_map, image_dir, batch_size=32, img_size=(512, 512), strip_width=512, max_text_len=128, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!'\" "):
        self.image_text_map = image_text_map
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.strip_width = strip_width
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
            strips = self.process_image_strips(image)
            batch_images.extend(strips)
            
            encoded_text = self.encode_text(text)
            batch_texts.append(encoded_text)
        
        batch_images = np.array(batch_images).reshape(-1, self.img_size[0], self.img_size[1], 1)
        batch_texts = np.array(batch_texts)
        
        return batch_images, batch_texts

    def process_image_strips(self, image):
        strips = []
        height, width = image.shape
        num_strips = (width + self.strip_width - 1) // self.strip_width
        for i in range(num_strips):
            start_x = i * self.strip_width
            end_x = min(start_x + self.strip_width, width)
            strip = image[:, start_x:end_x]
            if strip.shape[1] < self.strip_width:
                pad_width = self.strip_width - strip.shape[1]
                strip = np.pad(strip, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
            strip = cv2.resize(strip, self.img_size)
            strip = strip.astype("float32") / 255.0
            strips.append(strip)
        return strips

#custom attention layer for sequence decoding
class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, encoder_output, decoder_hidden):
        #expands decoder hidden state to match encoder output dimensions
        decoder_hidden_time_axis = tf.expand_dims(decoder_hidden, 1)
        #computes attention scores
        score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(decoder_hidden_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        #applies attention weights to encoder output
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


def model_arch(input_shape, vocab_size):
    input_img = layers.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    #flatten and add an RNN layer with attention
    conv_output = layers.Reshape(target_shape=(-1, x.shape[-1]))(x)
    encoder_rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(conv_output)
    
    #decoder with attention
    decoder_rnn = layers.LSTM(128, return_sequences=True, return_state=True)
    attention_layer = AttentionLayer(128)
    decoder_output, _, _ = decoder_rnn(encoder_rnn)
    context_vector, _ = attention_layer(encoder_rnn, decoder_output)
    concat_output = layers.Concatenate(axis=-1)([context_vector, decoder_output])

    output = layers.TimeDistributed(layers.Dense(vocab_size + 1, activation="softmax"))(concat_output)  # +1 for CTC blank label

    model = models.Model(inputs=input_img, outputs=output)
    return model

#custom CTC loss function
def ctc_loss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    sequence_length = tf.shape(y_pred)[1]
    
    input_length = tf.ones(shape=(batch_size, 1)) * tf.cast(sequence_length, dtype="float32")
    label_length = tf.ones(shape=(tf.shape(y_true)[0], 1)) * tf.cast(tf.shape(y_true)[1], dtype="float32")
    
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

#training function with callbacks
def train_model(data_generator, model, checkpoint_path="ocr_checkpoint.keras"):
    model.compile(optimizer="adam", loss=ctc_loss)

    checkpoint = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="loss", mode="min")
    early_stopping = callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded previous model")

    model.fit(data_generator, epochs=200, callbacks=[checkpoint, early_stopping])

#parameters
IMG_SIZE = (512, 512)  
BATCH_SIZE = 32 
MAX_TEXT_LEN = 128 
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!'\" "

ground_truth_file = "./archive/IAM/gt_test.txt"
image_dir = "./archive/IAM/image"
image_text_map = load_GT(ground_truth_file)
data_generator = StripDataGenerator(image_text_map, image_dir, BATCH_SIZE, IMG_SIZE, MAX_TEXT_LEN, CHARSET)

#model setup
vocab_size = len(CHARSET)
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
model = model_arch(input_shape, vocab_size)


train_model(data_generator, model)
