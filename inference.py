from tqdm import tqdm
import sys
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('TensorFlow', tf.__version__)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.enable_eager_execution()

def get_image(image_path, h=180, w=480):
    '''returns image, pixels scaled [0-1]'''
    img = np.array(tf.keras.preprocessing.image.load_img(image_path, target_size=(h, w), color_mode='grayscale'), dtype=np.float32)[:,:,None]
    img /= 255.
    return img


def l2_distance(tensors):
    x = tensors[0]
    y = tensors[1]
    distance = tf.sqrt(tf.reduce_sum(tf.square(x-y), axis=1, keepdims=True))
    return distance

def conv_block(x, n_filters, size, strides=1):
    x = tf.keras.layers.Conv2D(filters=n_filters,
               kernel_size=size,
               padding='same',
               strides=strides,
               kernel_initializer='he_normal',
               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def residual_block(x, n_filters):
    skip = x
    x = conv_block(x, n_filters//2, 1)
    x = conv_block(x, n_filters, 3)
    x = tf.keras.layers.add([x, skip])
    return x

def base_model(H=180, W=480):
    input_layer = tf.keras.layers.Input(shape=(H, W, 1))
    x = conv_block(input_layer, 32, 3)
    x = conv_block(x, 64, 3, strides=2)
    x = residual_block(x, 64)
    x = conv_block(x, 128, 3, strides=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = conv_block(x, 256, 3, strides=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    output_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='base_model')
    return model   


base = base_model()

right_image = tf.keras.layers.Input(shape=(180, 480, 1))
left_image = tf.keras.layers.Input(shape=(180, 480, 1))

right_out = base(right_image)
left_out = base(left_image)

y = tf.keras.layers.Lambda(l2_distance )([right_out, left_out])
model = tf.keras.Model(inputs=[right_image, left_image], outputs=y)
model.load_weights('signature_weights.h5')


def validate(image_path_a, image_path_b, threshold=0.5):
    img_a = get_image(image_path_a)[None, ...]
    img_b = get_image(image_path_b)[None, ...]
    
    distance = np.squeeze(model.predict([img_a, img_b]))
    distance = np.clip(distance, 0, 1)
    match = 'matched' if distance < threshold else 'not matched'
    confidence = (1 - distance)
    print('\n\nStatus : ', match)
    print('Score : ', confidence)
    
validate(sys.argv[1], sys.argv[2])