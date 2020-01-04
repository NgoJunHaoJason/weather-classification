import os
import tensorflow as tf
from constants import *


# references:
# https://www.tensorflow.org/tutorials/images/transfer_learning
# https://www.tensorflow.org/tutorials/keras/save_and_load
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

os.makedirs('models', exist_ok=True)

base_model.save('models/weather_classification_model.h5')
