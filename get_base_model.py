import os
import tensorflow as tf


# references:
# https://www.tensorflow.org/tutorials/images/transfer_learning
# https://www.tensorflow.org/tutorials/keras/save_and_load

IMG_SIZE = 224  # All images will be resized to 160x160

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

os.makedirs('models', exist_ok=True)

base_model.save('models/weather_classification_model.h5')
