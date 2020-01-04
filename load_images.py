import os
import tensorflow as tf
from constants import *


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_image(image):
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)

    # resize the image to the desired size.
    return tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)

    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = decode_image(image)

    return image, label


def prepare_for_training(dataset, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    # dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def get_train_data(train_data_path):
    train_list = tf.data.Dataset.list_files(train_data_path + '/*/*')
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_train = train_list.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_batches = prepare_for_training(labeled_train, cache=False)

    return train_batches


def get_validation_data(validation_data_path):
    validation_list = tf.data.Dataset.list_files(validation_data_path + '/*/*')
    labeled_validation = validation_list.map(
        process_path, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    validation_batches = labeled_validation.batch(BATCH_SIZE)
    validation_batches = validation_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return validation_batches
