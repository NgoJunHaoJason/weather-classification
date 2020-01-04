import datetime
from load_images import *

# fix for incompatible cudnn version: 
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# end of fix

# get dataset
train_data_path = os.path.join(DATASET_BASE_PATH, 'processed_images/train')
train_data = get_train_data(train_data_path)

validation_data_path = os.path.join(DATASET_BASE_PATH, 'processed_images/validation')
validation_data = get_val_or_test_data(validation_data_path)
# finish getting training dataset

model = tf.keras.models.load_model(MODEL_RELATIVE_PATH)
base_model = model.layers[0]
base_model.trainable = True

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=BASE_LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs =  INITIAL_EPOCHS + fine_tune_epochs

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_data,
    epochs=total_epochs,
    initial_epoch=INITIAL_EPOCHS,
    validation_data=validation_data,
    callbacks=[tensorboard_callback],
)

model.save(MODEL_RELATIVE_PATH)
