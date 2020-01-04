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
validation_data = get_validation_data(validation_data_path)
# finish getting training dataset

# set up model
base_model = tf.keras.models.load_model(MODEL_RELATIVE_PATH)
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(
    5,
    activation='softmax'
)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# finish setting up model

initial_epochs = 10
# num_train = len(list(train_data))
# steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 20

model.evaluate(validation_data, steps = validation_steps)

history = model.fit(
    train_data,
    epochs=initial_epochs,
    validation_data=validation_data,
)

model.save(MODEL_RELATIVE_PATH)
