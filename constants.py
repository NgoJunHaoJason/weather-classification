# define constants here


IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 10
VALIDATION_STEPS = 20

CLASS_NAMES = ['cloudy', 'foggy', 'rain', 'snow', 'sunny']
DATASET_BASE_PATH = ''  # set your own path
MODEL_RELATIVE_PATH = 'models/weather_classification_model.h5'
