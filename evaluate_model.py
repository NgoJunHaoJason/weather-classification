from load_images import *

# fix for incompatible cudnn version: 
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# end of fix

test_data_path = os.path.join(DATASET_BASE_PATH, 'processed_images/test')
test_data = get_val_or_test_data(test_data_path)

model = tf.keras.models.load_model(MODEL_RELATIVE_PATH)

model.evaluate(test_data)
