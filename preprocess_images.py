import os
import shutil
from constants import DATASET_BASE_PATH


data_dir_path = DATASET_BASE_PATH

source_dir = 'Image'
source_dir_path = os.path.join(data_dir_path, source_dir)

destination_dir = 'processed_images'
destination_dir_path = os.path.join(data_dir_path, destination_dir)
os.makedirs(destination_dir_path, exist_ok=True)

train_dir_path = os.path.join(destination_dir_path, 'train')
validation_dir_path = os.path.join(destination_dir_path, 'validation')
test_dir_path = os.path.join(destination_dir_path, 'test')

os.makedirs(train_dir_path, exist_ok=True)
os.makedirs(validation_dir_path, exist_ok=True)
os.makedirs(test_dir_path, exist_ok=True)

for class_dir in os.listdir(source_dir_path):
    class_dir_path = os.path.join(source_dir_path, class_dir)

    if not os.path.isdir(class_dir_path) or class_dir == 'z-other':
        continue

    print('working on', class_dir, 'images')
    
    class_train_dir_path = os.path.join(train_dir_path, class_dir)
    class_validation_dir_path = os.path.join(validation_dir_path, class_dir)
    class_test_dir_path = os.path.join(test_dir_path, class_dir)

    os.makedirs(class_train_dir_path, exist_ok=True)
    os.makedirs(class_validation_dir_path, exist_ok=True)
    os.makedirs(class_test_dir_path, exist_ok=True)

    for index, image in enumerate(os.listdir(class_dir_path)):
        image_path = os.path.join(class_dir_path, image)

        if not os.path.isfile(image_path) or not image.endswith('.jpg'):
            continue

        if index % 10 == 3:
            image_destination_path = os.path.join(class_validation_dir_path, image)
        elif index % 10 == 6:
            image_destination_path = os.path.join(class_test_dir_path, image)
        else:
            image_destination_path = os.path.join(class_train_dir_path, image)

        shutil.copy(image_path, image_destination_path)

print('done')
