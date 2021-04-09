import os
import cv2
import numpy as np
import random
def load_img_list(path):
    img = []
    for filename in os.listdir(path):
        img.append(f'{path}/{filename}')
    return img

def augment_data(list_name, save_path):
    width = 400
    height = 400
    dim = (width, height)
    for i in range(len(list_name)):
        raw_img = cv2.imread(f'{list_name[i]}')
        prepared_normal_img = cv2.resize(raw_img, dim, interpolation=cv2.INTER_AREA)
        # flipping images
        flip_both = cv2.flip(prepared_normal_img, -1)
        flip_horizontal = cv2.flip(prepared_normal_img, 1)
        flip_vertical = cv2.flip(prepared_normal_img, 0)
        # random values for brightness changes
        random_num_flip_both_img = random.randint(0, 50)
        random_num_flip_horizontal_img = random.randint(0, 50)
        random_num_flip_vertical_img = random.randint(0, 50)
        # brightness increasing: # while adding images must be the same dtype!
        prepared_flip_both_img = cv2.add(flip_both, random_num_flip_both_img * np.ones(
            (width, height, prepared_normal_img.shape[-1]), dtype=np.uint8))
        prepared_flip_horizontal_img = cv2.add(flip_horizontal, random_num_flip_horizontal_img * np.ones(
            (width, height, prepared_normal_img.shape[-1]), dtype=np.uint8))
        prepared_flip_vertical_img = cv2.add(flip_vertical, random_num_flip_vertical_img * np.ones(
            (width, height, prepared_normal_img.shape[-1]), dtype=np.uint8))

        cv2.imwrite(f'{list_name[i]}', prepared_normal_img)
        cv2.imwrite(f'{save_path}/flip_both_{i}.jpeg', prepared_flip_both_img)
        cv2.imwrite(f'{save_path}/flip_horizontal_{i}.jpeg', prepared_flip_horizontal_img)
        cv2.imwrite(f'{save_path}/flip_vertical_{i}.jpeg', prepared_flip_vertical_img)
    print(f'Data augmentation of {save_path} path has ended successfully!')


path_train_normal = 'train/NORMAL'
path_train_pneumonia = 'train/PNEUMONIA'

path_val_normal = 'val/NORMAL'
path_val_pneumonia = 'val/PNEUMONIA'

path_test_normal = 'test/NORMAL'
path_test_pneumonia = 'test/PNEUMONIA'


list_train_normal = load_img_list(path_train_normal)
list_train_pneumonia = load_img_list(path_train_pneumonia)
list_val_normal = load_img_list(path_val_normal)
list_val_pneumonia = load_img_list(path_val_pneumonia)
list_test_normal = load_img_list(path_test_normal)
list_test_pneumonia = load_img_list(path_test_pneumonia)

augment_data(list_train_normal, path_train_normal)
augment_data(list_train_pneumonia, path_train_pneumonia)
augment_data(list_val_normal, path_val_normal)
augment_data(list_val_pneumonia, path_val_pneumonia)
augment_data(list_test_normal, path_test_normal)
augment_data(list_test_pneumonia, path_test_pneumonia)