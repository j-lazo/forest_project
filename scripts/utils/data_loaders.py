import numpy as np
import tensorflow as tf
import os
import pandas as pd
import cv2
import random

def read_img(dir_image, img_size=(256, 256)):
    path_img = dir_image.decode()
    original_img = cv2.imread(path_img)
    img = cv2.resize(original_img, img_size, interpolation=cv2.INTER_AREA)
    img = img / 255.
    return img


def read_mask(path, img_size=(256, 256), thresh_value=127):
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, img_size, interpolation=cv2.INTER_AREA)
    thresh, x = cv2.threshold(x, thresh_value, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def adjust_brightness(image, gamma=1.0):
    image = image * 255.
    image = image.astype(np.uint8)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)/255

def add_salt_and_pepper_noise(image, noise_ratio=0.05):
    # salt & pepepr noise
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)

    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [1, 1, 1]

    return noisy_image
    
def add_gaussian_noise(image, mean=0, std=0.5):
    # gausian_noise 
    image = image * 255.
    image = image.astype(np.uint8)
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image/255 

def rotate_90(img, mask):
    # roatation 90
    rows, cols, ch = img.shape
    rot1 = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
    img = cv2.warpAffine(img, rot1, (cols, rows))
    mask = cv2.warpAffine(mask, rot1, (cols, rows))
    mask = np.expand_dims(mask,axis=-1)

    return img, mask

def rotate_180(img, mask):
    # roatation 180
    rows, cols, ch = img.shape
    rot2 = cv2.getRotationMatrix2D((cols/2,rows/2), 180, 1)
    img = cv2.warpAffine(img, rot2, (cols, rows))
    mask = cv2.warpAffine(mask, rot2, (cols, rows))
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

def rotate_270(img, mask):
    # rotation 270
    rows, cols, ch = img.shape
    rot3 = cv2.getRotationMatrix2D((cols/2,rows/2), 270, 1)
    img = cv2.warpAffine(img, rot3, (cols, rows))
    mask = cv2.warpAffine(mask, rot3, (cols, rows))
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

def random_rotate(img, mask):
    choice = random.randint(0,2)
    if choice == 0:
        img, mask = rotate_90(img, mask)

    elif choice == 1:
        img, mask = rotate_180(img, mask)

    elif choice == 2:
        img, mask = rotate_270(img, mask)

    return img, mask

def flip_vertical(img, mask):
    # vertical flip 
    img = cv2.flip(img, 1)
    mask = cv2.flip(mask, 1)
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

def flip_horizontal(img, mask):
    # horizontal flip 
    img = cv2.flip(img, 0)
    mask = cv2.flip(mask, 0)
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

# TF dataset cloud-points
def tf_dataset_cloudpoints(annotations_dict, batch_size=8, training_mode=False, analyze_dataset=False, num_points=2048):
    img_size = img_size

    # def tf_parse(x, y):
    #     def _parse(x):
    #         x = read_asl(x)
    #         return x

    #     x = tf.numpy_function(_parse, [x], [tf.float64])
    #     #x.set_shape([img_size, img_size, 3])
    #     return x, y
    
    def configure_for_performance(dataset, batch_size):
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.batch(batch_size, drop_remainder=True)        
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    list_files = list(annotations_dict.keys())
    path_imgs = list()
    images_class = list()
    data_points = list()


    if training_mode:
        random.shuffle(list_files)
    
    for img_id in list_files:
        path_imgs.append(annotations_dict[img_id]['path_raster_file'])
        #path_imgs.append(annotations_dict[img_id]['path_ir_img'])
        #biomass_above = annotations_dict[img_id]['biomass_above']
        #basal_area = annotations_dict[img_id]['basal_area']
        volume = annotations_dict[img_id]['volume']
        #dgv = annotations_dict[img_id]['dgv']
        hgv = annotations_dict[img_id]['hgv']
        points_x = annotations_dict[img_id]['points_x']
        points_y = annotations_dict[img_id]['points_y']
        points_z = annotations_dict[img_id]['points_z']
        #data_points = np.transpose(np.array[points_x, points_y, points_z])
        data_points.append(tf.transpose([points_x, points_y, points_z]))
        #images_class.append([biomass_above, basal_area, volume, dgv, hgv])
        images_class.append([volume, hgv])

    labels = images_class
    dataset = tf.data.Dataset.from_tensor_slices((data_points, labels))

    if analyze_dataset:
        filenames_ds = tf.data.Dataset.from_tensor_slices(path_imgs)
        dataset = tf.data.Dataset.zip(dataset, filenames_ds)

    if training_mode:
        dataset = configure_for_performance(dataset, batch_size=batch_size)
    else:
        dataset = dataset.batch(batch_size,  drop_remainder=True)

    dataset = dataset.prefetch(1)
    print(f'TF dataset with {len(path_imgs)} elements')

    return dataset