import numpy as np
import tensorflow as tf
import os
import pandas as pd
import cv2
import random
import tqdm
import json

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


def get_points_in_radius(points_x, points_y, points_z, radius=10):
    seleceted_x = list()
    seleceted_y = list()
    seleceted_z = list()

    min_point_x = - int(radius/2)
    min_point_y = - int(radius/2)
    
    max_point_x = int(radius/2)
    max_point_y = int(radius/2)
    
    new_array = np.array([points_x, points_y, points_z])
    list_points = list(np.transpose(new_array))
    for point in list_points:
         if (min_point_x < point[0] < max_point_x) and (min_point_y < point[1] < max_point_y):
            seleceted_x.append(point[0])
            seleceted_y.append(point[1])
            seleceted_z.append(point[2])

    return seleceted_x, seleceted_y, seleceted_z
        

def extend_list(original_list, num_points):
    # Make a copy of the original list to avoid modifying it directly
    extended_list = original_list[:]
    # Keep adding random elements until the threshold is reached
    while len(extended_list) < num_points:
        extended_list.append(random.choice(original_list))
    
    return extended_list

def get_cloud_points_json(path_file_json, max_num_points=1024, radius=25):

    f = open (path_file_json, "r")
    data = json.loads(f.read())
    f.close()

    px = data.get('points_x')
    py = data.get('points_y')
    pz = data.get('points_z')
    points_x, points_y, points_z = get_points_in_radius(px, py, pz, radius=radius)
    new_array = np.array([points_x, points_y, points_z])
    reshaeped = np.transpose(new_array).tolist()

    if len(points_x) < max_num_points:
        seleceted = extend_list(reshaeped, max_num_points)

    else:
        seleceted = random.sample(reshaeped, max_num_points)

    seleceted_x = [x[0] for x in seleceted]
    seleceted_y = [x[1] for x in seleceted]
    seleceted_z = [x[2] for x in seleceted]

    return seleceted_x, seleceted_y, seleceted_z


# Dict annotations
def build_list_dict(df_annotations, list_data_points, path_annotations, file_format, 
                    selected_variables=['Volume', 'Hgv', 'Dgv', 'Basal_area', 'Biomassa_above'],
                    selection_radius=30, max_num_points=2048):
    
    dict_data = {}
    mask = df_annotations['Description'].isin(list_data_points)
    new_df = df_annotations[mask]

    temp_dict_variables = {x:None for x in selected_variables}
    list_names = new_df['Description'].tolist()
    
    for variable_name in selected_variables:
        temp_dict_variables[variable_name] = new_df[variable_name].tolist()

    for idx, data_name in enumerate(tqdm.tqdm(list_names, desc='Buidling dictionary')):
        
        path_cloud_point_data = os.path.join(path_annotations, str(data_name) + file_format)
        if os.path.isfile(path_cloud_point_data):
            try: 
                points_x, points_y, points_z = get_cloud_points_json(path_cloud_point_data, max_num_points=max_num_points, 
                                                                        radius=selection_radius)    
                dict_data_point = {'id_name': list_names[idx],
                                  'path_file_point_cloud': path_cloud_point_data,
                                  'points_x': points_x, 
                                  'points_y': points_y, 
                                  'points_z': points_z}

                for variable_name in selected_variables:
                    dict_data_point[variable_name] = temp_dict_variables[variable_name][idx]
                
                dict_data[data_name] = dict_data_point
            except:
                print(f'{path_cloud_point_data} broken')
            
    return dict_data


# TF dataset cloud-points
def tf_dataset_cloudpoints(annotations_dict, batch_size=8, training_mode=False, analyze_dataset=False, num_points=1024, 
                           selected_variables=['Volume', 'Hgv', 'Dgv', 'Basal_area', 'Biomassa_above']):
    
    def configure_for_performance(dataset, batch_size):
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.batch(batch_size, drop_remainder=True)        
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    list_files = list(annotations_dict.keys())
    path_files_data = list()
    list_all_variables = list()
    data_cloud_points = list()
    if training_mode:
        random.shuffle(list_files)
    
    for data_id in list_files:
        path_file_data = annotations_dict.get(data_id).get('path_file_point_cloud')
        path_files_data.append(path_file_data)
        points_x = annotations_dict.get(data_id).get('points_x')
        points_y = annotations_dict.get(data_id).get('points_y')
        points_z = annotations_dict.get(data_id).get('points_z')
        data_cloud_points.append(tf.transpose([points_x, points_y, points_z]))
        data_point_vars = list()
        for j, variable_name in enumerate(selected_variables):
            data_point_vars.append(annotations_dict.get(data_id).get(variable_name))
        list_all_variables.append(data_point_vars)

    
    dataset = tf.data.Dataset.from_tensor_slices((data_cloud_points, list_all_variables))

    if analyze_dataset:
        filenames_ds = tf.data.Dataset.from_tensor_slices(path_files_data)
        dataset = tf.data.Dataset.zip(dataset, filenames_ds)

    if training_mode:
        dataset = configure_for_performance(dataset, batch_size=batch_size)
    else:
        dataset = dataset.batch(batch_size,  drop_remainder=True)

    dataset = dataset.prefetch(1)
    print(f'TF dataset with {len(path_files_data)} elements')

    return dataset