import tensorflow as tf
import tqdm
import random
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dython.nominal import associations
import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import yaml
from utils import data_loaders as dl
from models import PointNet as pointnet

from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.client import device_lib
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard


def custome_train(model, train_ds, learning_rate, max_epoxhs, val_ds, new_results_id, results_directory):

    checkpoint_filepath = os.path.join(results_directory, new_results_id + "_model_flly_supervised.h5")
    training_history = model.fit(train_ds, epochs=max_epoxhs, validation_data=val_ds) 
    model.save_weights(checkpoint_filepath)
    print('Model saved at: ', checkpoint_filepath) 
    return training_history


def main(_argv):
    tf.keras.backend.clear_session()
    path_dataset = FLAGS.path_dataset
    path_annotations = FLAGS.path_annotations
    project_folder = FLAGS.project_folder
    name_model = FLAGS.name_model
    list_variables = FLAGS.list_variables
    #  Hyperparameters 
    lr = FLAGS.learning_rate
    epochs = FLAGS.max_epochs
    batch_size = FLAGS.batch_size
    selection_radius = FLAGS.radius
    file_format = FLAGS.file_format
    num_points = FLAGS.max_num_points

    list_names_gpus = list()
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == 'GPU':
            desci = device.physical_device_desc
            list_names_gpus.append(desci.split('name: ')[-1].split(',')[0])

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    print('List names GPUs:', list_names_gpus)
    print("Num GPUs:", len(physical_devices))
    print(physical_devices)
    version_tf = tf.__version__
    print('TF Version:', version_tf)

    df_annotations = pd.read_csv(path_annotations,  on_bad_lines='skip')
    img_names_annotations = df_annotations['Plot'].tolist()
    
    # here we create 2 lists with the names of the points that will be used for training/val and test
    train_val_pointss, test_points = train_test_split(img_names_annotations, test_size=0.2, random_state=42)
    print(f'train/val data: {len(train_val_pointss)}, test data: {len(test_points)}')

    # now lets divide again into train/val:
    train_points, val_points = train_test_split(train_val_pointss, test_size=0.15, random_state=42)
    print(f'train data: {len(train_points)}, val data: {len(val_points)}')

    print('train cases:', len(train_points))
    print('val cases:', len(val_points))
    print('test cases:', len(test_points))

    dict_train = dl.build_list_dict(df_annotations, train_points, path_dataset, 
                                  file_format=file_format, selection_radius=selection_radius, 
                                  max_num_points=num_points, selected_variables=list_variables)

    dict_val = dl.build_list_dict(df_annotations, val_points, path_dataset, 
                                  file_format=file_format, selection_radius=selection_radius, 
                                  max_num_points=num_points, selected_variables=list_variables)

    train_ds = dl.tf_dataset_cloudpoints(dict_train, batch_size=batch_size, training_mode=True, selected_variables=list_variables)
    val_ds = dl.tf_dataset_cloudpoints(dict_val, batch_size=batch_size, training_mode=True, selected_variables=list_variables)    
    opt = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics=[tf.keras.metrics.MeanAbsoluteError()]

    # Compile the model 
    model = pointnet.build_pointnet(5, num_points=num_points)
    print('Compiling model')
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    training_time = datetime.now()
    new_results_id = ''.join([name_model, 
                            '_lr_',
                            str(lr),
                            '_bs_',
                            str(batch_size),
                            '_radius_',
                            str(selection_radius), 
                            '_num_points_', 
                            str(num_points),
                            '_', training_time.strftime("%d_%m_%Y_%H_%M"),
                            ])

    results_directory = ''.join([project_folder, 'results/', new_results_id, '/'])
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Save Training details in YAMLß

    patermets_traning = {'Model name':name_model, 
                         'Type of training': 'custome-training', 
                         'training date': training_time.strftime("%d_%m_%Y_%H_%M"), 
                         'learning rate':lr, 
                         'list_variables': list_variables,
                         'batch size': batch_size, 
                         'data format': file_format,
                         'radius': selection_radius,
                         'num points': num_points, 
                         'TF Version': version_tf,
                         'GPUs names': list_names_gpus, 
                         'dataset': os.path.split(path_dataset)[-1], 
                         }

    path_yaml_file = os.path.join(results_directory, 'parameters_training.yaml')
    with open(path_yaml_file, 'w') as file:
        yaml.dump(patermets_traning, file)

    train_history = custome_train(model, train_ds, lr, results_directory=results_directory, 
                                  new_results_id=new_results_id, max_epoxhs=epochs, val_ds=val_ds)
   
    try:
        path_history_plot = os.path.join(results_directory, new_results_id + "_training_history.jpg")
        plt.figure()
        plt.subplot(121)
        plt.title('MAE')
        plt.plot(model.history.history.get('mean_absolute_error'),  '-o', color='blue', label='train')
        plt.plot(model.history.history.get('val_mean_absolute_error'),  '-o', color='orange', label='val')
        plt.subplot(122)
        plt.title('Loss')
        plt.plot(model.history.history.get('loss'),  '-o', color='blue', label='train')
        plt.plot(model.history.history.get('val_loss'), '-o', color='orange', label='val')
        plt.legend(loc='best')
        plt.savefig(path_history_plot)
        plt.close()
        print('History plot saaved at: ', path_history_plot)
    
    except:
        print('Not possible to print history')
    
    vals_mae_train = model.history.history.get('mean_absolute_error')
    vals_mae_val = model.history.history.get('val_mean_absolute_error')

    vals_loss_train = model.history.history.get('loss')
    vals_loss_val = model.history.history.get('val_loss')

    df_hist = pd.DataFrame(list(zip(vals_mae_train, vals_mae_val, vals_loss_train, vals_loss_val)),
                  columns =['train MAE', 'val MAE', 'train loss', 'val loss'])
    training_history_path = os.path.join(results_directory, new_results_id + "_training_history.csv")
    df_hist.to_csv(training_history_path, index=False)
    
    # run the test 
    dict_test = dl.build_list_dict(df_annotations, test_points, path_dataset, 
                                  file_format=file_format, selection_radius=selection_radius, 
                                  max_num_points=num_points, selected_variables=list_variables)

    new_test_ds = dl.tf_dataset_cloudpoints(dict_test, batch_size=1, training_mode=False, analyze_dataset=True, 
                                            selected_variables=list_variables)

    name_files = list()
    real_vals_list = [list() for _ in list_variables]
    predictions_list = [list() for _ in list_variables]

    for x, file_path in tqdm.tqdm(new_test_ds, desc='Analyzing test dataset'):
        name_file = file_path.numpy()[0].decode("utf-8").split('/')[-1].split('.')[0]
        cloudpoint_batch = x[0]
        real_vals_batch = x[1].numpy()
        predictions = model.predict(cloudpoint_batch, verbose=0)
        name_files.append(name_file)
        for j, real_val, in enumerate(real_vals_batch.tolist()[0]):
            real_vals_list[j].append(float(real_val))

        for i, predicted_val, in enumerate(predictions.tolist()[0]):
            predictions_list[i].append(float(predicted_val))

    column_names = list_variables
    prediction_names = ['Pred. ' + f for f in column_names]
    column_names = column_names + prediction_names
    column_names.insert(0, 'File name')
    # save results in csv
    df_preds = pd.DataFrame(list(zip(name_files, *real_vals_list, *predictions_list)), columns = column_names)
    name_predictions_file = os.path.join(results_directory, f'predictions_test_ds_{str(selection_radius)}_{str(num_points)}_.csv')
    df_preds.to_csv(name_predictions_file, index=False)

    print('Experiment finished')

if __name__ == '__main__':

    flags.DEFINE_string('path_dataset', os.path.join(os.getcwd(), 'dataset'), 'directory dataset')
    flags.DEFINE_string('path_annotations', '', 'path annotations')
    flags.DEFINE_string('project_folder', os.getcwd(), 'path project folder')
    flags.DEFINE_string('results_directory', os.path.join(os.getcwd(), 'results'), 'path where to save results')
    flags.DEFINE_string('name_model', 'PointNet', 'name of the model')
    flags.DEFINE_string('file_format', '.json', 'format of the annotations')
    flags.DEFINE_list('list_variables', ['Volume', 'Hgv', 'Dgv', 'Basal_area', 'Biomassa_above'], 'variables to be used')


    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_integer('max_epochs', 4, 'epochs')
    flags.DEFINE_integer('max_num_points', 1024, 'input impage size')
    flags.DEFINE_integer('radius', 25, 'input impage size')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    #flags.DEFINE_list('num_filers', [32,64,128,256,512,1024], 'mumber of filters per layer')

    flags.DEFINE_string('type_training', '', 'eager_train or custom_training')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')

    try:
        app.run(main)
    except SystemExit:
        pass