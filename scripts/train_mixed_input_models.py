import tensorflow as tf
import tqdm
import random
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
import yaml
from utils import data_loaders as dl
from models import PointNet as pointnet


from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.client import device_lib
from datetime import datetime
import rasterio
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from models import conv_models as cm
from models import mixed_input_models as mi


def model_fit(ds_train, ds_val, model):

    loss_fn = tf.keras.losses.MeanSquaredError()
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    train_history = model.fit(ds_train, epochs=5, validation_data=ds_val)

    return train_history


def custome_train(model, train_dataset, num_training_samples, learning_rate, max_epoxhs, val_dataset, new_results_id, results_directory, patience=5):

    @tf.function
    def train_step(inputs, labels):
        list_metrics_batch = list()
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            train_loss = loss_fn(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        for metric in train_metrics:
            list_metrics_batch.append(metric(labels, predictions))

        return train_loss, list_metrics_batch

    @tf.function
    def valid_step(inputs, labels):
        list_metrics_batch = list()
        predictions = model(inputs, training=False)
        val_loss = loss_fn(labels, predictions)
        for metric in train_metrics:
            list_metrics_batch.append(metric(labels, predictions))

        return val_loss, list_metrics_batch
    
    @tf.function
    def prediction_step(images):
        predictions = model(images, training=False)
        return predictions

    # define loss and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # train/val loss and metrics histories     
    train_loss_hist = list()
    val_loss_hist = list()

    train_metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    val_metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

    #train_metrics = [metrics.metric_RootMeanSquaredError, metrics.metric_MeanSquaredError, metrics.metric_MeanAbsoluteError]
    #val_metrics = [metrics.metric_RootMeanSquaredError, metrics.metric_MeanSquaredError, metrics.metric_MeanAbsoluteError]
    metrics_names = ['MSE', 'RMSE', 'MAE'] 

    train_metrics_hist = [list() for _ in train_metrics]
    val_metrics_hist = [list() for _ in val_metrics]

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    f_metrics_train = list()
    f_metrics_val = list()
    for j, _ in enumerate(train_metrics_hist):
        f_metrics_train.append(tf.keras.metrics.Mean(name='train_' + metrics_names[j]))
        f_metrics_val.append(tf.keras.metrics.Mean(name='val_' + metrics_names[j]))

    patience = patience
    wait = 0
    best = 0
    #num_training_samples = [i for i,_ in enumerate(train_dataset)][-1] + 1
    checkpoint_filepath = os.path.join(results_directory, new_results_id + "_model_.h5")
    # start training
    for epoch in range(max_epoxhs):
        print("\nepoch {}/{}".format(epoch+1, max_epoxhs))
        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=metrics_names)
        epoch_train_loss_list = list()
        epoch_train_loss_val = list()

        epoch_train_metrics_list = [list() for _ in range(len(train_metrics))]
        epoch_val_metrics_list = [list() for _ in range(len(val_metrics))]
        
        for idX, batch_ds in enumerate(train_dataset):
            inputs = [batch_ds[0], batch_ds[1]]
            train_labels = batch_ds[2]
            train_loss_value, batch_training_metrics = train_step(inputs, train_labels)
            epoch_train_loss_list.append(train_loss_value)
            values_prog_bar = list()
            for j, metric in enumerate(batch_training_metrics):
                epoch_train_metrics_list[j].append(metric)
                values_prog_bar.append(('train_' + metrics_names[j],  metric))
            values_prog_bar.append(('train_loss', train_loss_value))
            progBar.update(idX+1, values=values_prog_bar) 
        
        # calcualate the train loss and metrics and save them into the history list 
        loss_epoch = train_loss(epoch_train_loss_list)
        for j, metric_list in enumerate(epoch_train_metrics_list):
            m = f_metrics_train[j](metric_list).numpy()
            train_metrics_hist[j].append(m)

        train_loss_hist.append(loss_epoch.numpy())

        # Reset training metrics at the end of each epoch
        train_loss.reset_states()
        for f in f_metrics_train:
            f.reset_states()
                
        if val_dataset:
            for batch_ds in val_dataset:
                inputs = [batch_ds[0], batch_ds[1]]
                valid_labels = batch_ds[2]
                val_loss_value, batch_validation_metrics = valid_step(inputs, valid_labels)
                epoch_train_loss_val.append(val_loss_value)
                values_prog_bar_val = list()
                for j, metric in enumerate(batch_validation_metrics):
                    epoch_val_metrics_list[j].append(metric)
                    values_prog_bar_val.append(('val_' + metrics_names[j],  metric))
                values_prog_bar_val.append(('val_loss', val_loss_value))
            progBar.update(idX+1, values=values_prog_bar_val) 

            # calcualate the val loss and metrics and save them into the history list 
            loss_epoch_val = valid_loss(epoch_train_loss_val).numpy()
            for j, metric_list in enumerate(epoch_val_metrics_list):
                m = f_metrics_val[j](metric_list).numpy()
                val_metrics_hist[j].append(m)

            val_loss_hist.append(loss_epoch_val)

            # Reset validation metrics at the end of each epoch
            valid_loss.reset_states()
            for f in f_metrics_val:
                f.reset_states()
            
        wait += 1
        if epoch == 0:
            best = loss_epoch_val
        if loss_epoch_val < best:
            best = loss_epoch_val
            wait = 0
            model.save_weights(checkpoint_filepath)

        if wait >= patience:
            print(f'Early stopping triggered at epoch {epoch}: wait time > patience')
            break
    
    fina_model_name = os.path.join(results_directory, new_results_id + "_model_last_epoch.h5")
    model.save_weights(fina_model_name)
    print('Model saved at: ', checkpoint_filepath)    

    dict_history = {'train_loss': train_loss_hist,
                    'val_loss':val_loss_hist}
    
    for i, metric_list in enumerate(train_metrics_hist):
        dict_history['train_' + metrics_names[i]] = metric_list

    for i, metric_list in enumerate(val_metrics_hist):
        dict_history['val_' + metrics_names[i]] = metric_list

    return dict_history


def main(_argv):
    tf.keras.backend.clear_session()

    path_raster_files = FLAGS.path_raster_files
    path_cloud_points = FLAGS.path_cloud_points
    path_annotations = FLAGS.path_annotations
    project_folder = FLAGS.project_folder
    name_model = FLAGS.name_model
    list_variables = FLAGS.list_variables
    data_type_raster = FLAGS.data_type
    num_points = FLAGS.max_num_points
    radius_pointcloud = FLAGS.radius_pointcloud
    selection_radius_raster = FLAGS.radius_raster
    num_channels = FLAGS.num_channels

    #  Hyperparameters 
    lr = FLAGS.learning_rate
    epochs = FLAGS.max_epochs
    batch_size = FLAGS.batch_size
    augmentation_functions_raster = FLAGS.augmentation_functions_raster
    augmentation_functions_cloudpoints = FLAGS.augmentation_functions_cloudpoints

    if augmentation_functions_raster == ['all']:
        augmentation_functions_raster = ['None', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_vertical', 'flip_horizontal']
    
    if augmentation_functions_cloudpoints == ['all']:
        augmentation_functions_cloudpoints = ['None', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_vertical', 'flip_horizontal']

    list_names_gpus = list()
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == 'GPU':
            desci = device.physical_device_desc
            list_names_gpus.append(desci.split('name: ')[-1].split(',')[0])

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        print("Name:", gpu.name, "Type:", gpu.device_type)

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


    dict_train = dl.build_list_dict_mixed_input_data(df_annotations, train_points, path_raster_files, path_cloud_points, selected_variables=list_variables)
    dict_val = dl.build_list_dict_mixed_input_data(df_annotations, val_points, path_raster_files, path_cloud_points, selected_variables=list_variables)
    dict_test = dl.build_list_dict_mixed_input_data(df_annotations, test_points, path_raster_files, path_cloud_points, selected_variables=list_variables)


    train_ds, num_training_samples = dl.tf_dataset_raster_and_cloudpoints(dict_train, batch_size=batch_size, training_mode=True, num_points=num_points, 
                                                    augmentation_raster=True, augmentation_cloudpoints=True, radius_raster=selection_radius_raster,
                                                    radius_cloudpoints=radius_pointcloud, augmentation_functions_raster=augmentation_functions_raster, 
                                                    augmentation_functions_cloudpoints=augmentation_functions_cloudpoints)
    val_ds, _ = dl.tf_dataset_raster_and_cloudpoints(dict_val, batch_size=batch_size, training_mode=True, num_points=num_points, 
                                                    augmentation_raster=False, augmentation_cloudpoints=False, radius_raster=selection_radius_raster,
                                                    radius_cloudpoints=radius_pointcloud)
    new_test_ds, _ = dl.tf_dataset_raster_and_cloudpoints(dict_test, batch_size=1, analyze_dataset=True, training_mode=False, num_points=num_points,
                                                       augmentation_raster=False, augmentation_cloudpoints=False, radius_raster=selection_radius_raster,
                                                       radius_cloudpoints=radius_pointcloud)

    opt = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics=[tf.keras.metrics.MeanAbsoluteError()]

    # Compile the model 
    if name_model == 'Simple_mixed_model':

        model = mi.mixed_input_model(len(list_variables), 
                                     input_shape_iamge=(selection_radius_raster*2+1,selection_radius_raster*2+1,num_channels), 
                                     num_points=num_points)
        print('Compiling simple Mixed Model')
        model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    if num_channels == 60:
        extra_data = 'raster'
    elif num_channels == 13:
        extra_data = 'Sentinel2'

    training_time = datetime.now()
    new_results_id = ''.join([name_model, '_mixed_input_data_', 'pointclouds+', extra_data, 
                            '_lr_',
                            str(lr),
                            '_bs_',
                            str(batch_size),
                            '_radiusraster_',
                            str(selection_radius_raster), 
                            '_radiuspointcloud_',
                            str(radius_pointcloud),
                            '_npoints_', 
                            str(num_points),
                            '_', training_time.strftime("%d_%m_%Y_%H_%M"),
                            ])

    results_directory = ''.join([project_folder, 'results/', new_results_id, '/'])
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Save Training details in YAML

    patermets_traning = {'Model name':name_model, 
                            'Type of training': 'custome-training', 
                            'training date': training_time.strftime("%d_%m_%Y_%H_%M"), 
                            'learning rate':lr, 
                            'list_variables': list_variables,
                            'batch size': batch_size, 
                            'radius raster': selection_radius_raster,
                            'radius pointcloud': radius_pointcloud,
                            'num points': num_points,
                            'TF Version': version_tf,
                            'GPUs names': list_names_gpus, 
                            'dataset raster': os.path.split(path_raster_files)[-1],
                            'dataset poinstclouds': os.path.split(path_cloud_points)[-1], 
                            'augmentation operations raster': augmentation_functions_raster,
                            'augmentation operations pointclouds': augmentation_functions_cloudpoints,
                            }

    path_yaml_file = os.path.join(results_directory, 'parameters_training.yaml')
    with open(path_yaml_file, 'w') as file:
        yaml.dump(patermets_traning, file)
    train_history_dict = custome_train(model, train_ds, num_training_samples, lr, results_directory=results_directory, 
                                    new_results_id=new_results_id, max_epoxhs=epochs, val_dataset=val_ds)


    try:
        path_history_plot = os.path.join(results_directory, new_results_id + "_training_history.jpg")
        plt.figure()

        name_metrics = train_history_dict.keys()
        name_metrics = [s.replace('train_', '').replace('val_', '') for s in name_metrics]

        unique_metric_names = list(np.unique(name_metrics))

        plt.figure(figsize=(11,13))
        for j, metric_name in enumerate(unique_metric_names):
            plt.subplot(len(unique_metric_names), 1, j+1)
            plt.title(metric_name)
            plt.plot(train_history_dict.get('train_' + metric_name), '-o', color='blue', label='train')
            plt.plot(train_history_dict.get('val_' + metric_name),  '-o', color='orange', label='val')
            plt.legend(loc='best')

        plt.savefig(path_history_plot)
        plt.close()
        print('History plot saaved at: ', path_history_plot)

    except:
        print('Not possible to print history')


    df_hist = pd.DataFrame.from_dict(train_history_dict)

    training_history_path = os.path.join(results_directory, new_results_id + "_training_history.csv")
    df_hist.to_csv(training_history_path, index=False)

    # run the test 

    name_files = list()
    real_vals_list = [list() for _ in list_variables]
    predictions_list = [list() for _ in list_variables]

    list_all_models = os.listdir(results_directory)
    list_all_models = [f for f in list_all_models if f.endswith('.h5')]
    
    if len(list_all_models) > 1:
        list_all_models  = [m for m in list_all_models if '_model_last_epoch' not in m]

    best_model_name = list_all_models.pop()
    path_best_model = os.path.join(results_directory, best_model_name)

    if name_model == 'Simple_mixed_model':

        model_prediction = mi.mixed_input_model(len(list_variables), 
                                        input_shape_iamge=(selection_radius_raster*2+1,selection_radius_raster*2+1,num_channels), 
                                        num_points=num_points)
        print('Prediction Model Loaded')

    model_prediction.load_weights(path_best_model) #tf.keras.models.load_model(path_best_model)

    for x, file_path in tqdm.tqdm(new_test_ds, desc='Analyzing test dataset'):
        name_file = file_path.numpy()[0].decode("utf-8").split('/')[-1].split('.')[0]
        cloudpoint_batch = x[0]
        real_vals_batch = x[1].numpy()
        predictions = model_prediction.predict(cloudpoint_batch, verbose=0)
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
    name_predictions_file = os.path.join(results_directory, f'predictions_test_ds_.csv')
    df_preds.to_csv(name_predictions_file, index=False)

    print('Experiment finished')

if __name__ == '__main__':

    flags.DEFINE_string('path_raster_files', os.path.join(os.getcwd(), 'dataset'), 'directory dataset raster files')
    flags.DEFINE_string('path_cloud_points', os.path.join(os.getcwd(), 'dataset'), 'directory dataset cloupoint files')
    flags.DEFINE_string('path_annotations', '', 'path annotations')
    flags.DEFINE_string('project_folder', os.getcwd(), 'path project folder')
    flags.DEFINE_string('results_directory', os.path.join(os.getcwd(), 'results'), 'path where to save results')
    flags.DEFINE_string('name_model', 'Simple_CNN', 'name of the model')
    flags.DEFINE_list('list_variables', ['Volume', 'Hgv', 'Dgv', 'Basal_area', 'Biomassa_above'], 'variables to be used')
    flags.DEFINE_string('data_type', 'npy', 'format of the annotations')

    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_integer('max_epochs', 4, 'epochs')
    flags.DEFINE_integer('radius_raster', 2, 'input impage size')
    flags.DEFINE_integer('radius_pointcloud', 2, 'radius of selection pointcloud')
    flags.DEFINE_integer('max_num_points', 1024, 'number of points samples')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('num_channels', 60, 'number of channels to use')
    flags.DEFINE_list('augmentation_functions_raster', ['all'], 'agumentation functions used')
    flags.DEFINE_list('augmentation_functions_cloudpoints', ['all'], 'agumentation functions used')


    flags.DEFINE_string('type_training', '', 'fit_training or custom_training')

    try:
        app.run(main)
    except SystemExit:
        pass