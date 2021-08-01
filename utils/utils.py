import os
import operator
import sklearn
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.constants import DATASETS_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

## Reads UCR File
def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


## Creates directory for various use provided by 
## https://github.com/hfawaz/dl-4-tsc/blob/master/utils/utils.py
def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

## Creates paths for Results per Classifier and Archive Name -> Per Classifier AND Feature AND ARCHIVE TODO:
def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


## Save model test duration TODO:
def save_test_duration(output_path, duration):
    pass

## Reads dataset
def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'UCRArchive_2018':
        root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
        df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                    y_test.copy())
    else:
        file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = readucr(file_name + '_TRAIN')
        x_test, y_test = readucr(file_name + '_TEST')
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                    y_test.copy())

    return datasets_dict


def read_all_datasets(root_dir, archive_name, split_val=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []

    if archive_name == 'UCRArchive_2018':
        for dataset_name in DATASETS_NAMES:
            root_dir_dataset = cur_root_dir + '/data/' + archive_name + '/' + dataset_name

            # print(root_dir_dataset)

            df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                        y_test.copy())

    else:
        for dataset_name in DATA_SETS:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                        y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATA_SETS)):
            DATA_SETS[i] = dataset_names_to_sort[i][0]

    return datasets_dict


def shape_data(dataset):
    x_train, y_train, x_test, y_test = dataset

    _x_train = x_train.copy()
    _y_train = y_train.copy()
    _x_test = x_test.copy()
    _y_test = y_test.copy()

    nb_classes = len(np.unique(np.concatenate((_y_train, _y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((_y_train, _y_test), axis=0).reshape(-1, 1))
    _y_train = enc.transform(_y_train.reshape(-1, 1)).toarray()
    _y_test = enc.transform(_y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(_y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    return _x_train, _y_train, _x_test, _y_test, y_true, nb_classes, input_shape


## Calculate metrics: precision, accuracy, recall + report (duration) TODO:
def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                    columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res
    

## Saves model logs TODO:
def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, 
        y_true_val=None, y_pred_val=None, model_name='_'):
    
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + model_name +  'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + model_name + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                        'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+ model_name + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def plot_relevance(timeseries, relevance, figsize=(15, 1), alpha=.5, linewidth=3.6, colormap='Reds', 
        show=True, save=False, output_dir=None, file_name=None):
    cmap = plt.get_cmap(colormap)
    
    timeseries_in = np.array(timeseries)
    relevance_in = np.array(relevance)
    
    if timeseries_in.shape != relevance_in.shape:
        raise Exception('Time series and relevance are not the same shape')
    
    if type(timeseries_in.tolist())!=list or type(relevance_in.tolist())!=list:
        timeseries_in = np.array([timeseries])
        relevance_in = np.array([relevance])
        

    if len(timeseries_in.shape) == 1:
        n_ts = 1
        tsmin = timeseries_in.min()
        tsmax = timeseries_in.max()
        tsheight = round(abs(tsmin) + abs(tsmax)) + 1
        tswidth = timeseries_in.shape[0]
        x, y = generate_filled_mngrid(tsmin, tsmax, tswidth)
        tsheight = tsheight+(x.shape[0]-tsheight)
        z = np.array([relevance for idx in range(int(tsheight))])
        fig, ax = plt.subplots(nrows=n_ts, ncols=1, figsize=figsize)
        ax.pcolormesh(x, y, z, cmap='Reds', shading='auto')
        ax.plot(timeseries_in)
    else:
        n_ts = timeseries_in.shape[0]
        new_figsize = (15*.84, n_ts*1.35)            
        fig, ax = plt.subplots(nrows=n_ts, ncols=1, figsize=new_figsize)
        fig.tight_layout(pad=2)
        for i in range(n_ts):
            tsmin = timeseries_in[i].min()
            tsmax = timeseries_in[i].max()
            tsheight = round(abs(tsmin) + abs(tsmax)) + 1
            tswidth = timeseries_in[i].shape[0]
            x, y = generate_filled_mngrid(tsmin, tsmax, tswidth)
            tsheight = tsheight+(x.shape[0]-tsheight)
            z = np.array([relevance[i] for idx in range(int(tsheight))])
            ax[i].plot(timeseries_in[i])
            ax[i].pcolormesh(x, y, z, cmap='Reds', shading='auto')
    
    if show:
        plt.show()
    if output_dir and save:
        if file_name:
            save_pig(output_dir+'/'+file_name+'_explanation.png')
        else:
            save_pig(output_dir+'/explanation.png')
    
def generate_filled_mngrid(min_val, max_val, width):
    # (height, width) (abs(min)+max, width)
    x, y = [], []
    min_f, max_c = math.floor(min_val), math.ceil(max_val)
    height = abs(min_f) + abs(max_c)

    ## gen x
    for row_idx in range(height + 1):
        init_val = 0
        to_append = []
        for col_idx in range(width):
            to_append.append(init_val)
            init_val+=1
        x.append(to_append)
    x = np.array(x)

    ## gen y
    init_val = min_f
    for row_idx in range(height + 1):
        to_append = []
        for col_idx in range(width):
            to_append.append(init_val)
        init_val += 1
        y.append(to_append)
    y = np.array(y)
    
    return x, y


