import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import tensorflow as tf

from utils.utils import read_all_datasets
from explanations import OcclusionSensitivityUTS
from explanations import MeaningfulPerturbationUTS
from explanations import LimeUTS
from explanations import RiseUTS
from explanations import AnchorUTS

from lime import explanation
from lime import lime_base

from evaluations import PerturbationAnalysisUTS

################################### Config #####################################
np.set_printoptions(threshold=sys.maxsize)


################################### Methods ####################################

def shape_data(dataset):
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


################################ Load Data set #################################
## For Notebook
# root_directory = 'C:/git/explic-ai-tsc'

## For PC
root_directory = 'D:/git/explic-ai-tsc'

dataset_dict = read_all_datasets(root_directory, 'UCRArchive_2018')

curr_dataset = 'ECG5000'

x_train, y_train, x_test, y_test = dataset_dict[curr_dataset]

x_train, y_train, x_test, y_test, y_true, nb_classes, input_shape = shape_data(x_train, y_train, x_test, y_test)

print('Dataset shape')
print(x_train.shape)
print()
print('y_true')
print(y_true.shape)
# print(y_true)
print('y_train')
print(y_train.shape)
print('y_test')
print(y_test.shape)
# print(y_test)

timeseries_instance = x_test[0]

############################ Load Pretrained Model #############################
from classifiers import MLP
output_directory_model = root_directory + '/results/MLP/UCRArchive_2018_itr_0/ECG5000/'
model = MLP(output_directory_model, input_shape, nb_classes, verbose=True, build=False)

## Load model base accuracy (test accuracy)
output_directory_model_results = root_directory + '/results/MLP/UCRArchive_2018_itr_0/ECG5000/_df_metrics.csv'
test_accuracy = np.genfromtxt(output_directory_model_results, delimiter=',', skip_header=1)[1]

# print('test_accuracy')
# print(test_accuracy)
# print()

# metrics = model.predict(x_test, y_true, x_train, y_train, y_test)
# print(metrics)

################################## Occlusion ###################################
explainer = OcclusionSensitivityUTS()

# relevance = explainer.explain_instance(timeseries_instance, true_class=1, model=model, patch_size=4)
#################################### LIME ######################################
# explainer = LimeUTS()

## Hyperparams
# number of samples: 1000 (look for hyperparam)
# explained_ts = explainer.explain_instance(timeseries_instance, true_class=1, model=model, patch_size=4)
#################################### RISE ######################################
# explainer = RiseUTS()

## Hyperparams
# ResNet50 : 8000 masks 
# h = w = 7 and H = W = 224; 7 * 32 = 224 (1/32tel) / 
# explained_ts = explainer.explain_instance(timeseries_instance, true_class=1, model=model) 
################################### Anchor #####################################
# explainer = AnchorUTS()

# explained_ts = explainer.explain_instance(timeseries_instance, true_class=1, model=model, patch_size=4)
########################### Meaningful Perturbation ############################
# explainer = MeaningfulPerturbationUTS()

# explained_ts = explainer.explain_instance(timeseries_instance, true_class=1, model=model, patch_size=4)

############################ Perturbation Analysis #############################
evaluator = PerturbationAnalysisUTS()

print()
print()

print()
print('Evaluation of explanation method per instance')
# print(evaluation.shape)

evaluation = evaluator.evaluate__xai_method(
    test_accuracy, x_test, y_true, x_train, y_train, y_test,
    model, explainer, verification_method='all',batch_size=2,
    sequence_length=8
)

# evaluation = evaluator.evaluate__xai_method(
#     test_accuracy, x_test, y_true, x_train, y_train, y_test,
#     model, explainer, verification_method='zero_sequence',batch_size=2,
#     sequence_length=8
# )

# evaluation = evaluator.evaluate__xai_method(
#     test_accuracy, x_test, y_true, x_train, y_train, y_test,
#     model, explainer, verification_method='inverse_sequence',batch_size=2,
#     sequence_length=8
# )

# evaluation = evaluator.evaluate__xai_method(
#     test_accuracy, x_test, y_true, x_train, y_train, y_test,
#     model, explainer, verification_method='mean_sequence',batch_size=2,
#     sequence_length=8
# )

print()
print('Evaluation of explanation method per instance')
print(evaluation)

# print()
# print('Time series to explain')
# print(timeseries_instance.shape)
# print(timeseries_instance)


# print()
# print('Explained_ts')
# print(explained_ts.shape)
# print(explained_ts)



# exp = explained_ts.as_list(label=1)
# print(exp)

print()
print()
# print('Explanaaaaaaaaations')
# print(explained_ts.shape)
# print(explained_ts[0])
# print(timeseries_instance)
