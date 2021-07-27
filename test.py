import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import tensorflow as tf

from utils.utils import read_all_datasets
from explanations import OcclusionSensitivityUTS

from classifier import ShallowMLP

################################ Load Data set #################################
## For Notebook
root_directory = 'C:/git/explic-ai-tsc'

## For PC
# root_directory = 'D:/git/explic-ai-tsc'

dataset_dict = read_all_datasets(root_directory, 'UCRArchive_2018')

curr_dataset = 'BeetleFly'

x_train, y_train, x_test, y_test = dataset_dict[curr_dataset]

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

# print(x_train.shape)
# print(y_train.shape)

# timeseries = x_train[0]
timeseries = np.array(range(20))

############################ Load Pretrained Model #############################
model = ShallowMLP()


################################## Occlusion ###################################
explainer = OcclusionSensitivityUTS()


#################################### RISE ######################################


################################### Anchor #####################################


########################### Meaningful Perturbation ############################




explained_ts = explainer.explain_instance(timeseries, 1, tf.keras.Model, 4)