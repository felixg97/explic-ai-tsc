import os

import numpy as np
import pandas as pd

from pathlib import Path
from utils.utils import create_directory


## Constants
from utils.constants import ARCHIVE_NAMES
from utils.constants import DATASETS_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import EXPLANATIONS
from utils.constants import ITERATIONS


root_directory = 'D:/git/explic-ai-tsc'

print()
print('-----------------------------------------------------------------------')
print('Configurations:')
print()
print('Data sets: \t', DATASETS_NAMES)
print('Classifier: \t', CLASSIFIERS) 
print('Classifier: \t', EXPLANATIONS) 
print('Iterations: \t', ITERATIONS)
print('-----------------------------------------------------------------------')
print()

# Occlusion
occlusion = []

# LIME
lime = []

# RISE
rise = []


for classifier_name in CLASSIFIERS:
    print('Classifier:\t', classifier_name)
    pass

    for iteration in range(ITERATIONS):
        print('Iteration:\t', iteration)
        pass

        for dataset_name in DATASETS_NAMES:
            print('Data set:\t', dataset_name)
            pass

            print()
            results_dir = f'{root_directory}/results/{classifier_name}/UCRArchive_2018_itr_{iteration}/{dataset_name}'
            evaluations_dir = results_dir + '/evaluations'
            

            list = os.listdir(evaluations_dir)
            
            base_acc = float(pd.read_csv(results_dir + '/_df_best_model.csv', sep=',', header=None).to_numpy()[1][3])
            # print(base_acc)
            # print(type(base_acc))

            occlusion_files, lime_files, rise_files = [], [], []
            for file in list:
                if (file.find('occlusion') != -1):
                    print(file)
                    occlusion_files.append(file)
                elif (file.find('lime') != -1):
                    lime_files.append(file)
                elif (file.find('rise') != -1):
                    rise_files.append(file)

            evaluation_thresholds = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]

            # for file in occlusion_files:
            #     print(file)

            # break

            if 'Occlusion' in EXPLANATIONS:
                print()
                print('--- Occlusion ---')
                # Result: classifier, dataset, perturbation, patch_size, threshold, acc_base, acc_zero_tp, change_zero_tp, acc_inve_tp, change_inve_tp, acc_mean_tp, change_mean_tp
                
                for file in occlusion_files:
                    print(evaluations_dir + '/' + file)
                    csv_file = pd.read_csv(evaluations_dir + '/' + file, sep=',', header=None).to_numpy()
                    # split = file.split('_')
                    _1, perturbation, _2, patch_size, _3 = file.split('_') 
                    # print(csv_file[0])
                    for line in csv_file[1:]:
                        _l1, _l2, _l3, e, _l5, _l6, acc_zero_tp, acc_inve_tp, acc_mean_tp, _l10, _l11, _l12, _l13= line
                        res = [
                            classifier_name, 
                            dataset_name, 
                            perturbation, 
                            patch_size,
                            e,
                            base_acc,
                            float(acc_zero_tp),
                            base_acc - float(acc_zero_tp),
                            float(acc_inve_tp),
                            base_acc - float(acc_inve_tp),
                            float(acc_mean_tp),
                            base_acc - float(acc_mean_tp)
                        ]
                        occlusion.append(res)

            if 'LIME' in EXPLANATIONS:
                print()
                print('--- LIME ---')
                # Result: classifier, dataset, perturbation, distance_metric, patch_size, threshold, acc_base, acc_zero_tp, change_zero_tp, acc_inve_tp, change_inve_tp, acc_mean_tp, change_mean_tp
                for file in lime_files:
                    print(evaluations_dir + '/' + file)
                    csv_file = pd.read_csv(evaluations_dir + '/' + file, sep=',', header=None).to_numpy()
                    # print(file.split('_'))
                    _1, _2, distance_metric, perturbation, _5, patch_size, _7 = file.split('_')
                    
                    if distance_metric == 'cosine': continue
                    # print(csv_file[0])
                    for line in csv_file[1:]:
                        _l1, _l2, _l3, e, _l5, _l6, acc_zero_tp, acc_inve_tp, acc_mean_tp, _l10, _l11, _l12, _l13= line
                        res = [
                            classifier_name, 
                            dataset_name, 
                            perturbation, 
                            distance_metric,
                            patch_size,
                            e,
                            base_acc,
                            float(acc_zero_tp),
                            base_acc - float(acc_zero_tp),
                            float(acc_inve_tp),
                            base_acc - float(acc_inve_tp),
                            float(acc_mean_tp),
                            base_acc - float(acc_mean_tp)
                        ]
                        lime.append(res)


            if 'RISE' in EXPLANATIONS:
                print()
                print('--- RISE ---')
                print(evaluations_dir + '/' + file)
                # Result: classifier, dataset, interpolation, batch_size, threshold, acc_base, acc_zero_tp, change_zero_tp, acc_inve_tp, change_inve_tp, acc_mean_tp, change_mean_tp
                for file in rise_files:
                    csv_file = pd.read_csv(evaluations_dir + '/' + file, sep=',', header=None).to_numpy()
                    # print(file.split('_'))
                    _1, interpolation, _3, batch_size, _5 = file.split('_')
                    # print(csv_file[0])
                    for line in csv_file[1:]:
                        _l1, _l2, _l3, e, _l5, _l6, acc_zero_tp, acc_inve_tp, acc_mean_tp, _l10, _l11, _l12, _l13= line
                        res = [
                            classifier_name, 
                            dataset_name, 
                            interpolation,
                            batch_size,
                            e,
                            base_acc,
                            float(acc_zero_tp),
                            base_acc - float(acc_zero_tp),
                            float(acc_inve_tp),
                            base_acc - float(acc_inve_tp),
                            float(acc_mean_tp),
                            base_acc - float(acc_mean_tp)
                        ]
                        rise.append(res)
print()
print('--- Finished ---')
print()
print()
################################################################################

my_dir = root_directory + '/results'

cols_occlusion = ['classifier', 'dataset', 'perturbation', 'patch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inverse_tp', 'change_inverse_tp', 'acc_mean_tp', 'change_mean_tp']
cols_lime = ['classifier', 'dataset', 'perturbation', 'distance_metric', 'patch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inve_tp', 'change_inve_tp', 'acc_mean_tp', 'change_mean_tp']
cols_rise = ['classifier', 'dataset', 'interpolation', 'batch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inve_tp', 'change_inve_tp', 'acc_mean_tp', 'change_mean_tp']


### Occlusion save ###
np_occlusion = np.array(occlusion)
print(np_occlusion.shape)
# print(len(cols_occlusion))
df_occlusion = pd.DataFrame(data=np_occlusion, columns=cols_occlusion)

print(df_occlusion.head())
print()
### LIME save ###
np_lime = np.array(lime)
print(np_lime.shape)
# print(len(cols_lime))
df_lime = pd.DataFrame(data=np_lime, columns=cols_lime)

print(df_lime.head())
print()
### RISE save ###
np_rise = np.array(rise)
print(np_rise.shape)
# print(len(cols_rise))
df_rise = pd.DataFrame(data=np_rise, columns=cols_rise)

print(df_rise.head())

occlusion_rfile = my_dir + '/occlusion_results.csv'
lime_rfile = my_dir + '/lime_results.csv'
rise_rfile = my_dir + '/rise_results.csv'


print()
print('--------------------------------')
print()
df_occlusion.to_csv(occlusion_rfile, index=False)
print('written', occlusion_rfile)
df_lime.to_csv(lime_rfile, index=False)
print('written', lime_rfile)
df_rise.to_csv(rise_rfile, index=False)
print('written', rise_rfile)
print()
print('--------------------------------')
print()

print()
print('--------------------------------')
print()
print()
print('No. of Results: ', np_occlusion.shape[0] + np_lime.shape[0] + np_rise.shape[0])
print()
print()
print('--------------------------------')