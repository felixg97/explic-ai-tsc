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

# Random
random = []

pd.set_option('display.max_colwidth', None)
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

            occlusion_files, lime_files, rise_files, random_files = [], [], [], []
            for file in list:
                if (file.find('occlusion') != -1):
                    # print(file)
                    occlusion_files.append(file)
                elif (file.find('lime') != -1):
                    lime_files.append(file)
                elif (file.find('rise') != -1):
                    rise_files.append(file)
                elif (file.find('random') != -1):
                    random_files.append(file)

            evaluation_thresholds = [
                # 95, 
                90, 
                # 85, 
                # 80, 
                # 75, 
                # 70, 
                # 65, 
                # 60, 
                # 55, 
                # 50
            ]

            # for file in occlusion_files:
            #     print(file)

            # break

            if 'Occlusion' in EXPLANATIONS:
                print()
                print('--- Occlusion ---')
                # Result: classifier, dataset, perturbation, patch_size, threshold, acc_base, acc_zero_tp, change_zero_tp, acc_inve_tp, change_inve_tp, acc_mean_tp, change_mean_tp
                
                for file in occlusion_files:
                    print()
                    print(evaluations_dir + '/' + file)
                    csv_file = pd.read_csv(evaluations_dir + '/' + file, sep=',', header=None)
                    print(csv_file)
                    csv_file = csv_file.to_numpy()
                    exp, ps_s, patch_size, perturbation, ending = file.split('_') 
                    # print(csv_file[0])
                    for line in csv_file[1:]:
                        nan, verification, qm, e, sl, acc_zero_tp, acc_inve_tp, base_acc, duration = line
                        e = int(e)
                        if e not in evaluation_thresholds: continue
                        base_acc = float(base_acc)
                        acc_zero_tp = float(acc_zero_tp)
                        acc_inve_tp = float(acc_inve_tp)
                        res = [
                            classifier_name, 
                            dataset_name, 
                            perturbation, 
                            patch_size,
                            e,
                            base_acc,
                            acc_zero_tp,
                            base_acc - acc_zero_tp,
                            acc_inve_tp,
                            base_acc - acc_inve_tp
                        ]
                        occlusion.append(res)
            
            # break
            if 'LIME' in EXPLANATIONS:
                print()
                print('--- LIME ---')
                # Result: classifier, dataset, perturbation, distance_metric, patch_size, threshold, acc_base, acc_zero_tp, change_zero_tp, acc_inve_tp, change_inve_tp, acc_mean_tp, change_mean_tp
                for file in lime_files:
                    print(evaluations_dir + '/' + file)
                    csv_file = pd.read_csv(evaluations_dir + '/' + file, sep=',', header=None).to_numpy()
                    # print(file.split('_'))
                    exp, ps_s, patch_size, perturbation, distance_metric, ending = file.split('_')
                    
                    if distance_metric == 'cosine': continue
                    # print(csv_file[0])
                    for line in csv_file[1:]:
                        nan, verification, qm, e, sl, acc_zero_tp, acc_inve_tp, base_acc, duration = line
                        e = int(e)
                        if e not in evaluation_thresholds: continue
                        base_acc = float(base_acc)
                        acc_zero_tp = float(acc_zero_tp)
                        acc_inve_tp = float(acc_inve_tp)
                        res = [
                            classifier_name, 
                            dataset_name, 
                            perturbation, 
                            distance_metric,
                            patch_size,
                            e,
                            base_acc,
                            acc_zero_tp,
                            base_acc - acc_zero_tp,
                            acc_inve_tp,
                            base_acc - acc_inve_tp
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
                    exp, ps_s, patch_size, interpolation, ending = file.split('_')
                    # print(csv_file[0])
                    for line in csv_file[1:]:
                        nan, verification, qm, e, sl, acc_zero_tp, acc_inve_tp, base_acc, duration = line
                        e = int(e)
                        if e not in evaluation_thresholds: continue
                        base_acc = float(base_acc)
                        acc_zero_tp = float(acc_zero_tp)
                        acc_inve_tp = float(acc_inve_tp)
                        res = [
                            classifier_name, 
                            dataset_name, 
                            interpolation,
                            patch_size,
                            e,
                            base_acc,
                            acc_zero_tp,
                            base_acc - acc_zero_tp,
                            acc_inve_tp,
                            base_acc - acc_inve_tp
                        ]
                        rise.append(res)

            print()
            print('--- RANDOM ---')
            # Result: classifier, dataset, patch_size, threshold, acc_base, acc_zero_tp, change_zero_tp, acc_inve_tp, change_inve_tp, acc_mean_tp, change_mean_tp
            for file in random_files:
                csv_file = pd.read_csv(evaluations_dir + '/' + file, sep=',', header=None).to_numpy()
                print(file.split('_'))
                exp, ps_s, patch_size, ending = file.split('_')
                # print(csv_file[0])
                for line in csv_file[1:]:
                    nan, verification, qm, e, sl, acc_zero_tp, acc_inve_tp, base_acc, duration = line
                    e = int(e)
                    if e not in evaluation_thresholds: continue
                    base_acc = float(base_acc)
                    acc_zero_tp = float(acc_zero_tp)
                    acc_inve_tp = float(acc_inve_tp)
                    res = [
                        classifier_name, 
                        dataset_name, 
                        patch_size,
                        e,
                        base_acc,
                        acc_zero_tp,
                        base_acc - acc_zero_tp,
                        acc_inve_tp,
                        base_acc - acc_inve_tp
                    ]
                    random.append(res)

print()
print('--- Finished ---')
print()
print()
################################################################################

my_dir = root_directory + '/results'

cols_occlusion = ['classifier', 'dataset', 'perturbation', 'patch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inverse_tp', 'change_inverse_tp']
cols_lime = ['classifier', 'dataset', 'perturbation', 'distance_metric', 'patch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inve_tp', 'change_inve_tp']
cols_rise = ['classifier', 'dataset', 'interpolation', 'batch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inve_tp', 'change_inve_tp'] # batch_size == patch_size  in this array
cols_random = ['classifier', 'dataset', 'patch_size', 'threshold', 'acc_base', 'acc_zero_tp', 'change_zero_tp', 'acc_inve_tp', 'change_inve_tp']


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
print()
### Random save ###
np_random = np.array(random)
print(np_random.shape)
# print(len(cols_rise))
df_random = pd.DataFrame(data=np_random, columns=cols_random)

print(df_random.head())

occlusion_rfile = my_dir + '/occlusion_results.csv'
lime_rfile = my_dir + '/lime_results.csv'
rise_rfile = my_dir + '/rise_results.csv'
random_rfile = my_dir + '/random_results.csv'


print()
print('--------------------------------')
print()
df_occlusion.to_csv(occlusion_rfile, index=False)
print('written', occlusion_rfile)
df_lime.to_csv(lime_rfile, index=False)
print('written', lime_rfile)
df_rise.to_csv(rise_rfile, index=False)
print('written', rise_rfile)
df_random.to_csv(random_rfile, index=False)
print('written', random_rfile)
print()
print('--------------------------------')
print()

print()
print('--------------------------------')
print()
print()
print('No. of Results: ', np_occlusion.shape[0] + np_lime.shape[0] + np_rise.shape[0] + np_random.shape[0])
print()
print('W. r. t. threshold ', (np_occlusion.shape[0] + np_lime.shape[0] + np_rise.shape[0] + np_random.shape[0]) / 10, 'experiments')
print()
print('--------------------------------')