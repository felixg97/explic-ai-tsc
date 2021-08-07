import os
from time import time
import numpy as np
import sys
import sklearn
import utils

import math

from utils.utils import read_all_datasets
from utils.utils import create_directory

## Constants
from utils.constants import HELP_INFO
from utils.constants import UNIVARIATE_DATASET_NAMES
from utils.constants import DATASETS_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import EXPLANATIONS
from utils.constants import EVALUATIONS
from utils.constants import COMMANDS
from utils.constants import ARGUMENTS
from utils.constants import PARAMS
from utils.constants import CONFIGS
from utils.constants import ITERATIONS

# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

datasets = None
classifiers = None
iterations = None
explanations = None
evaluations = None

output_path =''
verbose = False


#################################### METHODS ###################################

def determine_arguments(sys_argv):
    print('-----------------------------------------------------------------------')
    datasets = DATASETS_NAMES
    classifiers = CLASSIFIERS
    iterations = ITERATIONS # TODO: make it work w/ utils.constants
    explanations = EXPLANATIONS
    evaluations = EVALUATIONS
    out_path = ''

    if len(sys_argv) == 1:
        print(f'Default settings will be used.')
    if ARGUMENTS[0] in sys_argv:
        datasets = [sys_argv[(sys_argv.index(ARGUMENTS[0]) + 1)]]
        print(f'Data set chosen:\t\t {datasets}')
    elif ARGUMENTS[1] in sys_argv:
        datasets = [sys_argv[(sys_argv.index(ARGUMENTS[1]) + 1)]]
        print(f'Data set:\t\t {datasets}')
    if ARGUMENTS[2] in sys_argv:
        classifiers = [sys_argv[(sys_argv.index(ARGUMENTS[2]) + 1)]]
        print(f'Classifier chosen:\t {classifiers}')
    elif ARGUMENTS[3] in sys_argv:
        classifiers = [sys_argv[(sys_argv.index(ARGUMENTS[3]) + 1)]]
        print(f'Classifier chosen:\t {classifiers}')
    if ARGUMENTS[4] in sys_argv:
        iterations = [sys_argv[(sys_argv.index(ARGUMENTS[4]) + 1)]]
        print(f'Iterations chosen:\t {iterations}')
    elif ARGUMENTS[5] in sys_argv:
        iterations = [sys_argv[(sys_argv.index(ARGUMENTS[5]) + 1)]]
        print(f'Iterations chosen:\t {iterations}')
        pass
    if ARGUMENTS[6] in sys_argv:
        explanations = [sys_argv[(sys_argv.index(ARGUMENTS[6]) + 1)]]
        print(f'Eplanation chosen:\t {explanations}')
    elif ARGUMENTS[7] in sys_argv:
        explanations = [sys_argv[(sys_argv.index(ARGUMENTS[7]) + 1)]]
        print(f'Eplanation chosen:\t {explanations}')
    if ARGUMENTS[8] in sys_argv:
        evaluations = [sys_argv[(sys_argv.index(ARGUMENTS[8]) + 1)]]
        print(f'Evaluation chosen:\t {evaluations}')
    elif ARGUMENTS[9] in sys_argv:
        evaluations = [sys_argv[(sys_argv.index(ARGUMENTS[9]) + 1)]]
        print(f'Evaluation chosen:\t {evaluations}')
    if ARGUMENTS[10] in sys_argv:
        out_path = sys_argv.index(ARGUMENTS[10])
        print(f'Output_path chosen:\t {out_path}')
    elif ARGUMENTS[11] in sys_argv:
        out_path = sys_argv.index(ARGUMENTS[11])
        print(f'Output_path chosen:\t {out_path}')

    return datasets, classifiers, iterations, explanations, evaluations, out_path

def determine_configurations(sys_argv):
    verbose = False
    load = False
    rebuild = False
    generate_plots = False
    if len(sys_argv) == 1:
        pass
    if '--verbose' in sys_argv:
        verbose=True
    if '--load' in sys_argv:
        load=True
    if '--rebuild' in sys_argv:
        rebuild=True
    if '--generate_plots' in sys_argv:
        generate_plots=True
    return verbose, load, rebuild, generate_plots

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


def fit_classifier(output_directory, dataset_dict, dataset_name, classifier_name, verbose=False, build=False, load_weights=False):
    x_train, y_train, x_test, y_test = dataset_dict[dataset_name]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=verbose, build=build, load_weights=load_weights)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, load_weights=False, build=True):
    ## Deep
    print('hallo hallo', output_directory)
    classifier_name = classifier_name.lower()
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.ResNet(output_directory, input_shape, nb_classes, verbose=verbose, build=build, load_weights=load_weights)
    elif classifier_name == 'mlp':
        from classifiers import MLP
        return MLP(output_directory, input_shape, nb_classes, verbose=verbose, build=build, load_weights=load_weights)
    elif classifier_name == 'inceptiontime':
        from classifiers import InceptionTime
        return InceptionTime(output_directory, input_shape, nb_classes, verbose=verbose, build=build, load_weights=load_weights)
    elif classifier_name == 'fcn':
        from classifiers import FCN
        return FCN(output_directory, input_shape, nb_classes, verbose=verbose, build=build, load_weights=load_weights)
    ## Ensemble
    # elif classifier_name == 'cote':
    #     from classifiers import COTE
    #     return COTE(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    # elif classifier_name == 'hivecote':
    #     from classifiers import HIVECOTE
    #     return HIVECOTE(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)


def create_explanation(explanation_name):
    explanation_name = explanation_name.lower()
    if explanation_name == 'occlusion':
        from explanations import OcclusionSensitivityUTS
        return OcclusionSensitivityUTS()
    elif explanation_name == 'lime':
        from explanations import LimeTimeSeriesExplainer
        return LimeTimeSeriesExplainer()
    elif explanation_name == 'rise':
        from explanations import RiseUTS
        return RiseUTS()
    elif explanation_name == 'anchor':
        from explanations import AnchorUTS
        return AnchorUTS()
    elif explanation_name == 'meaningfulperturbation':
        # from explanations import MeaningfulPerturbationUTS
        # return MeaningfulPerturbationUTS()
        pass


def create_evaluation(evaluation_name):
    evaluation_name = evaluation_name.lower()
    if evaluation_name == 'perturbationanalysis':
        from evaluations import PerturbationAnalysisUTS
        return PerturbationAnalysisUTS
    elif evaluation_name == 'sanitycheck':
        return None

def run_complete(classifiers, iterations, datasets, explanations, evaluations, verbose=False, load=False):
    pass


def run_classifiers(root_dir, classifiers, iterations, datasets, verbose=False, load=False, generate_plots=False):

    dataset_dict = read_all_datasets(root_dir, 'UCRArchive_2018')

    # print(dataset_dict.keys())

    for classifier_name in classifiers:
        # print('Classifier:\t', classifier_name)

        for iteration in [*range(iterations)]:
            # print('Iteration:\t', iteration)

            trr = '_itr_' + str(iteration)

            toutput_directory = root_dir + '/results/' + classifier_name + '/' + 'UCRArchive_2018' + trr + '/'

            # print(output_directory)

            for dataset_name in datasets:
                # print('Data set:\t', dataset_name)

                output_directory = toutput_directory + dataset_name + '/'

                print(output_directory)

                create_directory(output_directory)

                fit_classifier(output_directory, dataset_dict, dataset_name, classifier_name, verbose=verbose, load_weights=load)

                print('DONE')
                create_directory(output_directory + '/DONE')


def run_explanations(root_dir, classifiers, iterations, datasets, explanations, build=False, verbose=False, load=False):

    dataset_dict = read_all_datasets(root_dir, 'UCRArchive_2018')

    for classifier_name in classifiers:
        print('Classifier:\t', classifier_name)

        print(iterations)
        for iteration in range(iterations):
            print('Iteration:\t', iteration)

            for dataset_name in datasets:
                print('Data set:\t', dataset_name)

                x_train, y_train, x_test, y_test, y_true, nb_classes, input_shape = shape_data(dataset_dict[dataset_name])

                output_directory = f'{root_dir}/results/{classifier_name}/UCRArchive_2018_itr_{iteration}/{dataset_name}/'
                classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, build=False)
                explanations_dir = output_directory + 'explanations'
                create_directory(explanations_dir)

                timeseries_len = round(x_train.shape[1] / 4)

                perturbations = ['zero', 'mean']

                #### Occlusion
                # print('--- occlusion ---')
                # explainer = create_explanation('Occlusion')
                # for perturbation in perturbations:
                #     print('Perturbation:\t', perturbation)
                #     patch_size_step = round(timeseries_len/20)
                #     for patch_size in range(patch_size_step, timeseries_len, patch_size_step):
                #         print('Patch_size:\t', patch_size)
                #         relevance = explainer.explain(x_test, y_true, classifier, patch_size=patch_size)
                #         np.savetxt(explanations_dir + f'/occlusion_{perturbation}_ps_{patch_size}.csv', relevance, delimiter=',')

                #### LIME
                print()
                print('--- lime ---')
                explainer = create_explanation('LIME')
                distance_metrics = ['dtw', 'euclidean'] #, 'cosine']
                for distance_metric in distance_metrics:
                    print('Distance_metric:\t', distance_metric)
                    for perturbation in perturbations:
                        print('Perturbation:\t', perturbation)
                        ## samples 
                        patch_size_step = round(timeseries_len/20)
                        for patch_size in range(patch_size_step, timeseries_len, patch_size_step):
                            print('Patch_size:\t', patch_size)
                            print()
                            relevance = explainer.explain(x_test, y_true, classifier, 
                                patch_size=patch_size, distance_metric=distance_metric)
                            np.savetxt(explanations_dir + f'/lime_{distance_metric}_{perturbation}_ps_{patch_size}.csv', relevance, delimiter=',')

                ## RISE TODO:
                # print()
                # print('--- rise ---')
                # explainer = create_explanation('RISE')
                # interpolations = ['linear', 'fourier']
                # for interpolation in interpolations:
                #     print('Interpolation:\t', interpolation)
                #     start_batch = 50 if x_test.shape[0] > 50 else 5
                #     end_batch = x_test.shape[0] if x_test.shape[0] > 50 else 20
                #     batch_step = 5 if x_test.shape[0] > 50 else 1
                #     for batch_size in range(start_batch, end_batch, batch_step):
                #         print('Batch_size:\t', batch_size)
                #         print()
                #         relevance = explainer.explain(x_test, y_true, classifier, batch_size=batch_size, interpolation=interpolation)
                #         np.savetxt(explanations_dir + f'/rise_{interpolation}_batchs_{batch_size}.csv', relevance, delimiter=',')


def run_evaluations(root_dir, classifiers, iterations, datasets, explanations, evaluations, build=False, verbose=False, load=False):

    dataset_dict = read_all_datasets(root_dir, 'UCRArchive_2018')

    for classifier_name in classifiers:
        print('Classifier:\t', classifier_name)

        for iteration in iterations:
            print('Iteration:\t', iteration)

            for dataset_name in datasets:
                print('Data set:\t', dataset_name)

                x_train, y_train, x_test, y_test, y_true, nb_classes, input_shape = shape_data(dataset_dict[dataset_name])

                #### Occlusion
                print('--- occlusion ---')


                ### LIME
                print()
                print('--- lime ---')


                ### RISE
                print()
                print('--- rise ---')


    pass


##################################### MAIN #####################################

## For Notebook
# root_directory = 'C:/git/explic-ai-tsc'
## For PC
root_directory = 'D:/git/explic-ai-tsc'

sys_argv = sys.argv[1:]

command = sys_argv[0]

if command not in COMMANDS: 
    print('WRONG COMMANDS')
    print(HELP_INFO)
    exit()

datasets, classifiers, iterations, explanations, evaluations, out_path = determine_arguments(sys_argv)
verbose, load, rebuild, generate_plots = determine_configurations(sys_argv)

print('-----------------------------------------------------------------------')
print('Configurations:')
print()
print('Data sets: \t', datasets)
print('Classifier: \t', classifiers) 
print('Iterations: \t', iterations)
print('Explanations: \t',explanations)
print('Evaluations: \t', evaluations) 
print('Verbose: \t', verbose) 
print('Load: \t\t', load) 
print('Rebuild classifier: \t\t', rebuild) 
print('Generate plots for explanation: \t\t', generate_plots) 
print('-----------------------------------------------------------------------')

## sys.argv management
if command == 'help':
    print(HELP_INFO)
if command == 'run_complete':
    run_complete(root_directory, classifiers, iterations, datasets, explanations, evaluations, verbose=verbose, load=load)
elif command == 'run_classifier':
    run_classifiers(root_directory, classifiers, iterations, datasets, verbose=verbose, load=load)
elif command == 'run_explanations':
    run_explanations(root_directory, classifiers, iterations, datasets, explanations, verbose=verbose, load=load, build=rebuild)
elif command == 'run_evaluation':
    run_evaluations(root_directory, classifiers, iterations, datasets, explanations, evaluations, verbose=verbose, load=load, build=rebuild)



# Tests
# cls && python main.py run_explanations -c MLP -d ECG5000 -ex Occlusion -i 1 -ev PerturbationAnalysis
#
#
#
#
#
#
#
#
#
#