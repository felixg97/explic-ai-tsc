import os
import numpy as np
import sys
import sklearn
import utils

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
        print(f'Data set chosen:\t {datasets}')
    elif ARGUMENTS[1] in sys_argv:
        datasets = [sys_argv[(sys_argv.index(ARGUMENTS[1]) + 1)]]
        print(f'Data set:\t {datasets}')
    if ARGUMENTS[2] in sys_argv:
        classifiers = [sys_argv[(sys_argv.index(ARGUMENTS[2]) + 1)]]
        print(f'Classifier chosen:\t {classifiers}')
    elif ARGUMENTS[3] in sys_argv:
        classifiers = [sys_argv[(sys_argv.index(ARGUMENTS[3]) + 1)]]
        print(f'Classifier chosen:\t {classifiers}')
    if ARGUMENTS[4] in sys_argv:
        iterations = [sys_argv.index(ARGUMENTS[4] + 1)]
        print(f'Iterations chosen:\t {iterations}')
    elif ARGUMENTS[5] in sys_argv:
        iterations = [sys_argv.index(ARGUMENTS[5] + 1)]
        print(f'Iterations chosen:\t {iterations}')
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
    if len(sys_argv) == 1:
        pass
    if '--verbose' in sys_argv:
        verbose = True
    return verbose, load


def fit_classifier(output_directory, dataset_dict, dataset_name, classifier_name, verbose=False, load_weights=False):
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
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, load_weights=False):
    ## Deep
    classifier_name = classifier_name.lower()
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    elif classifier_name == 'mlp':
        from classifiers import ResNet
        return ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    elif classifier_name == 'inceptiontime':
        from classifiers import ResNet
        return ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    elif classifier_name == 'fcn':
        from classifiers import FCN
        return ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    ## Ensemble
    elif classifier_name == 'cote':
        from classifiers import ResNet
        return ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    elif classifier_name == 'hivecote':
        from classifiers import ResNet
        return ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)
    ## Linear
    elif classifier_name == 'rocket':
        from classifiers import ResNet
        return ResNet(output_directory, input_shape, nb_classes, verbose=verbose, load_weights=load_weights)

def run_complete(classifiers, iterations, datasets, explanations, evaluations, verbose=False, load=False):
    pass

def run_classifiers(root_dir, classifiers, iterations, datasets, verbose=False, load=False):

    dataset_dict = read_all_datasets(root_dir, 'UCRArchive_2018')

    print(dataset_dict.keys())

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

                continue

                fit_classifier(output_directory, dataset_dict, dataset_name, classifier_name, verbose=verbose, load=load)

                print('DONE')
                create_directory(output_directory + '/DONE')


def run_explanations(classifiers, iterations, datasets, explanations, verbose=False, load=False):

    for classifier_name in classifiers:
        pass


def run_evaluations(classifiers, iterations, datasets, explanations, evaluations, verbose=False, load=False):
    pass


##################################### MAIN #####################################

## For Notebook
root_directory = 'C:/git/explic-ai-tsc'

## For PC
# root_directory = 'D:/git/explic-ai-tsc'

sys_argv = sys.argv[1:]

command = sys_argv[0]

if command not in COMMANDS: 
    print('WRONG COMMANDS')
    print(HELP_INFO)
    exit()

datasets, classifiers, iterations, explanations, evaluations, out_path = determine_arguments(sys_argv)
verbose, load = determine_configurations(sys_argv)

print('-----------------------------------------------------------------------')
print('Configuration:')
print()
print('Data sets: \t', datasets)
print('Classifier: \t',classifiers) 
print('Iterations: \t',iterations)
print('Explanations: \t',explanations)
print('Evaluations: \t',evaluations) 
print('Verbose: \t',verbose) 
print('Load: \t\t',load) 
print('-----------------------------------------------------------------------')

## sys.argv management
if command == 'help':
    print(HELP_INFO)
if command == 'run_complete':
    run_complete(root_directory, classifiers, iterations, datasets, explanations, evaluations, verbose=verbose, load=load)
elif command == 'run_classifier':
    run_classifiers(root_directory, classifiers, iterations, datasets, verbose=verbose, load=load)
elif command == 'run_explanations':
    run_explanations(root_directory, classifiers, iterations, datasets, explanations, verbose=verbose, load=load)
elif command == 'run_evaluation':
    run_evaluations(root_directory, classifiers, iterations, datasets, explanations, evaluations, verbose=verbose, load=load)




# for classifier in classifiers:
#     for dataset in datasets:
#         pass



### RUN:
##### 1) Read all datasets D := {d1, ..., di} (search for a good example for visualization)
##### 2) Train all models M := {m1, ..., mn} per Dataset di

#### Run option 1 (most likely version)
##### 3) Classify all Time Series per di per mn
##### 4) Time Series Relevance: for each Perturbator P := {p1, ..., pm}
##### 4.1) For each Time Series Td := {t1, ..., tj} Relevance Vector r(tj, mn, p1)

#### Run option 2 ("getting more and more unlikely" version)
##### 3) Classify all Time Series per di per mn
##### 4) Time Series Relevance: for each Feature Perturbator FP := {fp1, ..., fpm}
##### 4.1) For each Time Series Td := {t1, ..., tj} Relevance Vector r(tj, mn, p1)

