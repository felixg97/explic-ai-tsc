import os
import numpy as np
import sys
import sklearn
import utils

from utils.constants import ARCHIVE_NAMES
from utils.constants import CLASSIFIERS

################################### METHODS ##################################

def fit_classifier(classifier_name):
    pass

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    pass

#################################### MAIN ####################################
"""
commands: run, 

-data, -d {dataset: default=all, specific}
-classifier, -c {classifier: default=all, specific}
-iterations, -i {iterations: default=?, specific}
-explanation, -ex {explanation: default=all, specific}
-evaluation, -ev {evaluation: default=all, specific}
-output_path, -o {output_path: string}

--test
--load (default)
--verbose
--build
--clear (clears logs and backends)
"""

root_directory = 'THIS NEEDS TO BE SPECIFIED'

test_sysargs = '--run all --fuck me --suck this'

sys_args_1 = ''
sys_args_2 = ''

## sys.argv management
if sys.argv[1] == 'run':
    pass
elif sys.argv[1] == '':
    pass
elif sys.argv[1] == '':
    pass
elif sys.argv[1] == '':
    pass



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


print('Hello my man')
