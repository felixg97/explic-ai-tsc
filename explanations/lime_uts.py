"""
lime_uts.py

LIME_TIMESERIES 
"""

### How Lime works:
# (1) let x element of R^d be the original representation of an instance being
#     explained
#
# (2) let x´ element of {0, 1}^d´ be a binary vector for its interpretable 
#     representation
#
# (3) Let g element of G be a explanation, where G is a class of interpretable
#     models. The domain of g is {0, 1}^d´
#
# (4) Let O(g) be a complexity measure of g element of G
#
# (5) Let the model being explained be denoted as f: R^d -> R, where f(x) is the
#     probability of x belonging to a certain class
#
# (6) Let pi[x](z) be a proximity measure between an instance z to x, so as to define
#     locality around x
#
# (7) Finally let L(f, g, pi[x]) be a measure of how unfaithful g is approximating
#     f in the locality defined by pi[x]
#
#
# In order to ensure both interpretability and local fidelity, we must minimize
# L(f, g, pi[x]) while having O(g) be low enough to be interpretable by humans
#
# The explanation produced by LIME is obtained by the following:
#   Eq.1: e(x)=argmin [g element G](L(f, g, pi[x]) + O(g))
#
#
# This can be used with different explanation classes G, fidelity function L,
# and complexity measures O
#
# G =
# L=
# O=
#
## Sampling for Local Exploraiton
# 
# (8) We want to minimize the locality-aware loss L(f, g, pi[x]) without making any
#     assumptions about f, since we want the explainer to be model-agnostic
#
# (9) Thus, in order to learn the local behavior of f as the interpretable inputs vary,
#     we approximate L(f, g, pi[x]) by drawing samples, weighted by pi[x]
#
# We sample instances around x´ by drawing nonzero elements of x´ uniformly at random
# (Where the number of such draws is also uniformly sampled)
#
# Given a perturbed sample z´ element of {0, 1}^d´ (which contains a fraction of the nonzero)
# elements of x´, we recover the sample in the original representation z element of R^d
# and obtain f(z) 
#
# Given this dataset Z of perturbed samples with the associated labels, we optimize
# Eq. 1 to get an explanation e(x).
#
# The goal is to sample instances both in the vicinity of x (high weight due to pi[x]) 
# and far away for x (low weight from pi[x])
# Event though f might be too complex to explain globaly LIME is locally faithful
#
## Sparse Linear Explanations
# 
# For the rest we let G be the class of linear models, such that g(z´)=w[g]+z´
#
# We use the locally weighted square loss as L is defined as:
#   Eq.2: L(f, g, pi[x])=Sum[z, z´ element of Z]( pi[x](z) * (f(z) - g(z))^2 )
# 
# where we let pi[x](z)=exp(-D(x, z)^2 / sig^2) be an exponential kernel defined on
# some distance function D (e.g. cosine distance for text, L2 distance for image)
# with width sig
#
# (!) se above: which distance measure to use for LIME_ts? (Eucledian =L2, DTW)
# 
# (10) Let K the number of non-zero coefficients as regularization term
#
# (11) Let O(g)=infinite 1 [||w[g]|| > K] (-> ?)
#
# Ribeiro et al. used K as constant (otherwise hyperparamter)(for text)
#
# K gets selected by LASSO regulariztaion path and then learning weights via 
# least squares (K-LASSO algorithm)
#
#
##################################### Specifics #################################
#
# e(x)=argmin [g element G](L(f, g, pi[x]) + O(g))
#
# Where:
#  L: L(f, g, pi[x])=Sum[z, z´ element of Z]( pi[x](z) * (f(z) - g(z))^2 ) (locally weighted square loss)
#  G: 
#  O: Complexity measure
#
#
# pi[x](z)=exp(-D(x, z)^2 / sig^2)
# 
# Where
#  (1) D: L2 distance  
#  (2) D: DTW distance

import math

from functools import partial
from os import terminal_size
import numpy as np
import scipy as sp
from scipy.sparse.construct import rand
import sklearn
from sklearn.linear_model import Ridge 
from sklearn.linear_model import lars_path
from sklearn.utils import check_random_state

from lime import lime_base
from lime import explanation

from utils.perturbations import UTSPerturbations

class UTSDomainMapper(explanation.DomainMapper):
    def __init__(self, patch_size):
        """Init function.
        Args:
            signal_names: list of strings, names of signals
        """
        self.patch_size = patch_size
        
    def map_exp_ids(self, exp, **kwargs):
        return exp


class LimeUTS(object):
    """Explains predictions on Time Series data."""


    def __init__(self, 
        kernel_width=25, 
        kernel=None, 
        verbose=False, 
        random_state=None,
        feature_selection='auto',
        signal_names=["not specified"]):
        """Init function
        
        Args:
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)    
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose=verbose, random_state=self.random_state)


    def explain_instance(self, 
        timeseries_instance,
        true_class=1,
        patch_size=1,
        labels=(1,), # TODO: testen ob das damit geht -> true class
        top_labels=None,
        model=None, 
        num_features=10, 
        num_samples=500, 
        random_seed=None, 
        perturbation='occlusion',
        distance_metric='eucledian',
        model_regressor=None,
        ):
        """Generates explanations for a prediction.

        Generates neighborhood by randomly perturbing features from the instance.
        It then learns locally weighted linear models on this neighborhood data
        to explain each of the class.

        Args:
            timeseries_instance: Time series to be explained.
            true_class: Class to be explained
            model: Classifier 
        """
        if not model:
            raise Exception('No model given.')

        permutations, predictions, distances = self._data_labels_distance(
            timeseries_instance, model, num_samples, patch_size=patch_size, 
            perturbation=perturbation, distance_metric=distance_metric)

        # if self.class_names is None:
        #     self.
        class_names = [str(x) for x in range(predictions[0].shape[0])]

        domain_mapper = UTSDomainMapper(patch_size)

        ret_exp = explanation.Explanation(domain_mapper=domain_mapper, 
            class_names=class_names)

        ret_exp.predict_proba = predictions[0]

        # print(top_labels)

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels[0]:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[int(label)],
                ret_exp.local_exp[int(label)],
                ret_exp.score,
                ret_exp.local_pred) = self.base.explain_instance_with_data(
                permutations, predictions, distances, label, num_features,
                model_regressor=model_regressor, feature_selection=self.feature_selection)
        return ret_exp


    def _data_labels_distance(
        self, 
        timeseries_instance,
        model,
        num_samples,
        patch_size=1,
        perturbation='occlusion',
        distance_metric='eucledian'
        ):
        """Generates time series and predictions in the neighborhood of this time series.
        """
        pass

        ### distance measures

        ## cosine 
        def distance_cosine(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='cosine').ravel() * 100

        ## eucedian
        def distance_eucledian(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='eucledian').ravel() * 100

        num_channels = 1 # cause univariate
        len_ts = len(timeseries_instance)
        num_slices = math.ceil(len_ts / patch_size)
        values_per_slice = patch_size

        deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries_instance.copy()]

        perturbator = UTSPerturbations()

        for i, num_inactive in enumerate(deact_per_sample, start=1):
            # print(f'Sample {i}, inactivating {num_inactive}')

            # choose random slices to perturb
            inactive_idxs = np.random.choice(features_range, num_inactive,
                replace=False)

            num_channels_to_perturb = np.random.randint(1, num_channels+1)

            channels_to_perturb = np.random.choice(range(num_channels),
                num_channels_to_perturb, replace=False)

            # print(f'Sample {i}, perturbung {channels_to_perturb}')

            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0

            tmp_series = timeseries_instance.copy()

            for idx in inactive_idxs:
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len_ts)

                perturbator.apply_perturbation(tmp_series, start_idx, end_idx, 
                    perturbation=perturbation)
            original_data.append(tmp_series)

        predictions = model.predict_input(np.array(original_data))

        # print()
        # print()
        # print('######## Predictions: ########')
        # print(predictions.shape)
        # print(predictions)
        # print()
        # print()

        perturbation_matrix = perturbation_matrix.reshape((num_samples, num_channels * num_slices))

        distances = None
        if distance_metric == 'cosine':
            distances = distance_cosine(perturbation_matrix)
        elif distance_metric == 'eucledian':
            distances = distance_cosine(perturbation_matrix)

        return perturbation_matrix, predictions, distances



