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
################################ Explained other way ###########################
#
# 1) Generate N perturbed samples of the interpretable version of the instance to
#    explain y´. 
#    Let {z[i]´ element of X´ | i=1, ..., N} be the set of observations
#
# 2) Recover the observed observations in the original feature space by means of 
#    the mapping of the mapping function. 
#    Let {z[i] be equivalent to h^y(z[i]´) element of X | i=1, ..., N} be the set
#    in the original representation
# 
#
#
##################################### Specifics #################################
#
# e(x)=argmin [g element G](L(f, g, pi[x]) + O(g))
#
# Where:
#  L: L(f, g, pi[x])=Sum[z, z´ element of Z]( pi[x](z) * (f(z) - g(z))^2 ) (locally weighted square loss)
#  G: ?
#  O: 
#
#
# pi[x](z)=exp(-D(x, z)^2 / sig^2)
# 
# Where
#  (1) D: L2 distance  
#  (2) D: DTW distance


from functools import partial
from os import terminal_size
import numpy as np
import scipy as sp
from scipy.sparse.construct import rand
from sklearn.linear_model import Ridge 
from sklearn.linear_model import lars_path
from sklearn.utils import check_random_state

from lime import lime_base
from lime import explanation


class UnivariateTimeSeriesDomainMapper():
    def __init__(self, signal_name, num_slices):
        """Init function
        
        """


class LimeUnivariateTimeSeriesExplainer(object):
    """Explains predictions on Time Series data."""


    def __init__(self, kernel_width=.25, 
        kernel=None, 
        verbose=False, 
        feature_selection='auto',
        random_state=None):
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
        self.base = lime_base.LimeBase(kernel_fn, verbose=verbose)


    def explain_instance(self, 
        timeseries_instance, 
        classifier_fn, 
        labels=(1,),
        top_labels=None, 
        num_features=100, 
        num_samples=1000, 
        batch_size=10,
        segmentation_fn=None, 
        random_seed=None, 
        progress_bar=True,
        perturbation_fn='occlusion',
        distance_fn='L2'):
        """Generates explanations for a prediction.

        Generates neighborhood by randomly perturbing features from the instance.
        It then learns locally weighted linear models on this neighborhood data
        to explain each of the class.

        """
        
        if perturbation_fn == 'occlusion':
            pass
        elif perturbation_fn == 'mean':
            pass
        
        if 

        pass

    def data_labels(self, 
        timeseries, 
        ):
        """Generates time series and predictions in the neighborhood of this time series.
        """
        pass

