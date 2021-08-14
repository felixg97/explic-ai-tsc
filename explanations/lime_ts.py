"""
https://github.com/emanuel-metzenthin/Lime-For-Time/blob/ma ster/lime_timeseries.py
"""
import numpy as np
import sklearn
from lime import explanation
from lime import lime_base
import math
import logging

from dtaidistance import dtw
from scipy.spatial import distance

class TSDomainMapper(explanation.DomainMapper):
    def __init__(self, signal_names, num_slices, is_multivariate):
        """Init function.
        Args:
            signal_names: list of strings, names of signals
        """
        self.num_slices = num_slices
        self.signal_names = signal_names
        self.is_multivariate = is_multivariate
        
    def map_exp_ids(self, exp, **kwargs):
        # in case of univariate, don't change feature ids

        # print(exp)
        if not self.is_multivariate:
            # print('is univariate')
            return exp


        
        names = []
        for _id, weight in exp:
            # from feature idx, extract both the pair number of slice
            # and the signal perturbed
            nsignal = int(_id / self.num_slices)
            nslice = _id % self.num_slices
            signalname = self.signal_names[nsignal]
            featurename = "%d - %s" % (nslice, signalname)
            names.append((featurename, weight))
        return names

class LimeTimeSeriesExplainer(object):
    """Explains time series classifiers."""

    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 signal_names=["not specified"]
                 ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
            signal_names: list of strings, names of signals
        """

        # exponential kernel
        def kernel(d): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.signal_names = signal_names

    def explain(self,
                timeseries_data,
                y_true,
                model,
                patch_size=1,
                labels=(0,),
                top_labels=None,
                num_features=None,
                num_samples=1000,
                model_regressor=None,
                perturbation='zero',
                distance_metric='euclidean'):
        _timeseries_data = np.array(timeseries_data)
        _y_true = np.array(y_true)
        
        if type(_timeseries_data.tolist())!=list:
            _timeseries_data = np.array([[timeseries_data]])
        if type(_y_true.tolist())!=list:
            _y_true = np.array([[y_true]])

        explanations = np.array([
            self.explain_instance(
                _timeseries_data[idx], 
                _y_true[idx], 
                model,
                patch_size=patch_size,
                labels=labels,
                top_labels=top_labels,
                num_features=num_features,
                num_samples=num_samples,
                perturbation=perturbation,
                distance_metric=distance_metric)
            for idx in range(_timeseries_data.shape[0]) 
        ])

        return explanations


    def explain_instance(self,
                        timeseries_instance,
                        y_true,
                        model,
                        patch_size=1,
                        labels=(0,),
                        top_labels=None,
                        num_features=None,
                        num_samples=5000,
                        model_regressor=None,
                        perturbation='zero',
                        lime_explanation=False,
                        distance_metric='euclidean'):
        """Generates explanations for a prediction.
        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).
        As distance function DTW metric is used.
        Args:
            time_series_instance: time series to be explained.
            classifier_fn: classifier prediction probability function,
                which takes a list of d arrays with time series values
                and outputs a (d, k) numpy array with prediction
                probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            num_slices: Defines into how many slices the time series will
                be split up
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
       """
        num_slices = math.ceil(len(timeseries_instance) / patch_size)

        _num_features = None
        if num_features == None:
            _num_features = num_slices

        permutations, predictions, distances = self.__data_labels_distances(
            timeseries_instance, model, y_true, num_samples, num_slices, 
            perturbation=perturbation, distance_metric=distance_metric)

        is_multivariate = len(timeseries_instance.shape) > 1
        
        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]

        domain_mapper = TSDomainMapper(self.signal_names, num_slices, is_multivariate)
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = predictions[0]

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[int(label)],
             ret_exp.local_exp[int(label)],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                permutations, predictions,
                distances, 
                label,
                _num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        # print()
        # print('len ts')
        # print(len(timeseries_instance))
        # print('patch')
        # print(patch_size)
        # print('num_slices')
        # print(num_slices)
        # print()
        if lime_explanation:
            return ret_exp

        ret_exp = ret_exp.as_list(label=y_true)
        ret_exp.sort(key=lambda tup:tup[0])
        ret_exp = np.array([list(tup) for tup in ret_exp])

        # print(ret_exp)
        # print(len(ret_exp))
        # print()
        relevant_slices = ret_exp[:,0]
        # generate relevance vector
        relevance = [ ]
        for slice_idx in range(num_slices):
            for patch_idx in range(patch_size):
                # print('slice_idx', slice_idx in relevant_slices, patch_idx, ' ')
                if slice_idx in relevant_slices:
                    for relevant_feature in ret_exp:
                        if relevant_feature[0] == slice_idx:
                            relevance.append(relevant_feature[1])
                else:
                    relevance.append(0)

        # print()
        # print(ret_exp[:,0])
        # print(np.array(relevance).shape)
        # print(relevance)
        # print()

        # cast and crop to len of ts
        relevance = np.array(relevance[:len(timeseries_instance)])
        return relevance

    def __data_labels_distances(cls,
                                timeseries,
                                model,
                                y_true,
                                num_samples,
                                num_slices,
                                perturbation='mean',
                                distance_metric='euclidean'):
        """Generates a neighborhood around a prediction.
        Generates neighborhood data by randomly removing slices from the
        time series and replacing them with other data points (specified by
        replacement_method: mean over slice range, mean of entire series or
        random noise). Then predicts with the classifier.
        Args:
            timeseries: Time Series to be explained.
                it can be a flat array (univariate)
                or (num_signals, num_points) (multivariate)
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear
                model (perturbation + original time series)
            num_slices: how many slices the time series will be split into
                for discretization.
            replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of slices in the time series. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: distance between the original instance and
                    each perturbed instance
        """

        def distance_cosine(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='cosine').ravel() * 100

        def distance_euclidean(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='euclidean').ravel()
                
        # NOTE: testing
        # def distance_euclidean_2(x):
        #     arr = x.copy()
        #     arr_reshaped = x[0].reshape([1, -1])

        #     distances = [
        #         distance.euclidean(arr[0], arr[i])
        #         for i in range(len(arr))
        #     ]
        #     distances = np.array(distances).ravel()
        #     return distances

        def distance_dtw(x):
            arr = x.copy()
            distances = [
                dtw.distance(arr[0], arr[i])
                for i in range(len(arr))
            ]
            distances = np.array(distances).ravel()
            return distances

        num_channels = 1
        len_ts = len(timeseries)
        if len(timeseries.shape) > 1:  # multivariate
            num_channels, len_ts = timeseries.shape
        
        # print()
        # print('-----')
        # print(len_ts)
        # print(num_slices)

        values_per_slice = math.ceil(len_ts / num_slices)
        deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries.copy()]

        # print('perturbation_matrix')
        # print(perturbation_matrix.shape)
        # print(perturbation_matrix[0])
        # print(perturbation_matrix[1])
        # print()

        for i, num_inactive in enumerate(deact_per_sample, start=1):
            logging.info("sample %d, inactivating %d", i, num_inactive)
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(features_range, num_inactive,
                                             replace=False)
            num_channels_to_perturb = np.random.randint(1, num_channels+1)

            channels_to_perturb = np.random.choice(range(num_channels),
                                                   num_channels_to_perturb,
                                                   replace=False)
            
            logging.info("sample %d, perturbing signals %r", i,
                         channels_to_perturb)
            
            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0
                
            tmp_series = timeseries.copy()

            for idx in inactive_idxs:
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len_ts)

                if perturbation == 'zero':
                    # use mean of slice as inactive
                    perturb_zero(tmp_series, start_idx, end_idx,
                                 channels_to_perturb)
                elif perturbation == 'mean':
                    # use mean of slice as inactive
                    perturb_mean(tmp_series, start_idx, end_idx,
                                 channels_to_perturb)
                elif perturbation == 'noise':
                    # use random noise as inactive
                    perturb_noise(tmp_series, start_idx, end_idx,
                                  channels_to_perturb)
                elif perturbation == 'total_mean':
                    # use total series mean as inactive
                    perturb_total_mean(tmp_series, start_idx, end_idx,
                                       channels_to_perturb)
            original_data.append(tmp_series)

        predictions = model.predict_input(np.array(original_data), y_true)
        
        # create a flat representation for features
        perturbation_matrix = perturbation_matrix.reshape((num_samples, num_channels * num_slices))

        distances = [ ]
        if distance_metric == 'cosine':
            distances = distance_cosine(perturbation_matrix)
        elif distance_metric == 'euclidean':
            distances = distance_euclidean(perturbation_matrix)
        elif distance_metric == 'dtw':
            distances = distance_dtw(perturbation_matrix)

        return perturbation_matrix, predictions, distances


def perturb_zero(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = 0
        return
    
    for chan in channels:
        m[chan][start_idx:end_idx] = 0


def perturb_total_mean(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = m.mean()
        return
    
    for chan in channels:
        m[chan][start_idx:end_idx] = m[chan].mean()

def perturb_mean(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.mean(m[start_idx:end_idx])
        return
    
    for chan in channels:
        m[chan][start_idx:end_idx] = np.mean(m[chan][start_idx:end_idx])
        
def perturb_noise(m, start_idx, end_idx, channels):
    # univariate
    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.random.uniform(m.min(), m.max(),
                                                 end_idx - start_idx)
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.random.uniform(m[chan].min(),
                                                       m[chan].max(),
                                                       end_idx - start_idx)