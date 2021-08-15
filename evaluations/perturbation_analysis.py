"""
perturbation_analysis.py

Perturbation Analyis proposed by schlegel2019

@inproceedings{Schlegel2019,
  doi = {10.1109/iccvw.2019.00516},
  url = {https://doi.org/10.1109/iccvw.2019.00516},
  year = {2019},
  month = oct,
  publisher = {{IEEE}},
  author = {Udo Schlegel and Hiba Arnout and Mennatallah El-Assady and Daniela Oelke and Daniel A. Keim},
  title = {Towards A Rigorous Evaluation Of {XAI} Methods On Time Series},
  booktitle = {2019 {IEEE}/{CVF} International Conference on Computer Vision Workshop ({ICCVW})}
}
"""

# How it works
#
# (a) dataset D consisting of n samples with C:{c[1], ... c[k]} classes
# (b) A sample of t of D consists of m time points t = (t[1], ..., t[m])
#
# (c) The local feature importance produces a relevance r[i] for each time point
#     t[i]. Afterward a tuple (t[i], r[i]) can be build (!). Or more general
#     for the time series vector t a relevance vector r
#
# (d) A model m trained on a subset X from D with labels Y can be formalized to
#     m(x)=y with x element of X and y element of Y
#
# (e) The model m learns based on the provided data X,Y to predict an unseen subset
#     X[new].
#
# (f) In the case of time series, x is a sample like t = (t[0], ..., t[m]) with 
#     m time points
# 
# (g) If then XAI method xai is incorporated to explain the decisions of such a
#     model, another layer on top of it is created
#
# (h) An explanation can be formalized as xai(x,m) = exp
#
# (i) With time series, the explanation exp is a relevance vector r = (r[0], ..., r[m])
#
#### 1 - Perturbation on time series
#
## (1) Perturbation analysis
#
# (a) Assumed that the relevance produced by XAI method should get worse results 
#     of the quality metric qm for the classifier if combined. (!)
#
# (b) A point t[i] gets changed if r[i] is larger than a certain threshold e,
#     e.g. the 90th percentile of r (!)
# 
# (c) Due to XAI methods having problems with some time series samples, the threshold
#     leads to only changing a small number of time points.
#
# (d) In the case of time series, the point t[i] is set to zero or the inverse: (max(t[i]) - t[i])
#     and leads to the new time samples t^zero and t^inverse
#
## (2) Perturbation verification
#
# (a) To verify the assumption, a random relevance r[r] = (r[0], ..., r[m]) is used
#     for the same procedure
#
# (b) The number of changed time points, amount of r[i] larger than the threshold e,
#     is the same as in the case before to set the same prerequisits for the classifier
#     
# (c) This technique creates new time series like the perturbation analysis such
#     as t^zero[r] and t^inverse[r].
#
# (d) The assumption to verify the model follows the schema like
#        like qm(t) >= qm(t^zero[r]) >= qm(t^zero) or
#        like qm(t) >= qm(t^inverse[r]) >= qm(t^inverse)
#     for a model that maximizes qm
#
#
#### 2 - Sequence evaluation
#
## (3) Swap Time Points (inverse sub sequence)
#
# (a) Method takes time series t, and relevance r. However, it takes the time points
#     with the relevance over the threshold as the starting point as the starting 
#     point for further changes of the time series. (!) 
#     => So r[i] > e describes the starting point to extract the sub-sequence 
#        t[sub]=(t[i], t[i+1], ..., t[i+n[s]]) with length n[s]
#
# (b) The sub-sequence then gets reversed to t[sub] = (t[i+n[s]], ..., t[i+1], t[i])
#     and inserted back to the time series
#
# (c) Further the subsequence gets set to zero to test the method
#
# (d) The same procedure is done with a random time point positions to verify the
#     time points relevance again
#
#
## (4) Mean Time Points (mean sub sequence)
#
# (a) Same as swap time points, but instead of swapping the time points the mean
#     is calculated
#     t[sub] = (t[i], ..., t[i+n[s]])
#     mean(t[sub]) = (mean(t[sub])[i], ..., mean(t[sub])[i+n[s]])
#
#
# Average changed accuracy
from os import terminal_size
import numpy as np
import time

from sklearn.metrics import accuracy_score

# from utils.perturbations import UTSPerturbations
from utils.utils import calculate_metrics

# NOTE: 
# Perturbation Analysis currently does only return accuracies after perturbations
# Average changes accuracies are not calculated here (not meant for here)
#
#
class PerturbationAnalysisUTS:

    def __init__(self):
        # self.perturbator = UTSPerturbations()
        pass

    def evaluate_relevance_vectors_for_explanation_set(self, test_accuracy, x_test, 
        y_true, x_train, y_train, y_test, classification_model, relevance_vectors,
        patch_size=None, quality_metric='acc', threshold=90, sequence_length=1, 
        verification_method='all', determine='threshold'):

        if patch_size == None: return

        evaluation = { }

        # measure time
        start_time = time.time()

        # time point evaluations
        eval_zero_tp = None
        eval_inverse_tp = None
        eval_mean_tp = None 

        # sequence evaluation
        eval_swap_sq = None
        eval_zero_sq = None
        eval_inverse_sq = None
        eval_mean_sq = None

        # batch_size_in = 0
        # if batch_size > 0:
        #     batch_size_in = batch_size
        # elif batch_size == 0:
        #     batch_size_in = x_test.shape[0]

        evaluation['verification'] = verification_method
        evaluation['quality_metric'] = quality_metric
        # evaluation['batch'] = batch_size_in
        evaluation['threshold'] = threshold
        evaluation['sequence_length'] = sequence_length

        _x_test = np.array(x_test.copy())
        _y_true = np.array(y_true.copy())
        _x_train = np.array(x_train.copy())
        _y_train = np.array(y_train.copy())
        _y_test = np.array(y_test.copy())

        # NOTE: workaround 
        xai_model = None

        if verification_method == 'zero_timepoint':
            eval_zero_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, 'occlusion', patch_size, relevance_vectors, determine=determine)
        elif verification_method == 'inverse_timepoint':
            eval_inverse_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, 'value_inversion', patch_size, relevance_vectors,determine=determine)
        # elif verification_method == 'mean_timepoint':
        #     eval_mean_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
        #         _y_train, _y_test, classification_model, xai_model, quality_metric, 
        #         threshold, batch_size_in, 'total_mean', xai_model_true=False, relevances_in=relevance_vectors)
        # elif verification_method == 'swap_sequence':
        #     eval_swap_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
        #         _y_train, _y_test, classification_model, xai_model, quality_metric, 
        #         threshold, batch_size_in, 'sequence_swap', sequence_length, xai_model_true=False, relevances_in=relevance_vectors)
        # elif verification_method == 'zero_sequence':
        #     eval_zero_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
        #         _y_train, _y_test, classification_model, xai_model, quality_metric, 
        #         threshold, batch_size_in, 'occlusion', sequence_length, xai_model_true=False, relevances_in=relevance_vectors)
        # elif verification_method == 'inverse_sequence':
        #     pass
        #     eval_inverse_sq = 0
            # INFO: Value inversion for sequences is currently paused, See perturbations.py
            # eval_inverse_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'value_inversion', sequence_length)
        # elif verification_method == 'mean_sequence':
        #     eval_mean_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
        #         _y_train, _y_test, classification_model, xai_model, quality_metric, 
        #         threshold, batch_size_in, 'total_mean', sequence_length, xai_model_true=False, relevances_in=relevance_vectors)
        elif verification_method == 'all':
            eval_zero_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, 'occlusion', patch_size, relevance_vectors, determine=determine)
            eval_inverse_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, 'value_inversion', patch_size, relevance_vectors, determine=determine)
            # eval_mean_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'total_mean', xai_model_true=False, relevances_in=relevance_vectors) 
            # eval_swap_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'sequence_swap', sequence_length, xai_model_true=False, relevances_in=relevance_vectors)
            # eval_zero_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'occlusion', sequence_length, xai_model_true=False, relevances_in=relevance_vectors)
            # eval_inverse_sq = 0
            # INFO: Value inversion for sequences is currently paused, See perturbations.py
            # eval_inverse_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'value_inversion', sequence_length)
            # eval_mean_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'total_mean', sequence_length, xai_model_true=False, relevances_in=relevance_vectors)

        #
        #
        # 
        #
        evaluation['threshold'] = threshold 
        evaluation['zero_timepoint'] = eval_zero_tp
        evaluation['inverse_timepoint'] = eval_inverse_tp
        # evaluation['mean_timepoint'] = eval_mean_tp
        # evaluation['swap_sequence'] = eval_swap_sq
        # evaluation['zero_sequence'] = eval_zero_sq
        # evaluation['inverse_sequence'] = eval_inverse_sq
        # evaluation['swap_sequence'] = eval_swap_sq
        evaluation['base_acc'] = test_accuracy
        evaluation['duration']=time.time()-start_time

        return evaluation



    def evaluate_relevance_for_instance(self, test_accuracy, x_test, y_true, x_train, y_train, 
        y_test, classification_model, xai_model, batch_size=0, quality_metric='acc', 
        threshold=90, sequence_length=1, verification_method='all', database_name='not specified', 
        classification_model_name='not specified', xai_method_name='not specified'):

        """Evaluate explanation method
        
        Hyperparameters: threshold e 
        """
        evaluation = { }

        # measure time
        start_time = time.time()

        # time point evaluations
        eval_zero_tp_sys = None
        eval_zero_tp_ran = None
        eval_inverse_tp_sys = None
        eval_inverse_tp_ran = None
        eval_mean_tp_sys = None 
        eval_mean_tp_ran = None 

        # sequence evaluation
        eval_swap_sq_sys = None
        eval_swap_sq_ran = None
        eval_zero_sq_sys = None
        eval_zero_sq_ran = None
        eval_inverse_sq_sys = None
        eval_inverse_sq_ran = None
        eval_mean_sq_sys = None
        eval_mean_sq_ran = None

        batch_size_in = 0
        if batch_size > 0:
            batch_size_in = batch_size
        elif batch_size == 0:
            batch_size_in = x_test.shape[0]

        evaluation['verification'] = verification_method
        evaluation['quality_metric'] = quality_metric
        evaluation['batch'] = batch_size_in
        evaluation['threshold'] = threshold
        evaluation['sequence_length'] = sequence_length

        _x_test = None
        _y_true = None
        _x_train = None
        _y_train = None
        _y_test = None
        if batch_size != x_test.shape[0]:
            _x_test = np.array(x_test[:batch_size].copy())
            _y_true = np.array(y_true[:batch_size].copy())
            _x_train = np.array(x_train[:batch_size].copy())
            _y_train = np.array(y_train[:batch_size].copy())
            _y_test = np.array(y_test[:batch_size].copy())
        else:
            _x_test = np.array(x_test.copy())
            _y_true = np.array(y_true.copy())
            _x_train = np.array(x_train.copy())
            _y_train = np.array(y_train.copy())
            _y_test = np.array(y_test.copy())

        if verification_method == 'zero_timepoint':
            eval_zero_tp_sys, eval_zero_tp_ran = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'occlusion')
        elif verification_method == 'inverse_timepoint':
            eval_inverse_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'value_inversion')
        elif verification_method == 'mean_timepoint':
            eval_mean_tp = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'total_mean')
        elif verification_method == 'swap_sequence':
            eval_swap_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'sequence_swap', sequence_length)
        elif verification_method == 'zero_sequence':
            eval_zero_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'occlusion', sequence_length)
        elif verification_method == 'inverse_sequence':
            pass
            eval_inverse_sq = 0
            # INFO: Value inversion for sequences is currently paused, See perturbations.py
            # eval_inverse_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'value_inversion', sequence_length)
        elif verification_method == 'mean_sequence':
            eval_mean_sq = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'total_mean', sequence_length)
        elif verification_method == 'all':
            eval_zero_tp_sys, eval_zero_tp_ran = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'occlusion')
            eval_inverse_tp_sys, eval_inverse_tp_ran = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'value_inversion')
            eval_mean_tp_sys, eval_mean_tp_ran = self._evaluate_timepoint(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'total_mean') 
            eval_swap_sq_sys, eval_swap_sq_ran = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'sequence_swap', sequence_length)
            eval_zero_sq_sys, eval_zero_sq_rand = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'occlusion', sequence_length)
            eval_inverse_sq = 0
            # INFO: Value inversion for sequences is currently paused, See perturbations.py
            # eval_inverse_sq_sys, eval_inverse_sq_ran = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
            #     _y_train, _y_test, classification_model, xai_model, quality_metric, 
            #     threshold, batch_size_in, 'value_inversion', sequence_length)
            eval_mean_sq_sys, eval_mean_sq_ran = self._evaluate_sequence(test_accuracy, _x_test, _y_true, _x_train, 
                _y_train, _y_test, classification_model, xai_model, quality_metric, 
                threshold, batch_size_in, 'total_mean', sequence_length)
            #end

        evaluation['zero_timepoint'] = eval_zero_tp_sys
        evaluation['zero_timepoint_random_verification'] = eval_zero_tp_ran
        evaluation['inverse_timepoint'] = eval_inverse_tp_sys
        evaluation['inverse_timepoint_random_verification'] = eval_inverse_tp_ran
        evaluation['mean_timepoint'] = eval_mean_tp_sys
        evaluation['mean_timepoint_random_verification'] = eval_mean_tp_ran
        evaluation['swap_sequence'] = eval_swap_sq_sys
        evaluation['swap_sequence_random_verification'] = eval_swap_sq_ran
        evaluation['zero_sequence'] = eval_zero_sq_sys
        evaluation['zero_sequence_random_verification'] = eval_zero_sq_ran
        evaluation['inverse_sequence'] = eval_inverse_sq_sys
        evaluation['inverse_sequence_random_verification'] = eval_inverse_sq_ran
        evaluation['swap_sequence'] = eval_swap_sq_sys
        evaluation['swap_sequence_random_verification'] = eval_swap_sq_ran
        evaluation['duration']=time.time()-start_time

        return evaluation

    def _evaluate_timepoint_old(self, test_accuracy, x_test, y_true, x_train, y_train, 
        y_test, classification_model, xai_model, quality_metric, threshold, perturbation_method, xai_model_true=True, relevances_in=None):


        # Calculate relevance of every time series if relevances are not present
        if xai_model_true:
            relevances = np.array([
                xai_model.explain_instance(x_test[idx], y_true[idx], classification_model) # TODO: perturbation für occlusion -> auf andere anpassen
                for idx in range(batch_size)
            ])
        else:
            relevances = relevances_in


        # Determine time points > threshold e for every time series 
        eth_percentile = np.array([
            np.percentile(relevance, threshold) 
            for relevance in relevances
        ])

        # Assign True to indices overshooting the eth_percentile False otherwise
        reltps_over_threshoold = np.array([
            [True if (jdx >= eth_percentile[idx]) else False for jdx in relevances[idx]]
            for idx in range(relevances.shape[0])
        ])


        # Perturb time series based on their value > threshold e
        perturbed_ts = []
        for i in range(relevances.shape[0]):
            # i: index of time series to perturb
            timeseries_to_perturb = x_test[i].copy()
            reference_timeseries = x_test[i].copy()
            for j in range(len(reltps_over_threshoold[i])):
                # j: index of time point to perturb
                if reltps_over_threshoold[i][j]:
                    timeseries_to_perturb = self.perturbator.apply_perturbation(
                        timeseries_to_perturb, j, j, perturbation=perturbation_method,
                        reference_timeseries=reference_timeseries)
            perturbed_ts.append(timeseries_to_perturb)
        perturbed_ts = np.array(perturbed_ts)


        # Predict perturbed time series
        metrics_sys_perturbed = classification_model.predict(perturbed_ts, y_true, x_train, y_train, y_test)
        metrics_rand_perturbed = classification_model.predict(perturbed_ts, y_true, x_train, y_train, y_test)

        # print(metrics_sys_perturbed)
        # print(metrics_rand_perturbed)

        return metrics_sys_perturbed['accuracy'][0], metrics_sys_perturbed['accuracy'][0]
        #end of _evaluate_timepoint

    def _evaluate_sequence(self, test_accuracy, x_test, y_true, x_train, y_train, y_test,
        classification_model, xai_model, quality_metric, threshold, batch_size, 
        perturbation_method, sequence_length, xai_model_true=True, relevances_in=None):

        # TODO: idee = sequence_length = stride width
        if sequence_length == 1:
            return 0

        # Calculate relevance of every time series # TOOD: integrate batch explaining
        if xai_model_true:
            relevances = np.array([
                xai_model.explain_instance(x_test[idx], y_true[idx], classification_model) # TODO: perturbation für occlusion -> auf andere anpassen
                for idx in range(batch_size)
            ])
        else:
            relevances = relevances_in

        # Determine time points > threshold e for every time series 
        eth_percentile = np.array([
            np.percentile(relevance, threshold) 
            for relevance in relevances
        ])

        # Assign true values to indices overshooting the eth_percentile
        reltps_over_threshoold = np.array([
            [True if (jdx >= eth_percentile[idx]) else False for jdx in relevances[idx]]
            for idx in range(relevances.shape[0])
        ])

        # Perturb time series based on their value > threshold e
        perturbed_ts = []
        true_count_ts = [] # factor that y_true[i] gets expanded by
        for i in range(relevances.shape[0]):
            # i: index of time series to perturb
            # count true values of timeseries x_test[i]
            true_count_ts.append(np.sum(reltps_over_threshoold[i]))

            # copy time series to append in perturbed_ts without mutating it
            timeseries_to_perturb = x_test[i].copy()
            reference_timeseries = x_test[i].copy()
            for j in range(len(reltps_over_threshoold[i])):
                # j: index of array 
                if reltps_over_threshoold[i][j]:    
                    # timeseries for appending
                    max_end = (reference_timeseries.shape[0]-1)
                    sequence_end = (j+sequence_length) if (j+sequence_length) <= max_end else max_end
                    timeseries_to_append = self.perturbator.apply_perturbation(
                        timeseries_to_perturb.copy(), j, sequence_end, perturbation=perturbation_method,
                        reference_timeseries=reference_timeseries)
                    perturbed_ts.append(timeseries_to_append)
        perturbed_ts = np.array(perturbed_ts)

        # print()
        # print(perturbation_method)
        # print()
        # print('perturbed_ts')
        # print(perturbed_ts.shape)
        # print(reltps_over_threshoold[0])
        # print(perturbed_ts[0])

        # Expand y_true to match the shape of perturbed_ts
        y_true_expanded = np.array([
            y_true[i] 
            for j in range(true_count_ts[i])
            for i in range(len(true_count_ts))
        ])

        # Predict perturbed time series
        metrics = classification_model.predict(perturbed_ts, y_true_expanded, x_train, y_train, y_test)
        
        return metrics['accuracy'][0]
        #end of _evaluate_sequence


################################################################################

    def _evaluate_timepoint(
            self, 
            test_accuracy, 
            x_test, 
            y_true, 
            x_train, 
            y_train, 
            y_test, 
            classification_model, 
            xai_model, 
            quality_metric, 
            threshold, 
            perturbation_method,
            patch_size,
            relevances_in,
            determine='threshold'
            ):
        
        # Calculate relevance of every time series if relevances are not present
        relevances = relevances_in

        # print(relevances.shape)
        # print(relevances[0])
        eth_percentile = None

        # Determine time points > threshold e for every time series 
        if determine == 'threshold':
            eth_percentile = np.array([
                self.determine_percentile(relevance, threshold, patch_size) 
                for relevance in relevances
            ])
        elif determine == 'random':
            eth_percentile = np.array([
                self.determine_random(relevance, patch_size)
                for relevance in relevances
            ])

        # print(eth_percentile.shape)
        # print(eth_percentile)

        # return
        perturbed_ts = []
        perturbed_ts_y_true = []
        for idx in range(len(eth_percentile)):
            timeseries_to_ptb = x_test[idx].copy()
            ptb_ts_y_true = y_true[idx].copy()
            # print('hallo', idx)
            for jdx in range(idx+1):
                # print(eth_percentile[idx][jdx])
                try:
                    if eth_percentile[idx][jdx] == 0:
                        idx_pert_start = jdx*patch_size
                        potential_end = idx_pert_start + patch_size 
                        max_end = len(timeseries_to_ptb)
                        idx_pert_end = potential_end if potential_end <= max_end else max_end
                        if perturbation_method == 'value_inversion':
                            perturbed_ts.append(self.perturb_inverse(timeseries_to_ptb, idx_pert_start, idx_pert_end))
                        elif perturbation_method == 'occlusion':
                            perturbed_ts.append(self.perturb_zero(timeseries_to_ptb, idx_pert_start, idx_pert_end))
                        perturbed_ts_y_true.append(ptb_ts_y_true)
                except:
                    pass

        perturbed_ts = np.array(perturbed_ts)
        perturbed_ts_y_true = np.array(perturbed_ts_y_true)

        if perturbed_ts.shape[0] == 0: return -1

        if perturbed_ts.shape[0] != perturbed_ts_y_true.shape[0]:
            print('No no, no classification at home.')

        # print(perturbed_ts.shape)
        # print(perturbed_ts[:5])
        # print(perturbed_ts_y_true.shape)
        # print(perturbed_ts_y_true[:5])


        # # Predict perturbed time series
        metrics_sys_perturbed = classification_model.predict(perturbed_ts, perturbed_ts_y_true, x_train, y_train, y_test)

        # print(metrics_sys_perturbed)

        # return
        return metrics_sys_perturbed['accuracy'][0]


    def determine_percentile(self, relevance, threshold, patch_size):

        # split into patches and mean them 
        relevance_inpatches = []

        for patch_idx in range(0, len(relevance), patch_size):
            patch = relevance[patch_idx:patch_idx+patch_size]
            mean = np.mean(patch)
            new_patch = [mean]*patch_size
            relevance_inpatches.append(new_patch)

        # print(relevance_inpatches[0])
        # print(np.array(relevance_inpatches).shape)
        relevance_inpatches_flattened = np.array(relevance_inpatches.copy()).flatten()
        max_relevance = np.max(relevance_inpatches)
        threshold_val = max_relevance * threshold/100
            
        result = []
        for patch in relevance_inpatches:
            if patch[0] >= threshold_val:
                result.append(0)
            else:
                result.append(1)
        
        # print(result)
        return result

    def determine_random(self, relevance, patch_size):
        patch_number = int(np.floor(len(relevance)/patch_size))

        # print(patch_number)

        # return

        randomized = np.random.randint(2, size=patch_number) < .5

        results = []
        for el in randomized:
            if el: results.append(0)
            else: results.append(1)

        return results 


    def perturb_zero(self, timeseries, start, end):
        _timeseries = timeseries.copy()
        if start == end:
            _timeseries[start] = 0
        else:
            _timeseries[start:end] = 0
        return _timeseries

    def perturb_inverse(self, timeseries, start, end):
        _timeseries = timeseries.copy()
        max_to_invert = np.max(timeseries)
        if start == end:
            val = _timeseries[start]
            _timeseries[start] = max_to_invert - val
        else: 
            for i in range(start, end):
                val = _timeseries[i]
                _timeseries[i] = max_to_invert - val
        return _timeseries

    # def perturb_mean(self):
    #     pass
