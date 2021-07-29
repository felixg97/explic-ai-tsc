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
import numpy as np

from sklearn.metrics import accuracy_score

from utils.perturbations import UTSPerturbations
from utils.utils import calculate_metrics

class PerturbationAnalysis:

    def __init__(self):
        self.perturbator = UTSPerturbations()


    def evaluate__xai_method(self, 
        test_accuracy, x_test, y_true, 
        classification_model, xai_model, batch_size=0,
        quality_metric='acc', threshold=90, verification_method='all'):

        """Evaluate explanation method
        
        Hyperparameters: threshold e 
        """
        evaluation = { }
        eval_zero_tp = None
        eval_inverse_tp = None
        eval_mean_tp = None 
        eval_swap_sq = None
        eval_zero_sq = None
        eval_inverse_sq = None
        eval_mean_sq = None

        batch_size_in = 0
        if batch_size > 0:
            batch_size_in = batch_size

        if verification_method == 'zero_tp':
            eval_zero_tp = self._evaluate_tp(test_accuracy, x_test, y_true, classification_model, 
                xai_model, quality_metric, threshold, batch_size_in, 'occlusion')
        elif verification_method == 'inverse_tp':
            eval_inverse_tp = self._evaluate_tp(test_accuracy, x_test, y_true, classification_model, 
                xai_model, quality_metric, threshold, batch_size_in, 'occlusion')
        elif verification_method == 'mean_tp':
            eval_mean_tp = self._evaluate_tp(test_accuracy, x_test, y_true, classification_model, 
                xai_model, quality_metric, threshold, batch_size_in, 'mean')
        elif verification_method == 'swap_sq':
            eval_swap_sq = None
        elif verification_method == 'zero_sq':
            eval_zero_sq = None
        elif verification_method == 'inverse_sq':
            eval_inverse_sq = None
        elif verification_method == 'mean_sq':
            eval_mean_sq = None
        elif verification_method == 'all':
            eval_zero_tp = None
            eval_inverse_tp = None
            eval_mean_tp = None 
            eval_swap_sq = None
            eval_zero_sq = None
            eval_inverse_sq = None
            eval_mean_sq = None
            #end


        # timeseries = np.array([timeseries_instance.copy()])

        # print('Explanation')
        # print(explanation.shape)

        # Determine time points > threshold e  
        # eth_percentile = np.percentile(explanation, threshold)

        # Assign true values to indices overshooting the eth_percentile
        # tps_with_threshold = np.array([
            # i if (i >= eth_percentile) else 0
        #     True if (i >= eth_percentile) else False
        #     for i in explanation
        # ])

        # print()
        # print('tps_with_threshold')
        # print(tps_with_threshold.shape)
        # print(tps_with_threshold)
        
        # print()
        # for

        # gen_arrs_w_idcs_dict = self._generate_arr_4_overshooting_tp(
            # tps_with_threshold, timeseries)

        

        ############ 1
        # eval_zero_tp = None

        # perturbed_ts = np.array([
            # self.perturbator.apply_perturbation(gen_arrs_w_idcs_dict[key][0], key, key)
            # for key in gen_arrs_w_idcs_dict
        # ])

        # print('perturbed_ts')
        # print(perturbed_ts.shape)

        # prediction_ts = model.predict_input(timeseries)
        # predictions_perturbed_ts = model.predict_input(perturbed_ts)

        # print('prediction_true_ts')
        # print(prediction_ts)
        # print('predictions')
        # print(predictions_perturbed_ts)

        # acacc = self._calculate_qm_average_changed_acc(true_class_index ,prediction_ts, predictions_perturbed_ts)

        # print(acacc)
        ############ 2
        # eval_inverse_tp = None

        # perturbed_ts = [
        #     self.perturbator.apply_perturbation(gen_arrs_w_idcs_dict[key][0], key, key)
        #     for key in gen_arrs_w_idcs_dict
        # ]

        evaluation['eval_zero_tp'] = eval_zero_tp
        evaluation['eval_inverse_tp'] = eval_inverse_tp
        evaluation['eval_mean_tp'] = eval_mean_tp
        evaluation['eval_swap_sq'] = eval_swap_sq
        evaluation['eval_zero_sq'] = eval_zero_sq
        evaluation['eval_inverse_sq'] = eval_inverse_sq
        evaluation['eval_swap_sq'] = eval_swap_sq
        return evaluation

    def _evaluate_tp(self, test_accuracy, x_test, y_true, classification_model, 
        xai_model, quality_metric, threshold, batch_size, perturbation_method):

        _x_test = None
        _y_true = None
        if batch_size != x_test.shape[0]:
            _x_test = np.array(x_test[:batch_size].copy())
            _y_true = np.array(y_true[:batch_size].copy())
        else:
            _x_test = np.array(x_test[:].copy())
            _y_true = np.array(y_true[:batch_size].copy())


        # Calculate relevance of every time series # TOOD: integrate batch explaining
        relevances = np.array([
            xai_model.explain_instance(x_test[idx], _y_true[idx], classification_model)
            for idx in range(batch_size)
        ])

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
        for i in range(relevances.shape[0]):
            # i: index of time series to perturb
            timeseries_to_perturb = _x_test[i].copy()
            for j in range(len(reltps_over_threshoold[i])):
                # j: index of time point to perturb
                if reltps_over_threshoold[i][j]:
                    self.perturbator.apply_perturbation(timeseries_to_perturb, j, 
                        j, perturbation_method=perturbation_method)
            perturbed_ts.append(timeseries_to_perturb)

        perturbed_ts = np.array(perturbed_ts)

        # Predict perturbed time series
        predictions_perturbed_ts = classification_model.predict_input(perturbed_ts)

        # Calculate metrics for perturbed prediction
        acc = self._calculate_accuracy_score(_y_true, predictions_perturbed_ts)
        
        # return abs(test_accuracy - acc)
        return (acc - test_accuracy) # erstmal differenz raus


    def _evaluate_sq(self, test_accuracy, x_test, y_true, classification_model, 
        xai_model, quality_metric, threshold, batch_size, perturbation_method):

        pass


    def _generate_arr_4_overshooting_tp(self, threshold_arr, timeseries_instance):
        gen_dict = {}
        for i in range(len(threshold_arr)):
            if threshold_arr[i] == True:
                gen_dict[i] = timeseries_instance.copy()
        return gen_dict


    def _calculate_accuracy_score(self, y_true, y_pred):
        _y_true = np.array(y_true)
        _y_pred = np.array(y_pred)

        print(_y_true)
        print(_y_pred)

        correctly_classified = 0
        for predition_idx in range(_y_pred.shape[0]):
            true_class = _y_true[predition_idx]
            predicted_class = np.argmax(_y_pred[predition_idx])
            if true_class == predicted_class: correctly_classified += 1

        acc = correctly_classified / _y_pred.shape[0]
        return acc


    
