"""
Occlusion.py

Perturbation-based explanation "Occlusion" of Zeiler and Fergus 2014

Inspired by https://github.com/sicara/tf-explain

This Explanation is just for univariate Time Series Explanations
"""
import math
import numpy as np

from utils.perturbations import UTSPerturbations

class OcclusionSensitivityUTS:        

    """
    Performs occlusion sensitivity on univariate time series
    """
    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(self, timeseries_data, y_true, model, patch_size=5,
        perturbation='zero'):
        _timeseries_data = np.array(timeseries_data)
        _y_true = np.array(y_true)

        # print('_timeseries_data')
        # print(_timeseries_data.shape)
        # print(_timeseries_data[0])
        # print()
        # print('_y_true')
        # print(_y_true.shape)
        # print(_y_true[0])
        # print()
        
        if type(_timeseries_data.tolist())!=list:
            _timeseries_data = np.array([[timeseries_data]])
        if type(_y_true.tolist())!=list:
            _y_true = np.array([[y_true]])

        explanations = np.array([
            self.explain_instance(_timeseries_data[idx], _y_true[idx], model,
                perturbation=perturbation)
            for idx in range(_timeseries_data.shape[0]) 
        ])
        return explanations


    # TODO: integrieren in jeder explanation
    def explain_instance( 
        self, 
        timeseries_instance,
        true_class,
        model,
        patch_size=1,
        perturbation='zero'
    ):
        """
        Computes Occlusion sensitity maps for a specific time_series instance,
        Explanation comes as attribution map = [[ts-feature1, attribution1], ...]

        Args:
            validation_data:
            model(tf.keras.Model): model to inspect 
            patch_size(int): Size of patch to apply on the image
        Returns:
            np.array: 2d array: ()
        """

        # if type(true_class) is not int:
            # raise Exception('True class label is not a type of integer')
        # if len(timeseries_instance) % patch_size != 0:
        #     raise Exception('The patch size does not divide the time series shape')

        perturbator = UTSPerturbations()

        # print('timeseries_instance')
        # print(timeseries_instance.shape)
        # print(timeseries_instance[0])

        # generate perturbed time series
        perturbed_timeseries = np.array([ 
            perturbator.apply_perturbation(timeseries_instance.copy(), 
                end_idx, (end_idx + patch_size), perturbation=perturbation)
            for start_idx, end_idx in enumerate(range(0, len(timeseries_instance), patch_size))
        ])

        # print('perturbed_timeseries')
        # print(perturbed_timeseries.shape)
        # print(perturbed_timeseries[0])

        # print('true_class')
        # print(true_class.shape)
        # print(true_class)
        # predict perturbed time series
        prediction_t = model.predict_input(perturbed_timeseries, true_class)
        predictions_Z = model.predict_input(perturbed_timeseries, true_class)

        

        # extract predictions of time series based on true class
        target_class_predictions = [
            # prediction[true_class - 1] for prediction in predictions
            prediction[true_class] for prediction in predictions_Z
        ]

        # print(target_class_predictions[0])

        # generate array with size of time series
        sensitivtiy_map = np.zeros(
            math.ceil(len(timeseries_instance))
        )

        # assign every prediction to the right place/subsequence of the
        # confidence map
        counter = 0
        for idx in range(len(sensitivtiy_map)):
            sensitivtiy_map[idx] = 1 - target_class_predictions[counter]
            if idx % patch_size == (patch_size - 1):
                counter +=1

        # attribution_map = [list(i) for i in zip(timeseries_instance, sensitivtiy_map)]

        # return np.array(attribution_map)
        return np.array(sensitivtiy_map)
