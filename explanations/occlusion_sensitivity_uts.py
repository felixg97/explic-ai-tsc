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

    def __init__(self, batch_size=None, perturbation='occlusion'):
        self.batch_size = batch_size
        self.perturbation = perturbation

    # TODO: integrieren in jeder explanation
    def explain_instance( 
        self, 
        timeseries_instance,
        true_class,
        model,
        patch_size=1,
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
        if len(timeseries_instance) % patch_size != 0:
            raise Exception('The patch size does not divide the time series shape')

        perturbator = UTSPerturbations()

        # print('timeseries_instance')
        # print(timeseries_instance)
        # print(timeseries_instance.shape)

        # generate perturbed time series
        perturbed_timeseries = np.array([ 
            perturbator.apply_perturbation(timeseries_instance.copy(), end_idx, (end_idx + patch_size))
            for start_idx, end_idx in enumerate(range(0, len(timeseries_instance), patch_size))
        ])

        # print('perturbed_timeseries')
        # print(perturbed_timeseries.shape)
        # print(perturbed_timeseries[0])
        # predict perturbed time series
        predictions = model.predict_input(perturbed_timeseries, true_class)

        # extract predictions of time series based on true class
        target_class_predictions = [
            prediction[true_class - 1] for prediction in predictions
        ]

        # generate array with size of time series
        confidence_map = np.zeros(
            math.ceil(len(timeseries_instance))
        )

        # assign every prediction to the right place/subsequence of the
        # confidence map
        counter = 0
        for idx in range(len(confidence_map)):
            confidence_map[idx] = 1 - target_class_predictions[counter]
            if idx % patch_size == (patch_size - 1):
                counter +=1

        # attribution_map = [list(i) for i in zip(timeseries_instance, confidence_map)]

        # return np.array(attribution_map)
        return np.array(confidence_map)
