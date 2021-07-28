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

    
    def explain_instance( 
        self, 
        timeseries_instance,
        true_class=None,
        model=None, 
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

        if type(true_class) is not int:
            raise Exception('True class label is not a type of integer')
        if len(timeseries_instance) % patch_size != 0:
            raise Exception('The patch size does not divide the time series shape')

        print('CONFIGURATIONS')
        print()
        print('True class:\t', true_class)
        print('Model:\t', model)
        print('Patch size:\t', patch_size)
        print()

        # print()
        # print()
        # print('######## Time series instance: ########')
        # print(timeseries_instance.shape)
        # print(timeseries_instance)
        # print()
        # print()

        perturbator = UTSPerturbations()

        perturbed_timeseries = [ 
            perturbator.apply_perturbation(timeseries_instance.copy(), end_idx, (end_idx + patch_size))
            for start_idx, end_idx in enumerate(range(0, len(timeseries_instance), patch_size))
        ]

        # print()
        # print()
        # print('######## Perturbed Time series instance: ########')
        # print(len(perturbed_timeseries))
        # print(perturbed_timeseries[:2])
        # print('CHECK - Funktioniert')
        # print()
        # print()

        predictions = model.predict_input(np.array(perturbed_timeseries))

        # print()
        # print()
        # print('######## Predictions: ########')
        # print(predictions.shape)
        # print(predictions)
        # print('CHECK - Funktioniert')
        # print()
        # print()


        target_class_predictions = [
            prediction[true_class - 1] for prediction in predictions # TODO: true_class schlauer lösen
        ]

        # print()
        # print()
        # print('######## Target class predictions: ########')
        # print(len(target_class_predictions))
        # print(target_class_predictions)
        # print()
        # print()

        confidence_map = np.zeros(
            math.ceil(len(timeseries_instance))
        )

        counter = 0
        for idx in range(len(confidence_map)):
            confidence_map[idx] = 1 - target_class_predictions[counter]
            if idx % patch_size == 3:
                counter +=1

        attibution_map = [list(i) for i in zip(timeseries_instance, confidence_map)]

        # print()
        # print('######## Attribution Map: ########')
        # print(attibution_map)
        # print()
        # print()

        return np.array(attribution_map)
