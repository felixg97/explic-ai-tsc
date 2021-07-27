"""
Occlusion.py

Perturbation-based explanation "Occlusion" of Zeiler and Fergus 2014

Inspired by https://github.com/sicara/tf-explain

This Explanation is just for univariate Time Series Explanations
"""
import math
import numpy as np

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
        true_class,
        model, 
        patch_size=1,
    ):
        """
        Computes Occlusion sensitity maps for a specific time_series instance,
        
        Args:
            validation_data:
            model(tf.keras.Model): model to inspect 
            patch_size(int): Size of patch to apply on the image
        Returns:
            np.array: 2d array: ()
        """

        if len(timeseries_instance) % patch_size != 0:
            raise Exception('The patch size does not divide the time series shape')

        sensitivity_map = np.zeros(
            math.ceil(len(timeseries_instance) / patch_size)
        )

        # print(timeseries_instance)

        perturbed_timeseries = [ 
            self.apply_perturbation(timeseries_instance.copy(), end_idx, (end_idx + patch_size))
            for start_idx, end_idx in enumerate(range(0, len(timeseries_instance), patch_size))
        ]

        # print(perturbed_timeseries)

        indices = [*range(len(timeseries_instance / patch_size))]

        # predictions = model.predict(np.array(perturbed_timeseries), batch_size=self.batch_size)
        predictions = []

        target_class_predictions = [
            prediction[true_class] for prediction in predictions
        ]


        for idx, confidence in zip(indices, target_class_predictions):
            sensitivity_map[idx] = 1 - confidence

        return sensitivity_map
    

    def apply_perturbation(self, timeseries_instance, start_idx, end_idx):
        if self.perturbation == 'occlusion':
            self.perturb_occlusion(timeseries_instance, start_idx, end_idx)
        elif self.perturbation == 'mean':
            self.perturb_mean(timeseries_instance, start_idx, end_idx)
        elif self.perturbation == 'total_mean':
            self.perturb_total_mean(timeseries_instance, start_idx, end_idx)
        elif self.perturbation == 'noise':
            self.perturb_noise(timeseries_instance, start_idx, end_idx)
        return timeseries_instance

    

    def perturb_occlusion(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = 0
        else:
            timeseries_instance[start_idx:end_idx] = 0
        return timeseries_instance


    def perturb_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        else:
            timeseries_instance[start_idx:end_idx] = np.mean(timeseries_instance[start_idx, end_idx])
        return timeseries_instance


    def perturb_total_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        else:
            timeseries_instance[start_idx:end_idx] = np.mean(timeseries_instance[start_idx:end_idx])
        return timeseries_instance


    def perturb_noise(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.random.uniform(timeseries_instance.min(), 
                timeseries_instance.max(), 1)
        else:
            for idx in range(start_idx, end_idx):
                timeseries_instance[idx] = np.random.uniform(timeseries_instance.min(), 
                    timeseries_instance.max(), 1)
        return timeseries_instance




# exlpainer = OcclusionSensitivityUTS()

# arr = np.array(range(20))

# exlpainer.explain_instance(arr, None, None, 2)
