"""
Occlusion.py

Perturbation-based explanation "Occlusion" of Zeiler and Fergus 2014

Inspired by https://github.com/sicara/tf-explain

This Explanation is just for univariate Time Series Explanations
"""

class OcclusionSensitivityUTS(object):        

    """
    Performs occlusion sensitivity on univariate time series
    """

    def __init__(self, perturbation='occlusion'):
        self.perturbation = perturbation

    
    def explain_instance(
        self, 
        timeseries_instance,
        true_class,
        model, 
        patch_size=1,
    ):
        """
        Computes Occlusion sensitity maps for a specific time_series,
        
        Args:
            validation_data:
            model(tf.keras.Model): model to inspect 
            patch_size(int): Size of patch to apply on the image
        Returns:
            np.array: 2d array: ()
        """


        return 
    

    def perturb_occlusion(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = 0
            return
        timeseries_instance[start_idx:end_idx] = 0


    def perturb_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        timeseries_instance[start_idx:end_idx] = np.mean(timeseries_instance[start_idx, end_idx])


    def perturb_total_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
            return
        timeseries_instance[start_idx:end_idx] = np.mean(timeseries_instance)


    def perturb_noise(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.random.uniform(timeseries_instance.min(), 
                timeseries_instance.max(), 1)
            return
        for idx in range(start_idx, end_idx):
            timeseries_instance[idx] = np.random.uniform(timeseries_instance.min(), 
                timeseries_instance.max(), 1)
            pass


import numpy as np

exlpainer = OcclusionSensitivityUTS()

arr = np.array([1,2,3,4,5,6,7,8])

exlpainer.perturb_noise(arr, 2, 5)

print(arr)