"""
rise.py

Inspired by https://github.com/eclique/RISE
"""

#
### Random Masking
#
# (1) Let f: I -> R be a black-box model that for a given input from I produces 
#     scalar confidence score, 
#     TS: I = { I | I : Lambda -> R } of size W: width/timestamps
#     (A = {1, ..., W})
# (2) Let M: Lambda -> {0, 1} be a random binary mask with distribution D
#  
# Consider f(I x M) where x denotes elemt-wise multiplication 
#
# First, the time series is masked by preserving only a subset of pixels
# Then, the confidence score for the masked image is computed by the black box
#
# We define importance of pixel lambda element of Lambda as the expected score
# over all possible masks M conditined on the event that pixel lambda is observed
# e. g. M(lambda) = 1
#
# S[I, f](lambda) = E[m][f(I x M) | M(lambda) = 1] # Score
#
# The score of f(I x M) is high when pixels preserved by mask M are important
#
# 1 - S[I, f](lamba) = Sum[m](f(I x M)P[M = m | M(lambda) = 1])
#
#
#
### Mask generation
# 1) Sample N binary masks of size t (temporal dimension smaller then T (t[n]) complete size)
#    by setting each element to 1 with probability p and to 0 with the remaining probability 
#
# 2) Upsample all masks to size (t+1)C[T] using bilinear interpolation, where C[T] = T/t
#    is the size of the cell in the upsampled mask
#
# 3) Crop areas T with uniformly random indents from (0) up to (C[T])
#
import numpy as np
import pandas as pd

from skimage.transform import resize
from scipy import signal
from scipy.interpolate import interp1d


class RiseUTS:
    
    """
    """
    def __init__(self):
        pass    

    
    def explain(self, timeseries_data, y_true, model, N=200, s=8, p=.5, 
        batch_size=100, interpolation='fourier'):
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
                N=N,
                s=s,
                p=p,
                batch_size=batch_size,
                interpolation=interpolation)
            for idx in range(_timeseries_data.shape[0]) 
        ])
        return explanations


    def explain_instance(self, timeseries_instance, y_true, model,
            N=200, s=8, p=.5, batch_size=100, interpolation='fourier'):
        """Explains instance

        Default values of N, s and p are hyperparameters.

        Args:
            timeseries_instance: instance to be explained
            model: Model
            true_class: class to be explained
            N: N masks
            s: Stride size 
            p: 
        """
        # Generate masks        
        masks = self._generate_masks(timeseries_instance, N, s, p, interpolation)
        predicitons = []
        len_ts = len(timeseries_instance)

        timeseries_instance_tmp = np.array([timeseries_instance.copy()])
        
        ## Matrix multiplication of masks and timeseries
        masked = timeseries_instance * masks

        for i in range(0, N, batch_size):
            predicitons.append(model.predict_input(masked[i:min(i+batch_size, N)], y_true))
        
        predicitons = np.concatenate(predicitons)

        saliency = predicitons.T.dot(masks.reshape(N, -1)).reshape(-1, *[len_ts])
        saliency = saliency / N / p

        if y_true:
            saliency[y_true]
        saliency = np.array(saliency[y_true])

        # print('y_true')
        # print(y_true)
        # print()
        # print('saliency')
        # print(saliency.shape)
        # print(saliency)
        return saliency

    def _generate_masks(self, timeseries_instance, N, s, p, interpolation):
        """Randomly generates binaray masks 
        Arg:
            N: hyperparam, N masks
            s: hyperparam, stride
            p: probability (default - random = .5)
        """
        len_ts = len(timeseries_instance)

        cell_size = np.ceil(np.array(len_ts) / s)
        up_size = (s + 1) * cell_size       
        up_size = int(up_size)

        # print('Y0')
        # print(np.array(len_ts))
        # print(np.array(len_ts) / s)
        # print()

        # print('cell size: ', cell_size)
        # print()
    
        grid = np.random.rand(N, s) < p # produces random binary map M (bool)
        grid = grid.astype('float32') # bool to float

        masks = np.empty((N, *[len_ts]))

        # print('grid')
        # print(grid.shape)
        # print(grid[0])

        # print('Masks')
        # print(masks.shape)
        # print(masks)
        # print()
        # print()
        # print('##################################################################')

        # print(len_ts, type(len_ts))
        # print(up_size, type(up_size))
        # print(up_size)

        # scipy.signal.resample - resample with fourier might be goo
        # here: bilinear resampling for image     
        # print()
        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size)

            # Upsample all masks to size (t+1)C[T] using bilinear interpolation, where C[T] = T/t
            # is the size of the cell in the upsampled mask

            # Upsampling Cropping
            masks[i, :] = self._interpolate(grid[i], up_size, interpolation_method=interpolation)[x:(x+len_ts)]
            
        # print() 
        return masks


    def _interpolate(self, array1d, out_shape, interpolation_method='fourier'):
        if interpolation_method == 'linear':
            factor = len(array1d) / out_shape
            n = int(np.ceil(len(array1d) / factor))
            f = interp1d(np.linspace(0, 1, len(array1d)), array1d, 'linear')
            return f(np.linspace(0, 1, n))

            # return np.interp(array1d, )
        elif interpolation_method == 'fourier':
            return signal.resample(array1d, out_shape, axis=0)
    