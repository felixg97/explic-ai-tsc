"""

Inspired by https://github.com/ajsanjoaquin/mPerturb
            https://github.com/jacobgil/pytorch-explain-black-box/

"""
# Kurz postponed weil ich gerade nich den unterschied zu rise sehe
#
#
#
# Meaningful Perturbations
#  
### Meaningful perturabtions
# 
# Since we do not have access to the instance generation process
# we consider three obvious proxies replacing the region R with:
# Constant value, Injection noise, Blurring the instance (image)
#
# Let m: Lambda --> [0, 1] be a mask associating each pixel u elmt
# of Lambda with a scalar value m(u). Then the perturbation operator
# is defined as
#
#                     { m(u)x[0](u) + (1 - m(u))mü[0], constant
# [psi(x[0]; m)](u) = { m(u)x[0](u) + (1 - m(u))eta[0], noise
#                     { Integral(g[sigma m[u]](v - u)x[0](v)dv), blur
#
#
#### Deletion & Preservation
## 1.1) deletion game:
#  
# Find the smallest deletion mask m that causes
# the score fc(psi(x[0];m)) << f[c](x[0]) (significantly lower)
#
## 1.2) preservation game
#
# Find the smallest subset of the image that must be retained
# to preserve the score f[c](psi(x[0],m)) >= f[c](x[0]): m*
#
## 2) Obtain a mask more respresentative of natural perturbations
# 
# Regularize m in (!) total-variation norm (!) and upsampling it
# from low resolution
# 
# 
#  

import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

class MeaningfulPerturbationUTS:
    
    def __init__(self):
        pass

    def explain(self, timeseries_data, y_true, model):

        _timeseries_data = np.array(timeseries_data)
        _y_true = np.array(y_true)

        if _timeseries_data.shape != _y_true.shape:
            raise Exception('Time series data and labels are not of the same shape')

        explanations = [
            self.explain_instance(_timeseries_data[i], y_true[i], model)
            for i in range(_timeseries_data.shape[0])
        ]

    def explain_instance(
        self, 
        timeseries_instance, 
        y_true, 
        model,
        learning_rate=1e-1,
        l1_lambda=1e-4,
        tv_lambda=1e-2,
        tv_beta=3,
        blurr_size=11,
        blurr_sigma=10,
        mask_size=28, 
        noise_std=0,
        less_lambda=0,
        optimizer='adam',
        epochs=300,
        interpolation_method='fourier'
        ):

        # mask_size -> 28, rise -> s = 8

        _timeseries_instance = np.array(timeseries_instance.copy())
        _y_true = np.array(y_true)

        ########### init mask ###########
        # sized 
        mask_init = 0.5*np.ones((mask_size), dtype = np.float32)


        ########### convert to variable ###########
        var1 = tf.Variable(10.0)


        ########### load optimizer ###########
        # beta1 = 0.9 # default
        # beta2 = 0.999 # default 
        epsilon = 1e-8 # default
        optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon)


        ########### ich glaub das hier brauch ich alles nich lul -> classifier
        # layer -> fc3 -> ???
        target_length = len(timeseries_instance) # hier out_keys rein ?
        # target softmax layer of model -> ich glaub hier wird der classifier (AlexNet) trained

        masks = [ ]
        ########### learn optimal mask ###########
        for i in range(epochs):
            # optimizer steps

            ########### upsample mask and cropping ###########
            mask = self._interpolate(mask_init, target_length, interpolation_method=interpolation_method)

            ########### perturb the time series ###########
            pertubed_timeseries = None

            ########### predict perturbed time series ###########
            predictions = model.predict_input(pertubed_timeseries, y_true)
            outputs = tf.nn.softmax()

            ########### determine losses ###########
            l1_loss = l1_lambda * 0
            tv_loss = tv_lambda * self._tv_norm(mask, tv_beta)
            less_loss = less_lambda * self._min_norm(predictions, target_length)
            class_loss = None
            total_loss = None

            ########### do optimizer step ###########
            step_count = optimizer.minimize(total_loss)

            pass


        print()
        print()
        print()
        return np.array([])



    def _tv_norm(self, mask, tv_beta):
        pass

    def _min_norm(self, mask):
        pass

    
    ## upsampling and interpolation 
    def _interpolate(self, array1d, out_shape, interpolation_method='fourier'):
        if interpolation_method == 'linear':
            factor = len(array1d) / out_shape
            n = int(np.ceil(len(array1d) / factor))
            f = interp1d(np.linspace(0, 1, len(array1d)), array1d, 'linear')
            return f(np.linspace(0, 1, n))

            # return np.interp(array1d, )
        elif interpolation_method == 'fourier':
            return signal.resample(array1d, out_shape, axis=0)
