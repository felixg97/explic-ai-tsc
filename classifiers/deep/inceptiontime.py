"""
inceptiontime.py

InceptionTime proposed by Fawaz et al. 2020. (Szegedy et al. 2015)

bibtex:
@article{IsmailFawaz2020,
  doi = {10.1007/s10618-020-00710-y},
  url = {https://doi.org/10.1007/s10618-020-00710-y},
  year = {2020},
  month = sep,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {34},
  number = {6},
  pages = {1936--1962},
  author = {Hassan Ismail Fawaz and Benjamin Lucas and Germain Forestier and Charlotte Pelletier and Daniel F. Schmidt and Jonathan Weber and Geoffrey I. Webb and Lhassane Idoumghar and Pierre-Alain Muller and Fran{\c{c}}ois Petitjean},
  title = {{InceptionTime}: Finding {AlexNet} for time series classification},
  journal = {Data Mining and Knowledge Discovery}
}

This Script has been provided by:
https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/inception.py

It is partially rewritten.
"""
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import calculate_metrics, save_logs
from utils.utils import save_test_duration

class InceptionTime(object):
    
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, 
        build=True, batch_size=64, lr=.001, nb_filters=32, use_residual=True,
        use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        ## build routine
        if build == True:
            self.model = self.build(input_shape, nb_classes)

            if verbose:
                self.model.summary()
            self.verbose = self.verbose

            if load_weights:
                self.model.load_weights(f'{self.output_directory}model_init.hdf5')
            else:
                self.model.save_weights('{self.output_directory}model_init.hdf5')



    def build(self, input_shape, nb_classes):

        ## input layer
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        ## block 1
        for d in range(self.depth):
            x = self._inception_module(x)
            
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        ## gap layer
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        ## output layer
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        ## model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
            min_lr=0.0001)

        ## file path to save model
        file_path = f'{self.output_directory}best_model.hdf5'

        ## model checkpoint (to save model)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, 
            monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]
        
        return model



    def fit(self, x_train, y_train, x_test, y_test, y_true):
        ## check if tensorflow-gpu is ready and up (for training)
        # (skipping for yet)

        ## batch_size/minibatch_site
        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        ## measure training time
        start_time = time.time()

        ## fit model to training data (keras History object)
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, 
            epochs=self.nb_epochs, verbose=self.verbose, validation_data=(x_val, y_val), 
            callbacks=self.callbacks)

        ## end of training time
        duration = time.time() - start_time

        ## save model as last_model as .hdf5
        self.model.save(f'{self.output_directory}last_model.hdf5')

        ## predict test data
        y_pred = self.predict(x_train, y_train, x_test, x_test, y_true, return_df_metrics=False)

        ## save predictions as .npy (fast read in comp. to csv)
        np.save(f'{self.output_directory}y_pred.npy', y_pred)

        ## addtionally save predictions as .csv (if true)
        if False:
            np.savetxt(f'{self.output_directory}y_pred.csv', y_pred, delimiter=',')

        ## convert predictions from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        ## calulate metrics
        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        ## clear keras session
        keras.backend.clear_session()

        ##
        return df_metrics


    
    def predict(self, x_train, y_train, x_test, y_test, y_true, return_df_metrics=True):
        ## measure prediction time
        start_time = time.time()

        ## model path to load model
        model_path = f'{self.output_directory}+best_model.hdf5'

        ## load model
        model = keras.models.load_model(model_path)

        ## predict test data
        y_pred = model.predict(x_test)

        if return_df_metrics:
            ## converts predictions from binary to integer
            y_pred = np.argmax(y_pred, axis=1) 

            ## calculate metrics
            df_metrics = calculate_metrics(y_true, y_pred, time.time(), 0.0)
        else:
            duration = time.time() - start_time
            save_test_duration(f'{self.output_directory}', duration)
            return y_pred



    def _inception_module(self, in_tensor, stride=1, activation='linear'):
        
        if self.use_bottleneck and int(intput_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                padding='same', activation=activation, use_bias=False)(in_tensor)
        else:
            input_inception = in_tensor

        ## kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []
        
        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, 
                kernel_size=kernel_size_s[i], strides=stride, padding='same', 
                activation=activation, use_bias=False)(input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, 
            padding='same')(in_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
            padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)

        return x

    def _shortcut_layer(self, in_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]),
            kernel_size=1, padding='same', use_bias=False)(in_tensor)

        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)

        return x
