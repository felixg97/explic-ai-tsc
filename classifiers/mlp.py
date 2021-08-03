"""
mlp.py

Multi Layer Perception Wang et al. 2017

@inproceedings{Wang2017,
  doi = {10.1109/ijcnn.2017.7966039},
  url = {https://doi.org/10.1109/ijcnn.2017.7966039},
  year = {2017},
  month = may,
  publisher = {{IEEE}},
  author = {Zhiguang Wang and Weizhong Yan and Tim Oates},
  title = {Time series classification from scratch with deep neural networks: A strong baseline},
  booktitle = {2017 International Joint Conference on Neural Networks ({IJCNN})}
}

This Script has been provided by:
https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

It is partially rewritten.
"""
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import calculate_metrics, save_logs
from utils.utils import save_test_duration

class MLP:
    
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis 
        input_layer_flattened = keras.layers.Flatten()(input_layer)
        
        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        file_path = self.output_directory+'best_model.hdf5' 

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val,y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 16
        nb_epochs = 5000

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
        
        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory+'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer 
        y_pred = np.argmax(y_pred , axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

    def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics=True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred


    def predict_input(self, _input, y_true):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)

        # print('_input')
        # print(_input.shape)
        # print(_input[0])

        y_pred = model.predict(_input)
        return y_pred
