"""
resnet.py

ResNet proposed by Wang et al. 2017

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
https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

It is partially rewritten.
"""
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import calculate_metrics, save_logs
from utils.utils import save_test_duration


# class ResNet:
    
#     def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
#         self.output_directory = output_directory

#         ## build routine
#         if build == True:
#             self.model = self.build(input_shape, nb_classes)

#             if verbose == True:
#                 self.model.summary()
#             self.verbose = verbose

#             if load_weights == True:
#                 self.model.load_weights(self.output_directory
#                                         .replace('resnet_augment', 'resnet')
#                                         .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
#                                         + '/model_init.hdf5')
#             else:
#                 self.model.save_weights('{self.output_directory}model_init.hdf5')


#     def build(self, input_shape, nb_classes):
#         """
#         https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
#         """
#         n_feature_maps = 64

#         ## input layer
#         input_layer = keras.layers.Input(input_shape)

#         ## block 1
#         conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
#         conv_x = keras.layers.BatchNormalization()(conv_x)
#         conv_x = keras.layers.Activation('relu')(conv_x)

#         conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
#         conv_y = keras.layers.BatchNormalization()(conv_y)
#         conv_y = keras.layers.Activation('relu')(conv_y)

#         conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
#         conv_z = keras.layers.BatchNormalization()(conv_z)

#         # expand channels for the sum
#         shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
#         shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

#         output_block_1 = keras.layers.add([shortcut_y, conv_z])
#         output_block_1 = keras.layers.Activation('relu')(output_block_1)

#         ## block 2
#         conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
#         conv_x = keras.layers.BatchNormalization()(conv_x)
#         conv_x = keras.layers.Activation('relu')(conv_x)

#         conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
#         conv_y = keras.layers.BatchNormalization()(conv_y)
#         conv_y = keras.layers.Activation('relu')(conv_y)

#         conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
#         conv_z = keras.layers.BatchNormalization()(conv_z)

#         # expand channels for the sum
#         shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
#         shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

#         output_block_2 = keras.layers.add([shortcut_y, conv_z])
#         output_block_2 = keras.layers.Activation('relu')(output_block_2)

#         ## block 3
#         conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
#         conv_x = keras.layers.BatchNormalization()(conv_x)
#         conv_x = keras.layers.Activation('relu')(conv_x)

#         conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
#         conv_y = keras.layers.BatchNormalization()(conv_y)
#         conv_y = keras.layers.Activation('relu')(conv_y)

#         conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
#         conv_z = keras.layers.BatchNormalization()(conv_z)

#         # no need to expand channels because they are equal
#         shortcut_y = keras.layers.BatchNormalization()(output_block_2)

#         output_block_3 = keras.layers.add([shortcut_y, conv_z])
#         output_block_3 = keras.layers.Activation('relu')(output_block_3)

#         ## gap layer    
#         gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

#         ## output layer
#         output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

#         ## model
#         model = keras.models.Model(inputs=input_layer, outputs=output_layer)

#         model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
#             metrics=['accuracy'])

#         reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

#         ## file path to save model
#         file_path = f'{self.output_directory}best_model.hdf5'

#         ## model checkpoint (to save model)
#         model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
# 			save_best_only=True)

#         self.callbacks = [reduce_lr, model_checkpoint]

#         return model


#     def fit(self, x_train, y_train, x_test, y_test, y_true):
#         ## check if tensorflow-gpu is ready and up (for training)
#         # (skipping for yet)

#         ## batch_size/minibatch_size
#         batch_size = 64

#         ## nb_epochs
#         nb_epochs = 1500

#         mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

#         ## measure training time
#         start_time = time.time()

#         ## fit model to training data (keras History object)
#         hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
#             verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

#         ## end of training time
#         duration = time.time() - start_time

#         ## save model as last_model as .hdf5
#         self.model.save(f'{self.output_directory}last_model.hdf5')

#         ## predict test data
#         y_pred = self.predict(x_train, y_train, x_test, x_test, y_true, return_df_metrics=False)

#         ## save predictions as .npy (fast read in comp. to csv)
#         np.save(f'{self.output_directory}y_pred.npy', y_pred)

#         ## addtionally save predictions as .csv (if true)
#         if False:
#             np.savetxt(f'{self.output_directory}y_pred.csv', y_pred, delimiter=',')

#         ## convert predictions from binary to integer
#         y_pred = np.argmax(y_pred, axis=1)

#         ## calulate metrics
#         df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

#         ## clear keras session
#         keras.backend.clear_session()

#         ##
#         return df_metrics


    
#     def predict(self, x_train, y_train, x_test, y_test, y_true, return_df_metrics=True):
#         ## measure prediction time
#         start_time = time.time()

#         ## model path to load model
#         model_path = f'{self.output_directory}+best_model.hdf5'

#         ## load model
#         model = keras.models.load_model(model_path)

#         ## predict test data
#         y_pred = model.predict(x_test)

#         if return_df_metrics:
#             ## converts predictions from binary to integer
#             y_pred = np.argmax(y_pred, axis=1) 

#             ## calculate metrics
#             df_metrics = calculate_metrics(y_true, y_pred, (time.time() - start_time))
#         else:
#             duration = time.time() - start_time
#             save_test_duration(f'{self.output_directory}', duration)
#             return y_pred

################################################################################

class ResNet:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.model_name = 'resnet'
        self.output_directory = output_directory
        
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration, model_name=self.model_name)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
