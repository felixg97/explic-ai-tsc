"""
cote.py

COTE proposed by Bagnall et al. 2016

Bagnall A, Lines J, Hills J, Bostrom A (2016) Time-series classification with COTE: the collective of
transformation-based ensembles. In: International conference on data engineering, pp 1548–1549

(fawaz git maybe)
"""

class COTE(object):
    pass
    
    # def __init__(self, outpt_directory, input_shape, nb_classes, verbose=False, build=True):
    #     self.outpt_directory = outpt_directory

    #     if build == True:
    #         self.model = self.build_model(input_shape, nb_classes)
    #         if(verbose):
    #             self.model.summary()
    #         self.verbose = self.verbose
    #         self.model.save_weights('{self.output_directory}model_init.hdf5')


    # def build_model(self, input_shape, nb_classes):
    #     ## model layers
    #     input_layer = keras.layers.Input(input_shape)


    #     ouput_layer = keras.layers.Dense(nb_classes, activation='softmax')(None) ## muss natürlich dann konfiguriert sein

    #     ## model
    #     model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    #     # model.compile()

    #     reduce_lr = None

    #     file_path = f'{self.output_directory}best_model.hdf5'

    #     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
    #         save_best_only=True)

    #     self.callbacks = [reduce_lr,model_checkpoint]

    #     return model


    # def fit(self, x_train, y_train):
    #     ## batch_size / minibatch_size
    #     batch_size = 0, minibatch_size = 0

    #     ## nb_epochs
    #     nb_epochs = 0

    #     start_time = time.time()
        
    #     ## hist = self.model.train()
    #     hist = None

    #     duration = time.time() - start_time

    #     self.model.save(f'{self.output_directory}last_model.hdf5')

    #     model = keras.models.load_model(f'{self.output_directory}best_model.hdf5')

    #     y_pred = model.predict(x_val)
    #     y_pred = np.argmax(y_pred, axis=1) # convert from binary to integer

    #     # save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

    #     keras.backend.clear_session()

    
    # def predict(self, x_test, y_test):
    #     model_path = f'{self.output_directory}best_model.hdf5'
    #     model = keras.models.load_model(model_path)
    #     y_pred = model.predict(x_test)

    #     # ABC
    #     if return_df_metrics:
    #         y_pred = np.argmax(y_pred, axis=1)
    #         df_metrics = calculate_metrics(y_true, y_pred, 0.0)
    #         return df_metrics
    #     else:
    #         return y_pred