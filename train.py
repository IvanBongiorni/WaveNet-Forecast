"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-08-17

Calls training functions
"""
import numpy as np
import tensorflow as tf

from pdb import set_trace as BP

# local imports
import tools, model


def SMAPE(y_true, y_pred):
    '''
    SMAPE (Symmetric Mean Absolute Percentage Error) is the error metric used in
    the official Kaggle competition. I will use it on Validation data.
    [ modified from: 'https://www.kaggle.com/cpmpml/smape-weirdness' ]
    '''
    import numpy as np

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0

    return np.mean(diff)


def train(model, params):
    '''
    This is pretty straigthforward.
    Function starts by loading an array of file names from /data_processed/Train/
    subdir, to index training observations. At each iteration, an observation (still
    2D arrays) is loaded and processed to 3D array for RNN input. This processed
    array is sampled to 'batch_size' size. A slice of each batch is taken, either
    at training and validation steps.
    An Autograph training function is called later to compute loss and update weights.
    Every k (100) iterations, performance on Validation data is printed.
    '''
    import os
    import pickle
    import time
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    # Depending on 'model_type' selected, loads a different batch processing fn
    # Checks of model_type correctness already happened in model.py
    if params['model_type'] == 1:
        from tools import get_processed_batch_for_regressor as get_processed_batch
    if params['model_type'] == 2:
        from tools import get_processed_batch_for_seq2seq as get_processed_batch

    MSE = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    @tf.function
    def train_on_batch(X_batch, Y_batch):
        with tf.GradientTape() as tape:
            # current_loss = tf.reduce_mean(tf.math.abs(model(X_batch) - Y_batch))
            current_loss = MSE(model(X_batch), Y_batch)
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss

    # Get list of all Training and Validation observations
    X_files = os.listdir(os.getcwd() + '/data_processed/Train/')
    if 'readme_train.md' in X_files: X_files.remove('readme_train.md')
    if '.gitignore' in X_files: X_files.remove('.gitignore')
    X_files = np.array(X_files)

    for epoch in range(params['n_epochs']):
        # Shuffle data by shuffling filename index
        if params['shuffle']:
            X_files = X_files[ np.random.choice(X_files.shape[0], X_files.shape[0], replace=False) ]

        for iteration in range(X_files.shape[0] // params['batch_size']):

            # Take batch of filenames and load list of np.arrays
            start = iteration * params['batch_size']
            batch = [ np.load('{}/data_processed/Train/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in X_files[start:start+params['batch_size']] ]

            # keep Train data - discard Validation for other tests
            batch = [ array[ :-int(len(array)*params['val_size']) , : ] for array in batch ]

            # Process each file and return list [x,y]
            batch = [ get_processed_batch(array, params) for array in batch ]

            # Extract X and Y and stack them in the final batches
            X_batch = [array[0] for array in batch]
            Y_batch = [array[1] for array in batch]
            X_batch = np.concatenate(X_batch)
            Y_batch = np.concatenate(Y_batch)

            # Train model
            train_loss = train_on_batch(X_batch, Y_batch)

            # Once in a while check and print progress on Validation data
            if iteration % 50 == 0:

                train_loss_smape = SMAPE(model.predict(X_batch), Y_batch)

                # Repeat loading but keep Validation data this time
                batch = [ np.load('{}/data_processed/Train/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in X_files[start:start+params['batch_size']] ]
                batch = [ array[ :-(int(len(array)*params['val_size'])+params['len_input']) , : ] for array in batch ]

                batch = [ get_processed_batch(array, params) for array in batch ]
                X_batch = [array[0] for array in batch]
                Y_batch = [array[1] for array in batch]
                X_batch = np.concatenate(X_batch)
                Y_batch = np.concatenate(Y_batch)

                # validation_loss = tf.reduce_mean(tf.math.abs(model(X_batch) - Y_batch))
                validation_loss_mse = MSE(model(X_batch), Y_batch)
                validation_loss_smape = SMAPE(model.predict(X_batch), Y_batch)

                print('{}.{} \tTraining Loss (MSE):   {}    \tTraining Loss (SMAPE):   {}'.format(
                    epoch, iteration, train_loss, train_loss_smape))
                print('\tValidation Loss (MSE): {}   \tValidation Loss (SMAPE): {}\n'.format(
                    validation_loss_mse, validation_loss_smape))

    print('\nTraining complete.\n')

    model.save('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name']))
    print('Model saved at:\n{}'.format('{}/saved_models/{}.h5'.format(os.getcwd(), params['model_name'])))

    return None


def main():
    ''' Wrapper of training pipeline. '''
    import os
    import yaml
    import tensorflow as tf
    # local imports
    import model
    import tools

    print('\nStart training pipeline.')

    print('\n\tLoading configuration parameters.')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    tools.set_gpu_configurations(params)

    # Check if pretrained model with 'model_name' exists, otherwise create a new one
    if params['model_name']+'.h5' in os.listdir(os.getcwd()+'/saved_models/'):
        print('Loading existing model: {}.'.format(params['model_name']))
        ANN = tf.keras.models.load_model(os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')
    else:
        print('\nNew model created as: {}\n'.format(params['model_name']))
        ANN = model.build(params)

    ANN.summary()

    print('\nStart training.\n')
    train(ANN, params)

    return None


if __name__ == '__main__':
    main()
