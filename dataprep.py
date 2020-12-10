"""
Author: Ivan Bongiorni
2020-08-17
Data preprocessing pipeline. Separated from model implementation and training.
"""
import os
import time
from pdb import set_trace as BP

import numpy as np
import pandas as pd

import tools  # local import


def processing_main():
    '''
    Main wrapper of the input pipe. Steps:
    1. Loads pandas dataframe, takes values only and converts to np.array
    2. Fills NaN's on the left with zeros, keeps NaN's within trends
    3. Extracts trends with NaN's and pickles them in /data/ folder, leaving
       a dataset of complete trends for training
    4. Shuffles dataset and operates Train-Validation-Test split based on params
    '''
    import os
    import time
    import yaml
    import pickle
    import numpy as np
    import pandas as pd

    pipeline_start = time.time()

    print('\nStart data processing pipeline.\n')
    print('\tLoading data and configuration parameters.')
    df = pd.read_csv(os.getcwd() + '/data_raw/train_2.csv')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    print('\tAdding imputed trends to dataset.')
    df.dropna(axis=0, inplace=True)

    # Check presence of imputed sub-df, in that case subst. else: impute 0 in all NaN's
    if 'imputed.csv' in os.listdir(os.getcwd()+'/data_raw/'):
        imputed = pd.read_csv(os.getcwd()+'/data_raw/imputed.csv')
        df = pd.concat([df, imputed], ignore_index=True)
        del imputed # free memory
    else:
        df.fillna(0, inplace=True)

    print('\tExtracting URL metadata from dataframe.')
    page_vars = [ tools.process_url(url) for url in df['Page'].tolist() ]
    page_vars = pd.DataFrame(page_vars)
    df.drop('Page', axis=1, inplace=True)

    # One-Hot encode, and leave one out to reduce matrix sparsity
    page_vars = pd.get_dummies(page_vars)
    page_vars.drop(['language_na', 'website_mediawiki', 'access_desktop', 'agent_spider'], axis=1, inplace=True)

    weekdays, yeardays = tools.get_time_schema(df)  # get fixed time variables

    df = df.values
    page_vars = page_vars.values

    print('\tScaling data.')
    # Find int threeshold between Train lenght and Val+Test in main df
    train_val_threshold = df.shape[1]-params['len_prediction'] - int((df.shape[1]-params['len_prediction']) * params['val_size'])
    df, scaling_percentile = tools.scale_trends(array=df, threshold=train_val_threshold)
    # Save scaling params to file
    scaling_dict = {'percentile': float(scaling_percentile)}
    yaml.dump(scaling_dict, open( os.getcwd() + '/data_processed/scaling_dict.yaml', 'w'))

    print('\tStart processing observations.')
    # Apply sequence of processing transformations and save to folder

    for i in range(df.shape[0]):
        array = tools.apply_processing_transformations(
            trend = df[i,:],
            vars = page_vars[i,:],
            weekdays = weekdays,
            yeardays = yeardays
        )

        X_train = array[ :-params['len_prediction'] , : ]
        X_test = array[ -(params['len_input']+params['len_prediction']): , : ]

        np.save(os.getcwd() + '/data_processed/Train/X_{}'.format(str(i).zfill(6)), X_train)
        np.save(os.getcwd() + '/data_processed/Test/X_{}'.format(str(i).zfill(6)), X_test)

    print('\tSaved {} Training observations.'.format(df.shape[0]))

    print('\n\tPipeline executed in {} ss.\n'.format(round(time.time()-pipeline_start, 2)))
    return None


if __name__ == '__main__':
    processing_main()
