
"""
Script to process the raw data
"""
import os
import argparse
import ast

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import pandas as pd


pd.set_option('mode.chained_assignment', None)


if __name__=="__main__":
    
    training_data_directory = '/opt/ml/input/data/train'
    print(f'Collect the training set data at {training_data_directory}')
    dfp_training = pd.read_csv(training_data_directory + '/dataset_train.csv')
    dfp_training['features'] = dfp_training['features'].apply(lambda features: ast.literal_eval(features))
    
    print('Set the random forest model')
    model = RandomForestClassifier(max_depth=2, random_state=0)
    
    print('Train the model')
    model.fit(dfp_training['features'].tolist(), dfp_training['label'].tolist())
    model_output_directory = os.path.join('/opt/ml/model', "model.joblib")
    
    print('Saving model to {}'.format(model_output_directory))
    joblib.dump(model, model_output_directory)
    print('DONE')
