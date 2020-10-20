
import os
import argparse
import ast

import pandas as pd
import numpy as np
import tarfile

from sklearn.externals import joblib

if __name__=="__main__":
    model_path = os.path.join('/opt/ml/processing/model', 'model.tar.gz')
    print('Extracting model from path: {}'.format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')
    print('Loading model')
    model = joblib.load('model.joblib')
    
    print('Loading test input data')
    dfp_toscore = pd.read_csv(os.path.join('/opt/ml/processing/score', 'dataset_score.csv'))
    dfp_toscore['features'] = dfp_toscore['features'].apply(lambda features: ast.literal_eval(features))
    
    dfp_toscore['probabilities'] = list(model.predict_proba(dfp_toscore['features'].tolist()))
    dfp_toscore['predictions'] = dfp_toscore['probabilities'].apply(lambda prediction_prob: np.argsort(prediction_prob)[::-1])
    
    labels = list(model.classes_)
    dfp_toscore['predictions'] = dfp_toscore['predictions'].apply(lambda predictions: [labels[idx] for idx in predictions])
    
    evaluation_output_path = os.path.join('/opt/ml/processing/predictions', 'predictions.csv')
    dfp_toscore.to_csv(evaluation_output_path, index=None)
    print('DONE')
