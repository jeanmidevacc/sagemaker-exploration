
"""
Script to process the raw data
"""
import os
import argparse
import ast

import numpy as np
import math as mth
import pandas as pd
import tarfile

from sklearn.externals import joblib

pd.set_option('mode.chained_assignment', None)

def get_precision_at_k(recommendations, item, k=5):
    if item in recommendations[:k]:
        return 1
    return 0

def get_ndcg_at_k(recommendations, item, k=5):
    for idx, elt in enumerate(recommendations[:k]):
        if item == elt:
            return mth.log(2) / mth.log(idx+2)
    return 0

if __name__=="__main__":
    model_path = os.path.join('/opt/ml/processing/model', 'model.tar.gz')
    print('Extracting model from path: {}'.format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')
    print('Loading model')
    model = joblib.load('model.joblib')
    
    print('Loading test input data')
    dfp_test = pd.read_csv(os.path.join('/opt/ml/processing/test', 'dataset_test.csv'))
    dfp_test['features'] = dfp_test['features'].apply(lambda features: ast.literal_eval(features))
    
    dfp_test['prediction'] = list(model.predict_proba(dfp_test['features'].tolist()))
    dfp_test['prediction'] = dfp_test['prediction'].apply(lambda prediction: np.argsort(prediction)[::-1])
    
    labels = list(model.classes_)
    dfp_test['prediction'] = dfp_test['prediction'].apply(lambda prediction: [labels[idx] for idx in prediction])

    metrics = []
    for k in [1,3,5]:
        dfp_test[f'precision_at_{k}'] = dfp_test.apply(lambda row: get_precision_at_k(row['prediction'], row['label'], k), axis=1)
        metrics.append(f'precision_at_{k}')
        if k > 1:
            dfp_test[f'ndcg_at_{k}'] = dfp_test.apply(lambda row: get_ndcg_at_k(row['prediction'], row['label'], k), axis=1)
            metrics.append(f'ndcg_at_{k}')
    dfp_metrics = dfp_test[metrics].mean().to_frame().reset_index()
    dfp_metrics.columns = ['metric', 'value']
    evaluation_output_path = os.path.join('/opt/ml/processing/evaluation', 'metrics.csv')
    dfp_metrics.to_csv(evaluation_output_path, index=None)
    print('DONE')
