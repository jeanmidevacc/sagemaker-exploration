#!/usr/bin/env python
# coding: utf-8

# In[59]:


import boto3
import ast
import itertools
from collections import Counter 
import os
import time
from datetime import datetime
import json

import pandas as pd
import numpy as np
import math as mth

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

pd.set_option('mode.chained_assignment', None)


# ## Process

# In[29]:


tic_process = time.time()


# In[30]:


# load the data
dfp_ml = pd.read_csv(f'{os.environ["AWS_SAGEMAKER_S3_LOCATION"]}/data/dataset_ml.csv')

# wrokin on sample for the scoring set
dfp_toscore = pd.read_csv(f'{os.environ["AWS_SAGEMAKER_S3_LOCATION"]}/data/dataset_toscore.csv').sample(frac=0.1).head(1000)


# In[31]:


dfp_ml['deck'] = dfp_ml['cards'].apply(lambda cards:ast.literal_eval(cards))


# In[32]:


# Build the training set and the testing set
dfp_train, dfp_test = train_test_split(dfp_ml, test_size=0.20, random_state=0)
dfp_train.reset_index(drop=True,inplace=True)
dfp_test.reset_index(drop=True,inplace=True)


# In[33]:


# Estimate the popularcard for a specific archetype
def compute_cards_count(decks):
    cards_count = {}
    for deck in decks:
        cards = ast.literal_eval(deck)
        for card in cards:
            if card in cards_count:
                cards_count[card] += 1
            else:
                cards_count[card] = 1    
    return cards_count

def compute_cards_count_v2(decks):
    return dict(Counter(list(itertools.chain.from_iterable(decks))))

def compute_ranking_cards(cards_count,k=5):
    dfp_ranking = pd.DataFrame.from_dict(cards_count, orient='index',columns=['count']).sort_values('count', ascending=False)
    return dfp_ranking['count'].tolist()[:k]

dfp_train_agg = dfp_train.groupby(['archetype'])['deck'].apply(list).to_frame()
dfp_train_agg['cards_count'] = dfp_train_agg['deck'].apply(lambda deck: compute_cards_count_v2(deck))
dfp_train_agg['cards_selection'] = dfp_train_agg['cards_count'].apply(lambda cards_count: compute_ranking_cards(cards_count))

cards_selection = list(dict.fromkeys(list(itertools.chain.from_iterable(dfp_train_agg['cards_selection'].tolist()))))


# In[34]:


# Encode the deck
def encode_deck(deck, cards_selection):
    encoded_deck = [0] * len(cards_selection)
    
    for card in deck:
        if card in cards_selection:
            idx = cards_selection.index(card)
            encoded_deck[idx] += 1
    return encoded_deck

dfp_train['encoded_deck'] = dfp_train['deck'].apply(lambda deck: encode_deck(deck, cards_selection))
dfp_test['encoded_deck'] = dfp_test['deck'].apply(lambda deck: encode_deck(deck, cards_selection))


# In[35]:


# build an indexer to encode the hero
encoder_hero = preprocessing.LabelEncoder()

heroes = list(dfp_train['hero'].unique())
encoder_hero.fit(heroes)

dfp_train['encoded_hero'] = encoder_hero.transform(dfp_train['hero'].to_list())
dfp_test['encoded_hero'] = encoder_hero.transform(dfp_test['hero'].to_list())

# build the feature vector
dfp_train['features'] = dfp_train.apply(lambda row: [row['encoded_hero']] + row['encoded_deck'], axis=1)
dfp_test['features'] = dfp_test.apply(lambda row: [row['encoded_hero']] + row['encoded_deck'], axis=1)


# In[36]:


# # Build an indexer for the output to predict
encoder_archetype = preprocessing.LabelEncoder()

archetypes = list(dfp_train['archetype'].unique())
encoder_archetype.fit(archetypes)

dfp_train['label'] = encoder_archetype.transform(dfp_train['archetype'].to_list())
dfp_test['label'] = encoder_archetype.transform(dfp_test['archetype'].to_list())

dfp_train_rtu = dfp_train[['label', 'features']]
dfp_test_rtu = dfp_test[['label', 'features']]


# In[37]:


# prepare the dataset to score
dfp_toscore['deck'] = dfp_toscore['cards'].apply(lambda cards:ast.literal_eval(cards))
dfp_toscore['encoded_deck'] = dfp_toscore['deck'].apply(lambda deck: encode_deck(deck, cards_selection))
dfp_toscore['encoded_hero'] = encoder_hero.transform(dfp_toscore['hero'].to_list())
dfp_toscore['features'] = dfp_toscore.apply(lambda row: [row['encoded_hero']] + row['encoded_deck'], axis=1)


# In[38]:


toc_process = time.time()


# ## Train

# In[39]:


tic_train = time.time()


# In[40]:


model = RandomForestClassifier(max_depth=2, random_state=0).fit(dfp_train_rtu['features'].tolist(), dfp_train_rtu['label'].tolist())


# In[41]:


toc_train = time.time()


# ## Evaluate

# In[42]:


tic_evaluate = time.time()


# In[43]:


def get_precision_at_k(recommendations, item, k=5):
    if item in recommendations[:k]:
        return 1
    return 0

def get_ndcg_at_k(recommendations, item, k=5):
    for idx, elt in enumerate(recommendations[:k]):
        if item == elt:
            return mth.log(2) / mth.log(idx+2)
    return 0


# In[44]:


dfp_test_rtu['prediction'] = dfp_test_rtu['features'].apply(lambda features: model.predict_proba([features])[0])
dfp_test_rtu['prediction'] = dfp_test_rtu['prediction'].apply(lambda prediction: np.argsort(prediction)[::-1])


# In[45]:


metrics = []
for k in [1,3,5]:
    dfp_test_rtu[f'precision_at_{k}'] = dfp_test_rtu.apply(lambda row: get_precision_at_k(row['prediction'], row['label'], k), axis=1)
    metrics.append(f'precision_at_{k}')
    if k > 1:
        dfp_test_rtu[f'ndcg_at_{k}'] = dfp_test_rtu.apply(lambda row: get_ndcg_at_k(row['prediction'], row['label'], k), axis=1)
        metrics.append(f'ndcg_at_{k}')


# In[46]:


dfp_test_rtu[metrics].mean()


# In[47]:


toc_evaluate = time.time()


# ## Score

# In[48]:


tic_score = time.time()


# In[49]:


# score the unknow 
dfp_toscore['prediction_prob'] = dfp_toscore['features'].apply(lambda features: model.predict_proba([features])[0])
dfp_toscore['predictions'] = dfp_toscore['prediction_prob'].apply(lambda prediction: np.argsort(prediction)[::-1])

labels = list(model.classes_)
dfp_toscore['predictions'] = dfp_toscore['predictions'].apply(lambda predictions: [labels[idx] for idx in predictions])


# In[50]:


toc_score = time.time()


# ## Monitor the execution

# In[60]:


execution_date = datetime.utcnow()
dict_run = {
    'execution_date': execution_date.strftime('%Y-%m-%d %H:%M:%S'),
    'time_process' : toc_process - tic_process,
    'time_train' : toc_train - tic_train,
    'time_evaluate' : toc_evaluate - tic_evaluate,
    'time_score' : toc_score - tic_score
}

filename = execution_date.strftime('%Y%m%d_%H-%M-%S')
with open(f'data/{filename}.json', 'w') as fp:
    json.dump(dict_run, fp)


# ## Debug
