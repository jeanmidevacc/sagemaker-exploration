
"""
Script to process the raw data
"""
import os
import argparse
import ast
from collections import Counter 
import itertools

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pandas as pd

pd.set_option('mode.chained_assignment', None)

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

def encode_deck(deck, cards_selection):
    encoded_deck = [0] * len(cards_selection)
    
    for card in deck:
        if card in cards_selection:
            idx = cards_selection.index(card)
            encoded_deck[idx] += 1
    return encoded_deck

def prepare_dataset_to_train(dfp, cards_selection, encoder_hero):
    dfp['encoded_deck'] = dfp['deck'].apply(lambda deck: encode_deck(deck, cards_selection))
    dfp['encoded_hero'] = encoder_hero.transform(dfp['hero'].to_list())
    dfp['features'] = dfp.apply(lambda row: [row['encoded_hero']] + row['encoded_deck'], axis=1)
    dfp = dfp[['archetype', 'features']]
    dfp.columns = ['label', 'features']
    return dfp

def prepare_dataset_to_score(dfp, cards_selection, encoder_hero):
    dfp['encoded_deck'] = dfp['deck'].apply(lambda deck: encode_deck(deck, cards_selection))
    dfp['encoded_hero'] = encoder_hero.transform(dfp['hero'].to_list())
    dfp['features'] = dfp.apply(lambda row: [row['encoded_hero']] + row['encoded_deck'], axis=1)
    dfp = dfp[['deckid', 'features']]
    return dfp

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    
    # Collect the data
    input_data_to_train_path = os.path.join('/opt/ml/processing/input/ml', 'dataset_ml.csv')
    input_data_to_score_path = os.path.join('/opt/ml/processing/input/toscore', 'dataset_toscore.csv')
    print('Reading input data to train from {}'.format(input_data_to_train_path))
    dfp_ml = pd.read_csv(input_data_to_train_path)
    dfp_ml['deck'] = dfp_ml['cards'].apply(lambda cards:ast.literal_eval(cards))
    
    # Build the training set and the testing set
    dfp_train, dfp_test = train_test_split(dfp_ml, test_size=args.test_size, random_state=0)
    dfp_train.reset_index(drop=True,inplace=True)
    dfp_test.reset_index(drop=True,inplace=True)
    
    # Build some encoder
    encoder_hero = preprocessing.LabelEncoder()
    heroes = list(dfp_train['hero'].unique())
    encoder_hero.fit(heroes)
    
    # Rank the cards for the deck encoding
    dfp_train_agg = dfp_train.groupby(['archetype'])['deck'].apply(list).to_frame()
    dfp_train_agg['cards_count'] = dfp_train_agg['deck'].apply(lambda deck: compute_cards_count_v2(deck))
    dfp_train_agg['cards_selection'] = dfp_train_agg['cards_count'].apply(lambda cards_count: compute_ranking_cards(cards_count))
    cards_selection = list(dict.fromkeys(list(itertools.chain.from_iterable(dfp_train_agg['cards_selection'].tolist()))))
    
    dfp_dataset_train = prepare_dataset_to_train(dfp_train, cards_selection, encoder_hero)
    dfp_dataset_test = prepare_dataset_to_train(dfp_test, cards_selection, encoder_hero)
    
    print('Reading input data to score from {}'.format(input_data_to_score_path))
    dfp_score = pd.read_csv(input_data_to_score_path).sample(frac=0.1).head(1000)
    dfp_score['deck'] = dfp_score['cards'].apply(lambda cards:ast.literal_eval(cards))
    dfp_dataset_score = prepare_dataset_to_score(dfp_score, cards_selection, encoder_hero)
    
    # Save the data
    print('Saving the data in /opt/ml/processing/train and test and score')
    train_output_path = os.path.join('/opt/ml/processing/train', 'dataset_train.csv')
    test_output_path = os.path.join('/opt/ml/processing/test', 'dataset_test.csv')
    score_output_path = os.path.join('/opt/ml/processing/score', 'dataset_score.csv')
    dfp_dataset_train.to_csv(train_output_path, index=None)
    dfp_dataset_test.to_csv(test_output_path, index=None)
    dfp_dataset_score.to_csv(score_output_path, index=None)
    print('DONE')
    
