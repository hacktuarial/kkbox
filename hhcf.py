# Compare two ways of estimating user- and item- bias factors
# 1. One hot encode them and put them into an xgboost model with some other features
# 2. Use `diamond` to fit a crossed random-effects model, then put these predictions into an xgboost model with the same set of other features
# 
# These other features include
# * genre encoding: one song can be in multiple genres
# * Categorical features
#     * source_system_tab
#     * source_screen_name
#     * source_type 
#     * artist_name
#     * composer
#     * lyricist
#     * language
#     * city
#     * gender
#     * registered_via
# * Numeric features
#     * song_length
#     * bd
#     * registration_init_time
#     * expiration_date

import numpy as np
import pandas as pd
import pickle
import click

from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

from diamond.glms.logistic import LogisticRegression
import xgboost as xgb
import utils
import dill
import os

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO)


class ExtraInfo(object):
    """ holds info about songs or members for making recs """
    def __init__(self, df):
        self.df = df
        self.encoding = None
        self.id_map = None

    def create_encoding(self, cat_features, id_col):
        """
        one hot encode categorical features. modifies `self.df` in-place
        """
        idx = id_col + 'x'
        for x in cat_features:
            self.df[x] = self.df[x].astype(str).fillna('unknown')
        self.id_map = id_to_index(self.df[id_col])
        self.df[idx] = self.df[id_col].apply(lambda x: self.id_map[x])
        cats = self.df[cat_features + [idx]].sort_values(idx)
        self.X = DictVectorizer().fit_transform(cats.T.to_dict().values())

    def save(self, f):
        with open(f, 'wb') as ff:
            dill.dump(self, ff)

    def load(self, f):
        with open(f, 'rb') as ff:
            self.__dict__.update(dill.load(ff).__dict__)


def id_to_index(ids):
    " return a dict like {id: idx} "
    unique_ids = set(ids)
    return dict(zip(unique_ids, np.arange(0, len(unique_ids))))


def encode_genres(song_map, df_songs):
    " this is a many-hot encoding of song to genre "
    path = 'data/processed/genres.dill'
    if os.path.exists(path):
        logging.info('reading genre encodings from disk')
        with open(path, 'rb') as ff:
            X_genres = dill.load(ff)
    else:
        logging.info('creating genre encodings')
        df_genres = df_songs['genre_ids'].apply(lambda s: pd.Series(str(s).split('|')))
        df_genres['song_idx'] = df_songs['song_id'].apply(lambda x: song_map[x])
        df_genres2 = pd.melt(df_genres, 'song_idx', value_name='genre').\
            drop('variable', axis=1).\
            dropna().\
            sort_values('song_idx')
        genre_map = id_to_index(df_genres2['genre'])
        df_genres2['genre_idx'] = df_genres2['genre'].apply(lambda g: genre_map[g])
        X_genres = sparse.coo_matrix((np.ones(len(df_genres2)),
                                     (df_genres2['song_idx'], df_genres2['genre_idx'])))
        X_genres = X_genres.tocsr()
        with open(path, 'wb') as ff:
            dill.dump(X_genres, ff)
    return X_genres


def xgb_params():
    xgb_params = {}
    xgb_params['objective'] = 'binary:logistic'
    xgb_params['eta'] = 0.75
    xgb_params['max_depth'] = 5
    xgb_params['silent'] = 1
    xgb_params['eval_metric'] = 'auc'
    return xgb_params


def merge_it_all_together(df,
                          songs,
                          genres,
                          members,
                          matrix_list,
                          diamond):
    """
    Merge member, song, and source data together
    Return (X, y) tuple
    This makes a _copy_ of df
    """

    # there are 80 interactions with unknown songs in the training set
    # remove them!
    logging.info('merging in song info')
    df = pd.merge(df.drop(SOURCE_FEATURES, axis=1),
                  songs.df[['song_id', 'song_idx', 'song_length']],
                  'inner',
                  'song_id')
    logging.info('merging in member info')
    member_features = ['msno', 'msnox', 'registration_init_time',
                       'expiration_date', 'bd']
    df = pd.merge(df, members.df[member_features], how='left', on='msno')
    n_nulls = df.isnull().sum()
    assert n_nulls.sum() == 0, n_nulls
    logging.info('there are no nulls')

    numeric_features = ['song_length', 'registration_init_time',
                        'expiration_date', 'bd']

    X_numeric = df[numeric_features].as_matrix()
    member_idx = df['msnox']
    song_idx = df['song_idx']
    y = df['target']
    del df  # free up the memory

    logging.info('adding numeric feature matrix')
    matrix_list.append(X_numeric)
    logging.info('creating song matrix')
    matrix_list.append(songs.X[song_idx, :])
    logging.info('creating genre matrix')
    matrix_list.append(genres[song_idx, :])
    logging.info('creating member matrix')
    matrix_list.append(members.X[member_idx, :])
    logging.info('concatenating matrices')
    X = sparse.hstack(matrix_list)
    return X, y


def get_source_encoding(source_features, df):
    path = 'data/processed/source_features.dill'
    if os.path.exists(path):
        logging.info('reading source feature encodings from disk')
        with open(path, 'rb') as ff:
            X_source = dill.load(ff)
    else:
        logging.info('creating source features encoding')
        # TODO save to disk
        LE, OHE = LabelEncoder(), OneHotEncoder()
        X_source = OHE.fit_transform(df[source_features].\
            fillna('unknown').\
            apply(LE.fit_transform))
        with open(path, 'wb') as ff:
            dill.dump(X_source, ff)
    return X_source


MAX_ITS = 200
SOURCE_FEATURES = ['source_system_tab', 'source_screen_name', 'source_type']


@click.command()
@click.option('--diamond/--no-diamond', required=True)
def main(diamond):
    # # 1. Read in data
    logging.info('reading in data')
    df_train = pd.read_csv('data/raw/train.csv')
    if os.path.isfile('data/processed/songs.dill'):
        logging.info('reading song encoding from disk')
        songs = ExtraInfo(df=None)
        songs.load('data/processed/songs.dill')
    else:
        logging.info('creating song encoding')
        songs = ExtraInfo(df=pd.read_csv('data/raw/songs.csv'))
        songs.create_encoding(id_col='song_id',
                              cat_features=['artist_name', 'composer', 'lyricist', 'language'])
        songs.save('data/processed/songs.dill')
    if os.path.isfile('data/processed/members.dill'):
        logging.info('reading member encoding from disk')
        members = ExtraInfo(df=None)
        members.load('data/processed/members.dill')
    else:
        members = ExtraInfo(df=pd.read_csv('data/raw/members.csv'))
        members.create_encoding(id_col='msno',
                                cat_features=['city', 'gender', 'registered_via'])
        members.save('data/processed/members.dill')

    # ### IDs: benchmark for diamond
    if diamond:
        numeric_features = ['diamond_prediction']
        matrix_list = []
        df_train, df_val = train_test_split(df_train)
        diamond_model = fit_diamond_model(df_train)
        df_train['diamond_pred'] = diamond_model.predict(df_train)
        df_val['diamond_pred'] = diamond_model.predict(df_val)
    else:
        logging.info('creating member id encoding')
        X_member_ids = sparse.coo_matrix((np.ones(len(members.df)),
                        (members.df['msnox'], members.df['msnox']))).tocsr()
        logging.info('creating song id encoding')
        X_song_ids = sparse.coo_matrix((np.ones(len(songs.df)),
                                        (songs.df['song_idx'], songs.df['song_idx']))).tocsr()
        logging.info('using one-hot encoding of member, song ids')
        matrix_list = [X_member_ids[member_idx, :],
                       X_song_ids[song_idx, :]]

    # ### Genres
    X_genres = encode_genres(songs.id_map, songs.df)

    # ### Source features
    # Thanks to [Ritchie Ng](http://www.ritchieng.com/machinelearning-one-hot-encoding/) for guidance on using One Hot Encoder
    X_source = get_source_encoding(source_features=SOURCE_FEATURES,
                                   df=df_train)

    logging.info('merging validation data together')
    X_val, y_val = merge_it_all_together(df_val,
                                         diamond=diamond,
                                         members=members,
                                         songs=songs,
                                         matrix_list=matrix_list,
                                         genres=X_genres)
    del df_val
    D_val = xgb.DMatrix(sparse.hstack([X_val, X_source[utils.TTS:, :]]))
    logging.info('merging training data together')
    X_train, y_train = merge_it_all_together(df_train,
                                             diamond=diamond,
                                             members=members,
                                             songs=songs,
                                             genres=X_genres)
    del df_train
    D_train = xgb.DMatrix(sparse.hstack([X_train, X_source[:utils.TTS, :]]),
                          y_train)
    del X_train, y_train
    # # TODO Use 5-fold CV and grid search to estimate hyperparameters
    logging.info('fitting xgboost model using diamond predictions')
    model = xgb.train(xgb_params(), D_train, MAX_ITS)
    logging.info("AUC = %f", roc_auc_score(y_val, model.predict(D_val)))


def train_test_split(df_train):
    TTS = 50000
    df_val = df_train.iloc[TTS:, :]
    df_train = df_train.iloc[:TTS, :]
    return df_train, df_val


def fit_diamond_model(df_train):
    path = 'models/diamond_partial.p'
    if os.path.exists(path):
        logging.info('reading saved diamond model')
        with open(path, 'rb') as ff:
            diamond_model = pickle.load(ff)
    else:
        logging.info('fitting diamond model')
        formula = 'target ~ 1 + (1|song_id) + (1|msno)'
        priors = pd.DataFrame({'group': ['song_id', 'msno'],
                               'var1': ['intercept'] * 2,
                               'var2': [np.nan] * 2,
                               # fit on a sample of data in R/lme4
                               'vcov': [0.00845, 0.07268]})
        diamond_model = LogisticRegression(df_train, priors)
        diamond_model.fit(formula, tol=1e-5, verbose=False, max_its=200)
        with open(path, 'wb') as ff:
            pickle.dump(diamond_model, ff)
    df_train.drop(['row_index', 'intercept'], axis=1, inplace=True, errors='ignore')
    return diamond_model


if __name__ == '__main__':
    main()
