import numpy as np
import pandas as pd
import os
import logging

from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from diamond.glms.logistic import LogisticRegression

# serialization
import dill

SOURCE_FEATURES = ['source_system_tab', 'source_screen_name', 'source_type']



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
        logging.info('creating encoding for %s', id_col)
        nunique = self.df[id_col].nunique()
        logging.info('there are %d unique values', nunique)
        idx = id_col + 'x'
        for x in cat_features:
            self.df[x] = self.df[x].astype(str).fillna('unknown')
        self.id_map = id_to_index(self.df[id_col])
        self.df[idx] = self.df[id_col].apply(lambda x: self.id_map[x])
        cats = self.df[cat_features + [idx]].sort_values(idx)
        self.X = DictVectorizer().fit_transform(cats.T.to_dict().values())
        logging.info('feature matrix has shape %d by %d',
                     self.X.shape[0], self.X.shape[1])
        if self.X.shape[0] != nunique:
            msg = "expected %d rows, but found only %d"
            raise ValueError(msg % (nunique, self.X.shape[0]))

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
    member_idx = df['msnox']
    song_idx = df['song_idx']
    y = df['target']
    if diamond:
        numeric_features += ['diamond_prediction']
    else:
        logging.info('indexing member matrix')
        matrix_list.append(members.X[member_idx, :])
    X_numeric = df[numeric_features].as_matrix()
    del df  # free up the memory
    logging.info('adding numeric feature matrix')
    matrix_list.append(X_numeric)
    logging.info('indexing song matrix')
    matrix_list.append(songs.X[song_idx, :])
    logging.info('indexing genre matrix')
    matrix_list.append(genres[song_idx, :])
    logging.info('concatenating matrices')
    X = sparse.hstack(matrix_list)
    return X, y


def get_source_encoding(source_features, df):
    logging.info('creating source features encoding')
    LE, OHE = LabelEncoder(), OneHotEncoder()
    X_source = OHE.fit_transform(df[source_features].\
        fillna('unknown').\
        apply(LE.fit_transform))
    return X_source


def train_test_split(df_train):
    """
    split labeled data into 2/3 train, 1/3 validation
    this reflects that the unlabeled test data is about
    half the size of the labeled data
    """
    TTS = 4918279
    df_val = df_train.iloc[TTS:, :]
    df_train = df_train.iloc[:TTS, :]
    return df_train, df_val


def fit_diamond_model(df_train):
    logging.info('fitting diamond model')
    formula = 'target ~ 1 + (1|song_id) + (1|msno)'
    priors = pd.DataFrame({'group': ['song_id', 'msno'],
                           'var1': ['intercept'] * 2,
                           'var2': [np.nan] * 2,
                           # fit on a sample of data in R/lme4
                           'vcov': [0.00845, 0.07268]})
    diamond_model = LogisticRegression(df_train, priors)
    diamond_model.fit(formula, tol=1e-5, verbose=False, max_its=200)
    df_train.drop(['row_index', 'intercept'],
                  axis=1, inplace=True, errors='ignore')
    return diamond_model

