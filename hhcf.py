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

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO)


def id_to_index(ids):
    " return a dict like {id: idx} "
    unique_ids = set(ids)
    return dict(zip(unique_ids, np.arange(0, len(unique_ids))))


def encode_categoricals(df, cat_features, id_col):
    """
    one hot encode categorical features. modifies `df` in-place
    """
    idx = id_col + 'x'
    for x in cat_features:
        df[x] = df[x].astype(str).fillna('unknown')
    id_map = id_to_index(df[id_col])
    df[idx] = df[id_col].apply(lambda x: id_map[x])
    df = df[cat_features + [idx]].        sort_values(idx)
    X = DictVectorizer().fit_transform(df.T.to_dict().values())
    return X.tocsr(), id_map


def encode_genres(song_map, df_songs):
    " this is a many-hot encoding of song to genre "
    df_genres = df_songs['genre_ids'].apply(lambda s: pd.Series(str(s).split('|')))
    df_genres['song_idx'] = df_songs['song_id'].apply(lambda x: song_map[x])
    df_genres2 = pd.melt(df_genres, 'song_idx', value_name='genre').        drop('variable', axis=1).        dropna().        sort_values('song_idx')
    genre_map = id_to_index(df_genres2['genre'])
    df_genres2['genre_idx'] = df_genres2['genre'].apply(lambda g: genre_map[g])
    X_genres = sparse.coo_matrix((np.ones(len(df_genres2)),
                                 (df_genres2['song_idx'], df_genres2['genre_idx'])))
    return X_genres.tocsr(), genre_map


def xgb_params():
    xgb_params = {}
    xgb_params['objective'] = 'binary:logistic'
    xgb_params['eta'] = 0.75
    xgb_params['max_depth'] = 5
    xgb_params['silent'] = 1
    xgb_params['eval_metric'] = 'auc'
    return xgb_params


def merge_it_all_together(df, songs, genres, members, diamond):
    """
    Merge member, song, and source data together
    Return (X, y) tuple
    This makes a _copy_ of df
    """

    logging.info('merging in song info')
    df = pd.merge(df.drop(SOURCE_FEATURES, axis=1),
                  songs[['song_id', 'song_idx', 'song_length']],
                  # there are 80 interactions with unknown songs in the training set
                  # remove them!
                  'inner',
                  'song_id')
    logging.info('merging in member info')
    df = pd.merge(df, members[['msno', 'msnox',
                                  'registration_init_time', 'expiration_date', 'bd']], 'left', 'msno')
    n_nulls = df.isnull().sum()
    assert n_nulls.sum() == 0, n_nulls
    logging.info('there are no nulls')

    numeric_features = ['song_length', 'registration_init_time', 'expiration_date', 'bd']
    X_numeric = df[numeric_features].as_matrix()
    member_idx = df['msnox']
    song_idx = df['song_idx']
    y = df['target']
    del df  # free up the memory

    if diamond:
        logging.info('using diamond predictions')
        numeric_features.append('diamond_prediction')
        matrix_list = []
    else:
        logging.info('using one-hot encoding of member, song ids')
        matrix_list = [X_member_ids[member_idx, :],
                       X_song_ids[song_idx, :]]

    logging.info('adding numeric feature matrix')
    matrix_list.append(X_numeric)
    logging.info('creating song matrix')
    matrix_list.append(songs[song_idx, :])
    logging.info('creating genre matrix')
    matrix_list.append(genres[song_idx, :])
    logging.info('creating member matrix')
    matrix_list.append(members[member_idx, :])
    logging.info('concatenating matrices')
    X = sparse.hstack(matrix_list)
    return X, y


MAX_ITS = 200
SOURCE_FEATURES = ['source_system_tab', 'source_screen_name', 'source_type']


@click.command()
@click.option('--diamond/--no-diamond', required=True)
def main(diamond):
    # # 1. Read in data
    logging.info('reading in data')
    df_train = pd.read_csv('data/raw/train.csv')
    df_songs = pd.read_csv('data/raw/songs.csv')
    df_members = pd.read_csv('data/raw/members.csv')

    # # 2. Create encoding matrices
    # ### Songs
    logging.info('creating songs encoding')
    X_songs, song_map = encode_categoricals(df_songs, ['artist_name', 'composer', 'lyricist', 'language'], 'song_id')
    assert X_songs.shape[0] == len(df_songs)

    # ### Members
    logging.info('creating members encoding')
    X_members, member_map = encode_categoricals(df_members,
                                                ['city', 'gender', 'registered_via'],
                                                'msno')
    assert X_members.shape[0] == len(df_members)

    # ### IDs: benchmark for diamond
    if not diamond:
        logging.info('creating member id encoding')
        X_member_ids = sparse.coo_matrix((np.ones(len(df_members)),
                        (df_members['msnox'], df_members['msnox']))).tocsr()
        logging.info('creating song id encoding')
        X_song_ids = sparse.coo_matrix((np.ones(len(df_songs)),
                                        (df_songs['song_idx'], df_songs['song_idx']))).tocsr()

    # ### Genres
    logging.info('creating genre encoding')
    X_genres, genre_map = encode_genres(song_map,
                                        df_songs)

    # ### Source features
    # Thanks to [Ritchie Ng](http://www.ritchieng.com/machinelearning-one-hot-encoding/) for guidance on using One Hot Encoder

    logging.info('creating source features encoding')
    LE, OHE = LabelEncoder(), OneHotEncoder()
    X_source = OHE.fit_transform(df_train[SOURCE_FEATURES].fillna('unknown').apply(LE.fit_transform))

    # # 3. Train diamond model
    # Need a train/test split for this and subsequent modeling
    logging.info('train/test split')
    df_val = df_train.iloc[utils.TTS:, :]
    df_train = df_train.iloc[:utils.TTS, :]

    logging.info('fitting diamond model')
    formula = 'target ~ 1 + (1|song_id) + (1|msno)'
    priors = pd.DataFrame({'group': ['song_id', 'msno'],
                           'var1': ['intercept'] * 2,
                           'var2': [np.nan] * 2,
                           # fit on a sample of data in R/lme4
                           'vcov': [0.00845, 0.07268]})
    diamond = LogisticRegression(df_train, priors)
    diamond.fit(formula, tol=1e-5, verbose=False, max_its=200)
    with open('models/diamond.p', 'wb') as ff:
        pickle.dump(diamond, ff)

    df_train.drop(['row_index', 'intercept'], axis=1, inplace=True)
    df_train['diamond_pred'] = diamond.predict(df_train)
    df_val['diamond_pred'] = diamond.predict(df_val)

    logging.info('merging everything together using diamond')
    X_val, y_val = merge_it_all_together(df_val,
                                         diamond=diamond,
                                         members=X_members,
                                         songs=X_songs,
                                         genres=X_genres)
    del df_val
    D_val = xgb.DMatrix(sparse.hstack([X_val, X_source[utils.TTS:, :]]))
    logging.info('merging everything together, not using diamond')
    X_train, y_train = merge_it_all_together(df_train,
                                             diamond=diamond,
                                             members=X_members,
                                             songs=X_songs,
                                             genres=X_genres)
    del df_train
    D_train = xgb.DMatrix(sparse.hstack([X_train, X_source[:utils.TTS, :]]),
                          y_train)
    del X_train, y_train
    # # TODO Use 5-fold CV and grid search to estimate hyperparameters
    logging.info('fitting xgboost model using diamond predictions')
    model = xgb.train(xgb_params(), D_train, MAX_ITS)
    logging.info("AUC = %f", roc_auc_score(y_val, model.predict(D_val)))


if __name__ == '__main__':
    main()
