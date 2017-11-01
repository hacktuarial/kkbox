# Compare two ways of estimating user- and item- bias factors
# 1. One hot encode them and put them into an xgboost model with some other features
# 2. Use `diamond` to fit a crossed random-effects model, then put these predictions into an xgboost model with the same set of other features

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
import click
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import roc_auc_score
import utils
import os
from joblib import Memory

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO)


MAX_ITS = 200


@click.command()
@click.option('--diamond/--no-diamond', required=True)
@click.option('--use-cache/--clear-cache', required=True)
def main(diamond, use_cache):
    # serialization
    if use_cache:
        memory = Memory('/tmp')
        fit_diamond_model = memory.cache(utils.fit_diamond_model)
        encode_genres = memory.cache(utils.encode_genres)
        get_source_encoding = memory.cache(utils.get_source_encoding)
        merge_it_all_together = memory.cache(utils.merge_it_all_together)
    else:
        fit_diamond_model = utils.fit_diamond_model
        encode_genres = utils.encode_genres
        get_source_encoding = utils.get_source_encoding
        merge_it_all_together = utils. merge_it_all_together

    # # 1. Read in data
    logging.info('reading in data')
    df_train = pd.read_csv('data/raw/train.csv')
    # there are 2.3M songs in the songs dataframe
    # but only 300K in the training data
    if os.path.isfile('data/processed/songs.dill') and use_cache:
        logging.info('reading song encoding from disk')
        # initialize empty object
        songs = utils.ExtraInfo(df=None)
        songs.load('data/processed/songs.dill')
    else:
        logging.info('creating song encoding')
        df_songs = pd.read_csv('data/raw/songs.csv')
        # restrict to songs present in the train/val data
        df_songs = df_songs[df_songs['song_id'].isin(df_train['song_id'])]
        x = ['artist_name', 'composer', 'lyricist', 'language']
        songs = utils.ExtraInfo(df=df_songs)
        songs.create_encoding(id_col='song_id', cat_features=x)
        songs.save('data/processed/songs.dill')
    # repeat for members
    # in both members and df_train, there are 30-35K members
    if os.path.isfile('data/processed/members.dill') and use_cache:
        logging.info('reading member encoding from disk')
        members = utils.ExtraInfo(df=None)
        members.load('data/processed/members.dill')
    else:
        members = utils.ExtraInfo(df=pd.read_csv('data/raw/members.csv'))
        x = ['city', 'gender', 'registered_via']
        members.create_encoding(id_col='msno',
                                cat_features=x)
        members.save('data/processed/members.dill')

    if diamond:
        logging.info('using diamond for member, song biases')
        matrix_list = []
        df_train, df_val = utils.train_test_split(df_train)
        diamond_model = fit_diamond_model(df_train)
        df_train['diamond_prediction'] = diamond_model.predict(df_train)
        df_val['diamond_prediction'] = diamond_model.predict(df_val)
    else:
        # IDs: benchmark for diamond
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
    X_source = get_source_encoding(source_features=utils.SOURCE_FEATURES,
                                   df=df_train)

    logging.info('merging training data together')
    X_train, y_train = merge_it_all_together(df_train,
                                             diamond=diamond,
                                             members=members,
                                             matrix_list=matrix_list,
                                             songs=songs,
                                             genres=X_genres)
    del df_train
    D_train = xgb.DMatrix(sparse.hstack([X_train, X_source[:utils.TTS, :]]),
                          y_train)
    del X_train, y_train
    logging.info('merging validation data together')
    X_val, y_val = merge_it_all_together(df_val,
                                         diamond=diamond,
                                         members=members,
                                         songs=songs,
                                         matrix_list=matrix_list,
                                         genres=X_genres)
    del df_val
    D_val = xgb.DMatrix(sparse.hstack([X_val, X_source[utils.TTS:, :]]))
    # # TODO Use 5-fold CV and grid search to estimate hyperparameters
    logging.info('fitting xgboost model using diamond predictions')
    model = xgb.train(utils.xgb_params(), D_train, MAX_ITS)
    logging.info("AUC = %f", roc_auc_score(y_val, model.predict(D_val)))



if __name__ == '__main__':
    main()
