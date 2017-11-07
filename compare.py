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

import os
import pickle
import click
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import roc_auc_score
import utils
import logging
from joblib import Memory

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO)


class ModelSupport(object):
    def __init__(self,
                 df,
                 numeric_features,
                 X_ids,
                 X_cats):
        self.df = df
        self.numeric_features = numeric_features
        self.X_ids = X_ids
        self.X_cats = X_cats
        self.y = df['target']

    def fit(self, parameters, diamond):
        logging.info("fitting model using diamond")
        if diamond:
            X_cats = self.X_cats
            diamond_features = self.numeric_features + ['diamond_prediction']
            X_numeric = sparse.csr_matrix(
                    self.df[diamond_features].as_matrix()
                    )
        else:
            X_numeric = sparse.csr_matrix(
                    self.df[self.numeric_features].as_matrix()
                    )
            X_cats = sparse.hstack([self.X_cats, self.X_ids])

        logging.info("Using %d numeric features", X_numeric.shape[1])
        logging.info("Using %d categorical features", X_cats.shape[1])
        D_train, D_val = utils.create_designs(self.y, X_numeric, X_cats)
        model = xgb.train(parameters, D_train, parameters['max_its'])
        auc = roc_auc_score(D_val.get_label(), model.predict(D_val))
        return model, auc


@click.command()
@click.option('--use-cache/--clear-cache', required=True)
def main(use_cache):
    if use_cache:
        memory = Memory('/tmp')
        c_get_data = memory.cache(utils.get_data)
        c_encode_categoricals = memory.cache(utils.encode_categoricals)
    else:
        c_get_data = utils.get_data
        c_encode_categoricals = utils.encode_categoricals
        # c_create_designs = create_designs

    logging.info('reading in training data')
    df = c_get_data()

    # encode categorical features
    # IGNORE GENRES FOR NOW
    numeric_features = [
            'song_length', 'registration_init_time',  # song features
            'expiration_date', 'bd',  # member features
            # there are no numeric 'source' features
                        ]
    cat_features = [
            'artist_name', 'composer', 'lyricist', 'language',  # song features
            'city', 'gender', 'registered_via',  # member features
            'source_system_tab', 'source_screen_name', 'source_type',  # source features
                    ]
    id_cols = ['msno', 'song_id']
    logging.info('checking %d numeric features' % len(numeric_features))
    for x in numeric_features:
        assert df[x].isnull().sum() == 0, "%s has nulls" % x
    logging.info('numeric features look ok')
    # drop any unused features
    df = df[numeric_features + cat_features + ['target'] + id_cols]
    logging.info('encoding %d categorical features', len(cat_features))
    # it takes 5 minutes even just to read this from disk
    X_cats = c_encode_categoricals(df, cat_features)

    # estimate member- and song bias terms
    logging.info('using diamond for member, song bias terms')
    path = 'models/diamond.p'
    if os.path.exists(path) and use_cache:
        logging.info('reading pickled diamond model from disk')
        with open(path, 'rb') as ff:
            diamond_model = pickle.load(ff)
    else:
        df_diamond = df.iloc[:utils.TTS, :][id_cols + ['target']].copy()
        # matrix_list = []
        diamond_model = utils.fit_diamond_model(df_diamond)
        logging.info('saving fitted diamond model to disk')
        with open(path, 'wb') as ff:
            pickle.dump(diamond_model, ff)
        # this is a mix of train + val observations
        del df_diamond
    df['diamond_prediction'] = diamond_model.predict(df)
    # IDs: benchmark for diamond
    X_ids = utils.encode_categoricals(df, ['msno', 'song_id'])
    MS = ModelSupport(df=df,
                      numeric_features=numeric_features,
                      X_ids=X_ids,
                      X_cats=X_cats)

    # these parameters were the best from
    # grid searching each model separately
    parameters = utils.xgb_params()
    parameters['eta'] = 0.9
    parameters['max_its'] = 100
    parameters['max_depth'] = 12

    # fit model with diamond
    diamond_model, diamond_auc = MS.fit(parameters, diamond=True)
    logging.info("diamond approach has an AUC of %f", diamond_auc)

    # fit model without diamond
    alt_model, alt_auc = MS.fit(parameters, diamond=False)
    logging.info("id's approach has an AUC of %f", alt_auc)


if __name__ == '__main__':
    main()
