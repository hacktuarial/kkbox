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
import sys
import os
import itertools
import pickle
import numpy as np
import pandas as pd
import click
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import roc_auc_score
import utils
import logging
from joblib import Memory

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO)


def cross_validate(X, K, params):
    """
    Do a grid search over parameters using K-fold cross validation
    :param: X. an xgb.DMatrix
    :param: K. number of folds
    :params: list of parameter dictionaries
    :max_its: how many trees to use
    """
    folds = np.random.choice(range(K), size=X.num_row())
    results = []
    for k in range(K):
        logging.info("now on fold %d/%d", k, K)
        train_slice = [i for i, f in enumerate(folds) if f != k]
        test_slice = [i for i, f in enumerate(folds) if f == k]
        X_train = X.slice(train_slice)
        X_test = X.slice(test_slice)
        for param in params:
            logging.info("using the following parameters")
            logging.info(param)
            model = xgb.train(param, X_train, param['max_its'])
            auc = roc_auc_score(X_test.get_label(),
                                model.predict(X_test))
            results.append((k, param, auc))
    return results


@click.command()
@click.option('--diamond/--no-diamond', required=True)
@click.option('--use-cache/--clear-cache', required=True)
def main(diamond, use_cache):
    if use_cache:
        memory = Memory('/tmp')
        c_get_data = memory.cache(utils.get_data)
        c_encode_categoricals = memory.cache(utils.encode_categoricals)
        # c_create_designs = memory.cache(create_designs)
    else:
        c_get_data = utils.get_data
        c_encode_categoricals = utils.encode_categoricals
        # c_create_designs = create_designs
    # set up logging
    logfile = 'output_diamond' if diamond else 'output_no_diamond'
    logfile += '.txt'
    sys.stdout = sys.stderr = open(logfile, 'a')
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
    if diamond:
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
        numeric_features.append('diamond_prediction')
    else:
        logging.info('Using OHE for member and song bias terms')
        # IDs: benchmark for diamond
        X_ids = utils.encode_categoricals(df, ['msno', 'song_id'])
        X_cats = sparse.hstack([X_cats, X_ids])

    logging.info('creating XGB design matrix')
    X_numeric = sparse.csr_matrix(df[numeric_features].as_matrix())
    y = df['target']
    # y_val = y[TTS:]
    del df
    logging.info("Using %d numeric features", X_numeric.shape[1])
    logging.info("Using %d categorical features", X_cats.shape[1])
    D_train, D_val = utils.create_designs(y, X_numeric, X_cats)
    del X_numeric, X_cats

    # etas = [0.5, 0.75, 0.9]
    # depths = [5, 12]
    # TODO greater depths, more iterations, higher eta
    etas = [0.9, 0.95]
    depths = [15, 20]
    parameters = []
    for eta, depth in itertools.product(etas, depths):
        pi = utils.xgb_params()
        pi['eta'] = eta
        pi['max_depth'] = depth
        pi['max_its'] = 200
        parameters.append(pi)

    logging.info('fitting xgboost models')
    results = cross_validate(X=D_train,
                             K=8,
                             params=parameters)
    with open('xval_results.p', 'wb') as ff:
        pickle.dump(results, ff)
    # each entry in results is (fold, auc, parameters)
    df_params = pd.DataFrame.from_dict([r[1] for r in results])
    df_auc = pd.DataFrame(results)[[0, 2]].rename(
            columns={0: 'fold', 2: 'auc'})
    df_results = pd.concat([df_params, df_auc], axis=1)
    df_results = df_results.groupby(['eta', 'max_depth', 'max_its'],
                                    as_index=False).\
        aggregate({'auc': np.mean}).\
        sort_values('auc', ascending=False)

    print("best parameters are")
    print(df_results.head(5))


if __name__ == '__main__':
    main()
