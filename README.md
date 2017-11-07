The primary analysis in this repo is the following.

* Use xgboost on 4 numerical features, plus a bunch of categorical features _excluding_ song- and member id's
* For these features, fit them two ways
  1. Use [diamond](http://github.com/stitchfix/diamond). Feed the predictions into the XGB model
  2. One-hot encode the id features. Include them as predictors in the XGB model
* Measure the AUC on a holdout set
