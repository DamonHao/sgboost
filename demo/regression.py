# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame
from sklearn.datasets import make_regression

from sgboost import SGBRegressor

X, y = make_regression(1000, n_features=5, n_informative=2, random_state=0)
df = DataFrame(np.hstack((X, y[:, None])), columns=['col'+str(idx) for idx in range(5)] + ['label'])
# df.to_csv('../data/regression_train.csv', index=False)

val_min_index = 800
train = df.iloc[:val_min_index]
val = df.iloc[val_min_index:]

X_train = train.drop('label', axis=1)
y_train = train.label
X_val = val.drop('label', axis=1)
y_val = val.label

params = dict(loss="square",
		learning_rate=0.3,
		max_depth=6,
		n_estimators=10,
		scale_pos_weight=1.0,
		subsample=0.7,
		colsample_bytree=0.7,
		colsample_bylevel=1.0,
		min_child_weight=3,
		reg_lambda=10,
		gamma=0,
		num_thread=-1)

sgbr = SGBRegressor(**params)
sgbr.fit(X_train, y_train, 'r2')
print sgbr.score(X_val, y_val)  # the score is close to 0.99
print sgbr.feature_importances_
