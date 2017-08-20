# -*- coding: utf-8 -*-

from sklearn.datasets import make_classification
from pandas import DataFrame
import numpy as np


from sgboost import SGBClassifier

if __name__ == '__main__':
	X, y = make_classification(1000, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=0)
	df = DataFrame(np.hstack((X, y[:, None])), columns=['col'+str(idx) for idx in range(5)] + ['label'])
	# df.to_csv('../data/classify_train.csv', index=False)


	val_min_index = 800
	train = df.iloc[:val_min_index]
	val = df.iloc[val_min_index:]

	X_train = train.drop('label', axis=1)
	y_train = train.label
	X_val = val.drop('label', axis=1)
	y_val = val.label

	params = dict(loss="logistic",
			learning_rate=0.3,
			max_depth=3,
			n_estimators=10,
			subsample=0.7,
			colsample_bytree=0.7,
			colsample_bylevel=1.0,
			min_child_weight=3,
			reg_lambda=10,
			scale_pos_weight=1,
			gamma=0,
			num_thread=-1)

	sgb = SGBClassifier(**params)
	sgb.fit(X_train, y_train, 'accuracy')
	print sgb.score(X_val, y_val)  # the score approximate 0.96
