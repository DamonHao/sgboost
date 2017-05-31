# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score

from loss import LABEL_COLUMN, GRAD_COLUMN, HESS_COLUMN, LogisticLoss, SquareLoss
from tree import Tree


class SGBModel(object):
	"""
	Simple Gradient Boosting
	"""

	def __init__(self, loss='logistic', learning_rate=0.3, n_estimators=20, max_depth=6,
			scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
			min_child_weight=1, min_sample_split=10, reg_lambda=1.0, gamma=0, num_thread=-1):

		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.subsample = subsample
		self.colsample_bytree = colsample_bytree
		self.colsample_bylevel = colsample_bylevel
		self.reg_lambda = reg_lambda
		self.gamma = gamma
		self.min_sample_split = min_sample_split
		self.num_thread = num_thread
		self.min_child_weight = min_child_weight
		self.scale_pos_weight = scale_pos_weight
		self.first_round_pred = 0.0
		self.trees = []

		if loss == 'logistic':
			self.loss = LogisticLoss(reg_lambda)
		elif loss == 'square':
			self.loss = SquareLoss(reg_lambda)
		else:
			raise Exception('do not support customize loss')

	def fit(self, X, y):
		self.trees = []

		X.reset_index(drop=True, inplace=True)
		y.reset_index(drop=True, inplace=True)

		# Y stores: label, y_pred, grad, hess, sample_weight
		Y = pd.DataFrame(y.values, columns=[LABEL_COLUMN])
		Y['y_pred'] = self.first_round_pred
		Y[GRAD_COLUMN] = self.loss.grad(Y.y_pred.values, Y.label.values)
		Y[HESS_COLUMN] = self.loss.hess(Y.y_pred.values, Y.label.values)
		# Y['sample_weight'] = 1.0
		# Y.loc[Y.label == 1, 'sample_weight'] = self.scale_pos_weight

		for idx in xrange(self.n_estimators):
			# subsample column and row before training the current tree
			X_sample_column = X.sample(frac=self.colsample_bytree, axis=1)
			data = pd.concat([X_sample_column, Y], axis=1)
			data = data.sample(frac=self.subsample, axis=0)

			X_feed = data[X_sample_column.columns]
			Y_feed = data[Y.columns]

			tree = Tree(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
					colsample_bylevel=self.colsample_bylevel, min_sample_split=self.min_sample_split,
					reg_lambda=self.reg_lambda, gamma=self.gamma, num_thread=self.num_thread)

			tree.fit(X_feed, Y_feed)

			# predict the whole train set to update the y_pred, grad and hess
			preds = tree.predict(X[X_sample_column.columns])

			Y['y_pred'] += self.learning_rate * preds
			Y[GRAD_COLUMN] = self.loss.grad(Y.y_pred.values, Y.label.values)
			Y[HESS_COLUMN] = self.loss.hess(Y.y_pred.values, Y.label.values)

			self.trees.append(tree)

			# score = accuracy_score(Y.label.values, self.loss.transform(Y.y_pred.values).round())
			# print '[SGBoost] train round :{0}, score: {1}'.format(idx, score)

	def predict(self, X):
		assert len(self.trees) > 0
		# TODO: add parallel tree prediction, but now a daemonic process is not allowed to create child processes
		preds = np.zeros((X.shape[0],))
		preds += self.first_round_pred
		for tree in self.trees:
			preds += self.learning_rate * tree.predict(X)
		return self.loss.transform(preds)


class SGBClassifier(SGBModel, ClassifierMixin):

	def predict(self, X):
		return super(SGBClassifier, self).predict().round()


class SGBRegressor(SGBModel, RegressorMixin):
	pass