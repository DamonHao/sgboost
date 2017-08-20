# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn import metrics

from loss import LABEL_COLUMN, GRAD_COLUMN, HESS_COLUMN, LogisticLoss, SquareLoss, CustomLoss
from tree import Tree
import utils

_EVAL_METRIC = {'accuracy': metrics.accuracy_score,
		'neg_mean_squared_error': utils.make_scorer(metrics.mean_squared_error, False),
		'r2': metrics.r2_score}


class SGBModel(object):
	"""
	Simple Gradient Boosting
	"""

	def __init__(self, loss='square', learning_rate=0.3, n_estimators=20, max_depth=6,
			subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
			min_child_weight=1, reg_lambda=1.0, gamma=0, num_thread=-1):

		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.subsample = subsample
		self.colsample_bytree = colsample_bytree
		self.colsample_bylevel = colsample_bylevel
		self.reg_lambda = reg_lambda
		self.gamma = gamma
		self.num_thread = num_thread
		self.min_child_weight = min_child_weight
		self.first_round_pred = 0.0
		self.trees = []
		self.eval_metric = None

		self._is_classifier = False

		if loss == 'logistic':
			self.loss = LogisticLoss()
		elif loss == 'square':
			self.loss = SquareLoss()
		else:
			if callable(loss):
				self.loss = CustomLoss(loss)
			else:
				raise Exception('unsupported loss function: {0}'.format(loss))

	def fit(self, X, y, eval_metric=None, early_stopping_rounds=None):
		self.trees = []
		self.feature_importances_ = {}
		self.eval_metric = _EVAL_METRIC[eval_metric] if eval_metric else None

		X.reset_index(drop=True, inplace=True)
		y.reset_index(drop=True, inplace=True)

		# Y stores: label, y_pred, grad, hess, sample_weight
		Y = pd.DataFrame(y.values, columns=[LABEL_COLUMN])
		Y['y_pred'] = self.first_round_pred
		Y[GRAD_COLUMN], Y[HESS_COLUMN] = self.loss.compute_grad_hess(Y.y_pred.values, Y.label.values)

		if self._is_classifier:
			Y['sample_weight'] = 1.0
			Y.loc[Y.label == 1, 'sample_weight'] = self.scale_pos_weight

		if self.eval_metric is not None and early_stopping_rounds is not None:
			assert early_stopping_rounds > 0
			best_val_score = -np.inf
			score_worse_round = 0
			best_round = 0

		for idx in xrange(self.n_estimators):
			if self._is_classifier:
				Y[GRAD_COLUMN] = Y[GRAD_COLUMN] * Y.sample_weight
				Y[HESS_COLUMN] = Y[HESS_COLUMN] * Y.sample_weight

			# subsample column and row before training the current tree
			X_sample_column = X.sample(frac=self.colsample_bytree, axis=1)
			data = pd.concat([X_sample_column, Y], axis=1)
			data = data.sample(frac=self.subsample, axis=0)

			X_feed = data[X_sample_column.columns]
			Y_feed = data[Y.columns]

			tree = Tree(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
					colsample_bylevel=self.colsample_bylevel, reg_lambda=self.reg_lambda, gamma=self.gamma,
					num_thread=self.num_thread)

			tree.fit(X_feed, Y_feed)

			# predict the whole train set to update the y_pred, grad and hess
			preds = tree.predict(X[X_sample_column.columns])

			Y['y_pred'] += self.learning_rate * preds
			Y[GRAD_COLUMN], Y[HESS_COLUMN] = self.loss.compute_grad_hess(Y.y_pred.values, Y.label.values)

			# only compute feature importance in "weight" type, xgboost support two more type "gain" and "cover"
			for feature, weight in tree.feature_importances_.iteritems():
				if feature in self.feature_importances_:
					self.feature_importances_[feature] += weight
				else:
					self.feature_importances_[feature] = weight

			self.trees.append(tree)

			if self.eval_metric is None:
				print '[SGBoost] train round: {0}'.format(idx)
			else:
				cur_val_score = self._eval_score(Y.label.values, Y.y_pred.values)
				print '[SGBoost] train round: {0}, eval score: {1}'.format(idx, cur_val_score)

				if early_stopping_rounds is not None:
					if cur_val_score > best_val_score:
						best_val_score = cur_val_score
						score_worse_round = 0
						best_round = idx
					else:
						score_worse_round += 1

					if score_worse_round > early_stopping_rounds:
						print '[SGBoost] train best round: {0}, best eval score: {1}'.format(best_round, best_val_score)
						break

		return self

	def predict(self, X):
		assert len(self.trees) > 0
		# TODO: add parallel tree prediction
		# but now a daemonic process is not allowed to create child processes
		preds = np.zeros((X.shape[0],))
		preds += self.first_round_pred
		for tree in self.trees:
			preds += self.learning_rate * tree.predict(X)
		return preds

	def _eval_score(self, y_true, y_pred):
		raise NotImplementedError()


class SGBClassifier(SGBModel, ClassifierMixin):

	def __init__(self, loss='logistic', learning_rate=0.3, n_estimators=20, max_depth=6,
			scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
			min_child_weight=1, reg_lambda=1.0, gamma=0, num_thread=-1):
		super(SGBClassifier, self).__init__(
			loss=loss,
			learning_rate=learning_rate,
			n_estimators=n_estimators,
			max_depth=max_depth,
			subsample=subsample,
			colsample_bytree=colsample_bytree,
			colsample_bylevel=colsample_bylevel,
			min_child_weight=min_child_weight,
			reg_lambda=reg_lambda,
			gamma=gamma,
			num_thread=num_thread,
		)
		self._is_classifier = True
		self.scale_pos_weight = scale_pos_weight

	def predict(self, X):
		probs = self.loss.compute_probs(super(SGBClassifier, self).predict(X))
		return probs.round()

	def _eval_score(self, y_true, y_pred):
		return self.eval_metric(y_true, self.loss.compute_probs(y_pred).round())


class SGBRegressor(SGBModel, RegressorMixin):

	def _eval_score(self, y_true, y_pred):
		return self.eval_metric(y_true, y_pred)