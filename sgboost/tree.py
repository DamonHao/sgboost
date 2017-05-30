# -*- coding: utf-8 -*-

import copy_reg
import types

import numpy as np
from multiprocessing import Pool
from functools import partial
import pandas as pd

from loss import LABEL_COLUMN, GRAD_COLUMN, HESS_COLUMN


def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class Tree(object):
	
	def __init__(self, max_depth=6, min_child_weight=1, colsample_bylevel=1.0, min_sample_split=10, reg_lambda=1.0, gamma=0.0, num_thread=-1):
		self.max_depth = max_depth
		self.min_child_weight = min_child_weight
		self.colsample_bylevel = colsample_bylevel
		self.min_sample_split = min_sample_split
		self.reg_lambda = reg_lambda
		self.gamma = gamma
		self.num_thread = num_thread
	
	def fit(self, X, Y):
		self.feature_importances_ = {}
		self.root_ = self._build(X, Y, self.max_depth)
	
	def _build(self, X, Y, max_depth):
		if max_depth == 0 or X.shape[0] < self.min_sample_split or Y.hess.sum() < self.min_child_weight:
			leaf_score = self._compute_leaf_score(Y)
			return TreeNode(is_leaf=True, leaf_score=leaf_score)
		
		X_sample_column = X.sample(frac=self.colsample_bylevel, axis=1)
		
		best_feature, best_threshold, best_gain, nan_direction = self._find_best_feature_threshold(X_sample_column, Y)
		
		# TODO add post pruning
		if best_gain < 0:
			leaf_score = self._compute_leaf_score(Y)
			return TreeNode(is_leaf=True, leaf_score=leaf_score)
		
		X_left, Y_left, X_right, Y_right = self._split_dataset(X, Y, best_feature, best_threshold, nan_direction)
		
		left_child = self._build(X_left, Y_left, max_depth-1)
		right_child = self._build(X_right, Y_right, max_depth-1)
		
		if best_feature in self.feature_importances_:
			self.feature_importances_[best_feature] += 1
		else:
			self.feature_importances_[best_feature] = 0
			
		return TreeNode(feature=best_feature, threshold=best_threshold, nan_direction=nan_direction,
				left_child=left_child, right_child=right_child)
		
	@staticmethod
	def _split_dataset(X, Y, feature, threshold, nan_direction):
		if nan_direction == 0:
			right_mask = X[feature] > threshold
			left_mask = ~ right_mask
		else:
			left_mask = X[feature] <= threshold
			right_mask = ~ left_mask
		return X[left_mask], Y[left_mask], X[right_mask], Y[right_mask]
		
	def _find_best_feature_threshold(self, X, Y):
		best_gain = -np.inf
		nan_direction = 0
		best_feature = None
		best_threshold = None
		
		cols = list(X.columns)
		data = pd.concat([X, Y], axis=1)
		func = partial(self._find_best_threshold, data)

		# for real parallel in python, use multi-process instead of multi-thread
		num_process = None if self.num_thread == -1 else self.num_thread
		pool = Pool(num_process)
		rets = pool.map(func, cols)
		pool.close()
		
		for ret in rets:
			if ret[2] > best_gain:
				best_gain = ret[2]
				best_feature = ret[0]
				best_threshold = ret[1]
				nan_direction = ret[3]

		return best_feature, best_threshold, best_gain, nan_direction

	def _find_best_threshold(self, data, col):
		selected_data = data[[col, LABEL_COLUMN, GRAD_COLUMN, HESS_COLUMN]]
		best_gain = - np.inf
		best_threshold = None
		best_nan_direction = None

		nan_mask = selected_data[col].isnull()
		nan_data = selected_data[nan_mask]
		G_nan = nan_data[GRAD_COLUMN].sum()
		H_nan = nan_data[HESS_COLUMN].sum()

		not_nan_data = selected_data[~nan_mask]
		not_nan_data.reset_index(drop=True, inplace=True)
		sorted_index = not_nan_data[col].argsort()
		not_nan_data = not_nan_data.ix[sorted_index]
		
		feature_values = not_nan_data[col]
		for idx in xrange(not_nan_data.shape[0]-1):
			cur_value = feature_values.iloc[idx]
			next_value = feature_values.iloc[idx+1]
			
			if cur_value == next_value:
				continue
			
			cur_threshold = (cur_value + next_value) / 2.0
			
			Y_left = not_nan_data.iloc[:idx+1]
			Y_right = not_nan_data.iloc[idx+1:]
			
			nan_goto_left_gain = self._compute_split_gain(Y_left, Y_right, G_nan, H_nan, 0)
			nan_goto_right_gain = self._compute_split_gain(Y_left, Y_right, G_nan, H_nan, 1)
			
			if nan_goto_left_gain < nan_goto_right_gain:
				cur_gain = nan_goto_right_gain
				nan_direction = 1
			else:
				cur_gain = nan_goto_left_gain
				nan_direction = 0
				
			if cur_gain > best_gain:
				best_gain = cur_gain
				best_threshold = cur_threshold
				best_nan_direction = nan_direction
			
		return col, best_threshold, best_gain, best_nan_direction

	def _compute_leaf_score(self, Y):
		return - Y.grad.sum() / (Y.hess.sum() + self.reg_lambda)

	def _compute_split_gain(self, Y_left, Y_right, G_nan, H_nan, nan_direction):
		if nan_direction == 0:
			G_left = Y_left[GRAD_COLUMN].sum() + G_nan
			H_left = Y_left[HESS_COLUMN].sum() + H_nan
			G_right = Y_right[GRAD_COLUMN].sum()
			H_right = Y_right[HESS_COLUMN].sum()
		else:
			G_left = Y_left[GRAD_COLUMN].sum()
			H_left = Y_left[HESS_COLUMN].sum()
			G_right = Y_right[GRAD_COLUMN].sum() + G_nan
			H_right = Y_right[HESS_COLUMN].sum() + H_nan
			
		gain = 0.5*(G_left**2/(H_left+self.reg_lambda) + G_right**2/(H_right+self.reg_lambda) -
				(G_left+G_right)**2/(H_left+H_right+self.reg_lambda)) - self.gamma
		
		return gain

	def _predict(self, tree_node, row_tuple):
		if tree_node.is_leaf:
			return tree_node.leaf_score

		feature_value = row_tuple[1][tree_node.feature]
		if pd.isnull(feature_value):
			if tree_node.nan_direction == 0:
				return self._predict(tree_node.left_child, row_tuple)
			else:
				return self._predict(tree_node.right_child, row_tuple)
		elif feature_value <= tree_node.threshold:
			return self._predict(tree_node.left_child, row_tuple)
		else:
			return self._predict(tree_node.right_child, row_tuple)

	def predict(self, X):
		rows = X.iterrows()
		func = partial(self._predict, self.root_)
		num_process = None if self.num_thread == -1 else self.num_thread
		pool = Pool(num_process)
		preds = pool.map(func, rows)
		pool.close()
		return np.array(preds)


class TreeNode(object):

	def __init__(self, is_leaf=False, leaf_score=None, feature=None, threshold=None, nan_direction=None,
			left_child=None, right_child=None):
		self.is_leaf = is_leaf
		self.leaf_score = leaf_score
		self.feature = feature
		self.threshold = threshold
		self.nan_direction = nan_direction
		self.left_child = left_child
		self.right_child = right_child
