# -*- coding: utf-8 -*-

import copy_reg
import types
from multiprocessing import Pool


def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def parallel_exec_func(num_thread, func, iterable):
	"""
	for real parallel in python, use multi-process instead of multi-thread
	:param num_thread:
	:param func:
	:param iterable:
	:return:
	"""
	num_process = None if num_thread == -1 else num_thread
	pool = Pool(num_process)
	rets = pool.map(func, iterable)
	pool.close()
	return rets


class _Scorer(object):
	def __init__(self, score_func, greater_is_better=True):
		self._score_func = score_func
		self._sign = 1 if greater_is_better else -1

	def __call__(self, y_true, y_pred):
		return self._sign * self._score_func(y_true, y_pred)


def make_scorer(score_func, greater_is_better=True):
	return _Scorer(score_func, greater_is_better)