# -*- coding: utf-8 -*-

import numpy as np

LABEL_COLUMN = 'label'
GRAD_COLUMN = 'grad'
HESS_COLUMN = 'hess'


class BaseLoss(object):

	def compute_grad_hess(self, y_preds, y_trues):
		raise NotImplementedError()

	def compute_probs(self, y_preds):
		return 1.0/(1.0+np.exp(-y_preds))


class LogisticLoss(BaseLoss):

	def compute_grad_hess(self, y_preds, y_trues):
		probs = self.compute_probs(y_preds)
		grads = probs - y_trues
		hess = probs * (1-probs)
		return grads, hess


class SquareLoss(BaseLoss):

	def compute_grad_hess(self, y_preds, y_trues):
		grad = 2 * (y_preds - y_trues)
		hess = np.full(y_trues.shape, 2)
		return grad, hess


class CustomLoss(BaseLoss):

	def __init__(self, loss_func):
		self._loss_func = loss_func

	def compute_grad_hess(self, y_preds, y_trues):
		return self._loss_func(y_preds, y_trues)