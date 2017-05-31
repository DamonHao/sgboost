# -*- coding: utf-8 -*-

import numpy as np

LABEL_COLUMN = 'label'
GRAD_COLUMN = 'grad'
HESS_COLUMN = 'hess'


class BaseLoss(object):
	def __init__(self, reg_lambda=0.0):
		self.reg_lambda = reg_lambda
	
	def grad(self, preds, labels):
		raise NotImplementedError()
	
	def hess(self, preds, labels):
		raise NotImplementedError()
	

class LogisticLoss(BaseLoss):
	
	def transform(self, preds):
		return 1.0/(1.0+np.exp(-preds))

	def grad(self, y_preds, y_true):
		y_preds = self.transform(y_preds)
		return (1 - y_true) / (1 - y_preds) - y_true / y_preds

	def hess(self,preds, labels):
		preds = self.transform(preds)
		return labels/np.square(preds) + (1-labels)/np.square(1-preds)


class SquareLoss(BaseLoss):

	def transform(self, y_preds):
		return y_preds

	def grad(self, y_preds, y_true):
		return 2 * (y_preds - y_true)

	def hess(self, preds, y_true):
		return np.full(y_true.shape, 2)