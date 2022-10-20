#-*- codeing = utf-8 -*-
# @Time : 2022/7/6 14:24
# @Author : 夏冰雹
# @File : Weights.py 
# @Software: PyCharm
import numpy as np
class LinearModelWeights:
    def __init__(self,w :np.array,fit_intercept,raise_overflow_error = False):
        self._weights = w
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.raise_overflow_error = raise_overflow_error
        if fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w

    @property
    def unboxed(self):
        return self._weights
