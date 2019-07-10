import random
import numpy as np
class LogisticRegression(object):
    def __init__(self, x, y, lr=0.0005, lam=0.1):
			"""
			x: features of examples
			y: label of examples
			lr: learning rate
			lambda: penality on theta
			"""
        self.lr = lr
        self.lam = lam
        self.theta = np.array([0.0] * (n + 1))
    def _sigmoid(self, x):
        z = 1.0 / (1.0 + np.exp((-1) * x))
        return z
    def loss_function(self):
        u = self.__sigmoid(np.dot(self.x, self.theta))
        c1 = (-1) * self.y * np.log(u)
        c2 = (1.0 - self.y) * np.log(1.0 - u)
        # compute the cross-entroy
        loss = np.average(sum(c1 - c2) + 0.5 * self.lam * sum(self.theta[1:] ** 2))
        return loss
    def _gradient(self, iterations):
        # m is the number of examples, p is the number of features.
        m, p = self.x.shape     
        for i in xrange(0, iterations):
            u = self._sigmoid(np.dot(self.x, self.theta))
            diff = h_theta - self.y
            for _ in xrange(0, p):
                self.theta[_] = self.theta[_] - self.lr * (1.0 / m) * (sum(diff * self.x[:, _]) + self.lam * m * self.theta[_])
            cost = self._loss_function()
    def run(self, iterations):
        self._gradient(iterations)
    def predict(self, X):
        preds = self.__sigmoid(np.dot(x, self.theta))
        np.putmask(preds, preds >= 0.5, 1.0)
        np.putmask(preds, preds < 0.5, 0.0)
        return preds