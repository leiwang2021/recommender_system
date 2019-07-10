import numpy as np
class SVM():
    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma=1.0, coef0=0.0,
                 tol=1e-4, alphatol=1e-7, maxiter=10000, numpasses=10,
                 random_state=None, verbose=0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.alphatol = alphatol
        self.maxiter = maxiter
        self.numpasses = numpasses
        self.random_state = random_state
        self.verbose = verbose
    def fit(self, X, y):
        """Fit the model to data matrix X and target y.
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The class labels.
        returns a trained SVM
        """
        self.support_vectors_ = check_array(X)
        self.y = check_array(y, ensure_2d=False)
        random_state = check_random_state(self.random_state)
        self.kernel_args = {}
        if self.kernel == "rbf" and self.gamma is not None:
            self.kernel_args["gamma"] = self.gamma
        elif self.kernel == "poly":
            self.kernel_args["degree"] = self.degree
            self.kernel_args["coef0"] = self.coef0
        elif self.kernel == "sigmoid":
            self.kernel_args["coef0"] = self.coef0
        K = pairwise_kernels(X, metric=self.kernel, **self.kernel_args)
        self.dual_coef_ = np.zeros(X.shape[0])
        self.intercept_ = _svm.smo(
            K, y, self.dual_coef_, self.C, random_state, self.tol,
            self.numpasses, self.maxiter, self.verbose)
        # If the user was using a linear kernel, lets also compute and store
        # the weights. This will speed up evaluations during testing time.
        if self.kernel == "linear":
            self.coef_ = np.dot(self.dual_coef_ * self.y, self.support_vectors_)
        # only samples with nonzero coefficients are relevant for predictions
        support_vectors = np.nonzero(self.dual_coef_)
        self.dual_coef_ = self.dual_coef_[support_vectors]
        self.support_vectors_ = X[support_vectors]
        self.y = y[support_vectors]
        return self
    def decision_function(self, X):
        if self.kernel == "linear":
            return self.intercept_ + np.dot(X, self.coef_)
        else:
            K = pairwise_kernels(X, self.support_vectors_, metric=self.kernel, **self.kernel_args)
            return (self.intercept_ + np.sum(self.dual_coef_[np.newaxis, :] * self.y * K, axis=1))
    def predict(self, X):
        return np.sign(self.decision_function(X))