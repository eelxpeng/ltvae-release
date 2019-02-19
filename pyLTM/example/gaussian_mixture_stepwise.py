import warnings
import numpy as np
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from itertools import cycle, islice

def _estimate_gaussian_parameters(X, resp, reg_covar):
    '''resp: responsibility of each data point to each of the K clusters
    i.e. posterior p(z=k|x)
    '''
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    covariances = avg_X2 - avg_means2 + reg_covar

    return nk, means, covariances

def _compute_precision_cholesky(covariances):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")
    if np.any(np.less_equal(covariances, 0.0)):
        raise ValueError(estimate_precision_error_message)
    precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol

def _estimate_log_gaussian_prob(X, means, precisions_chol):
    n_components, n_features = means.shape
    log_det = (np.sum(np.log(precisions_chol), axis=1))
    precisions = precisions_chol ** 2
    log_prob = (np.sum((means ** 2 * precisions), 1) -
                2. * np.dot(X, (means * precisions).T) +
                np.dot(X ** 2, precisions.T))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

def logsumexp(a, axis=None, keepdims=False):
    a_max = np.amax(a, axis=axis, keepdims=True)
    tmp = np.exp(a - a_max)
    s = np.sum(tmp, axis=axis, keepdims=keepdims)
    out = np.log(s)
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    return out
    
class GaussianMixture:
    '''Implementation of Gaussian Mixture model with EM
    with diagonal covariance
    Note: to initialize the parameter, use precision cholesky for precisions_init.
        Don't use covariance to initialize
    '''
    def __init__(self, n_components=1, tol=1e-3, reg_covar=1e-6, max_iter=100, 
                n_init=1, init_params='kmeans', weights_init=None, means_init=None, 
                precisions_init=None, random_state=None, warm_start=False,
                verbose=0, verbose_interval=10):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def fit(self, X, learning_rate=0.01, batch_size=100, max_iter=None, init=False):
        if init:
            self._initialize(X)
        self.lower_bound_ = -np.infty
        
        n = X.shape[0]
        self.converged_ = False
        for n_iter in range(self.max_iter):
            prev_lower_bound = self.lower_bound_

            batch_idx = np.random.choice(n, batch_size, replace=False)
            batchX = X[batch_idx]
            log_prob_norm, log_resp = self.stepwise_e_step(batchX)
            self.stepwise_m_step(batchX, log_resp, learning_rate, batch_size)
            self.lower_bound_ = log_prob_norm

            change = self.lower_bound_ - prev_lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                break

        if not self.converged_:
            warnings.warn('Initialization not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.')
        print("Averaged loglikelihood: %f, n_iter: %d" % (self.score(X), n_iter))

    def score(self, X):
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1).mean()

    def _initialize(self, X, n_init=10):
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=n_init,
                                   random_state=self.random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = self.random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        n_samples, _ = X.shape
        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances)
        else:
            self.precisions_cholesky_ = self.precisions_init


    def stepwise_e_step(self, batchX):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(batchX)
        return np.mean(log_prob_norm), log_resp

    def stepwise_m_step(self, batchX, log_resp, learning_rate, batch_size):
        n_samples, _ = batchX.shape
        assert(n_samples==batch_size)
        resp = np.exp(log_resp)
        # initialize sufficient statistics for batch_size of data
        sufficient_counts = batch_size * self.weights_      # (K, )
        sufficient_counts = np.expand_dims(sufficient_counts, axis=1) # (K, 1)
        sufficient_sum = self.means_ * sufficient_counts    # (K, D)
        sufficient_sum_square = (self.covariances_ + self.means_**2) * sufficient_counts

        # update sufficient statistics
        sufficient_counts = sufficient_counts + learning_rate * (
            np.sum(resp, axis=0, keepdims=True).T - sufficient_counts)
        resp_3d = np.expand_dims(resp, axis=1) # Nx1xK
        batchX_3d = np.expand_dims(batchX, axis=2) # NxDx1
        sufficient_sum = sufficient_sum + learning_rate * (
            np.sum(resp_3d * batchX_3d, axis=0).T - sufficient_sum)
        sufficient_sum_square = sufficient_sum_square + learning_rate * (
            np.sum(resp_3d * batchX_3d**2, axis=0).T - sufficient_sum_square)

        # update parameters
        self.weights_ = np.squeeze(sufficient_counts, axis=1) / batch_size
        self.means_ = sufficient_sum / sufficient_counts
        self.covariances_ = sufficient_sum_square / sufficient_counts - self.means_**2

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_)
        
    def log_prob(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_prob_norm

    def _estimate_log_prob_resp(self, X):
        """
        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def set_parameters(self, weights, means, covariances):
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(covariances)

    def predict(self, X):
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def grad(self, X, log_resp):
        resp = np.exp(log_resp)
        means_3d = np.expand_dims(self.means_.T, axis=0)
        precision_3d = np.expand_dims((self.precisions_cholesky_ ** 2).T, axis=0)
        X_3d = np.expand_dims(X, axis=2)
        resp_3d = np.expand_dims(resp, axis=1)
        return np.sum(resp_3d * precision_3d * (means_3d - X_3d), axis=2)


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad

def rel_error(x, y):
    """returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__=="__main__":
    n_samples = 1500
    n_centers = 3
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    X, y = blobs
    gmm = GaussianMixture(n_components=n_centers)
    gmm.fit(X, learning_rate=0.01, batch_size=100, init=True)
    y_pred = gmm.predict(X)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    plt.show()

    # gmm = GaussianMixture(n_components=3, init_params='kmeans')
    # gmm.set_parameters(np.array([0.5, 0.2, 0.3]), np.array([[0., 0.], [1.,1.],[2., 2.]]), np.array([[1.,1.], [1.,1.], [1., 1.]]))
    # input = np.random.randn(1,2)
    # log_prob_norm, log_resp = gmm._estimate_log_prob_resp(input)
    # grad_analytic = gmm.grad(input, log_resp)
    # grad_numerical = eval_numerical_gradient(gmm.log_prob, 
    #   input, h=1e-5)
    # print(grad_analytic)
    # print(grad_numerical)
    # print("Relative error: %f" % rel_error(grad_analytic, grad_numerical))
    
