from TML_ass01.laplace import approximation as lpl_approx
import numpy as np
import scipy
from scipy import special
from math import cos, sin, pi
from numpy.polynomial import hermite


class VonMisesRegression:
    def __init__(self, train_data, kernel, **kwargs):
        self.train_data = train_data
        self.test_data = None
        self.kernel = kernel
        self.kw = kwargs
        self.train_stat = None
        self.f_post = None
        self.f_pred = None
        self.ans = None

    def train_model(self):
        x_mat = np.array([[e[0], e[1]] for e in self.train_data])
        x_mean = np.average(x_mat, axis=0)
        x_stddev = np.std(x_mat, axis=0)
        x_mat = (x_mat - x_mean) / x_stddev  # normalization

        y_mat = np.array([cos(2 * pi * e[2]) for e in self.train_data] + [sin(2 * pi * e[2]) for e in self.train_data])
        kn_mat = compute_kernel(x_mat, x_mat.copy(), self.kernel,
                                **self.kw)  # (N, N) kernel matrix of training data
        kn_mat += 1e-6 * np.eye(kn_mat.shape[0])  # regularization

        """Use laplace approximation to estimate the mean and covariance of the Gaussian of latent variable f"""
        f_mean, cur_grad, f_prec = lpl_approx(kn_mat, y_mat, num_iter=self.kw['nr_iter'])
        assert np.linalg.norm(cur_grad, ord=2) < 1e-3

        self.train_stat = (x_mat, x_mean, x_stddev, kn_mat)
        self.f_post = (f_mean, f_prec)

    def infer_f(self, test_data):
        self.test_data = test_data
        x_test = np.array([[e[0], e[1]] for e in test_data])  # coordinates
        x_test = (x_test - self.train_stat[1]) / self.train_stat[2]

        """The predictive distribution of f* is also Gaussian. Use its mean for prediction."""
        kn_cross = compute_kernel(self.train_stat[0], x_test, self.kernel, **self.kw)  # shape (N_{train}, N_{test})
        inv_dot = np.linalg.solve(self.train_stat[3], kn_cross)  # shape (N_{train}, N_{test}), inv(K).dot(k*)
        xform = scipy.linalg.block_diag(inv_dot, inv_dot)  # (transpose of) transformation matrix between f* and f
        f_expt = xform.T.dot(self.f_post[0])  # shape (2N_{test},)

        """Compute the covariance matrix for each test point"""
        m = x_test.shape[0]
        kn_self = np.diag(compute_kernel(x_test, x_test, self.kernel, **self.kw))
        cond_var = kn_self - np.sum(kn_cross * inv_dot, axis=0)
        cov_c_prod = np.linalg.solve(self.f_post[1], xform)
        mgn_cov = []
        for i in range(x_test.shape[0]):
            idx = [i, i + m]
            mgn_cov.append(np.diag([cond_var[i], cond_var[i]]) + xform[:, idx].T.dot(cov_c_prod[:, idx]))

        self.f_pred = (f_expt, mgn_cov)
        return f_expt

    def infer_y(self, n_samp=1000):
        points, weights = hermite.hermgauss(deg=n_samp)
        all_samp = []
        all_w = []
        for i in range(n_samp):
            for j in range(n_samp):
                all_samp.append([points[i], points[j]])
                all_w.append(weights[i] * weights[j])
        all_samp = np.array(all_samp).T  # (2, P)
        all_w = np.array(all_w)
        f_mu = self.f_pred[0].reshape((2, -1)).T
        const = np.sqrt(2)
        prediction = []

        for i in range(len(self.test_data)):
            mu = f_mu[i]
            cov = self.f_pred[1][i]
            e, ev = np.linalg.eig(cov)
            affine = ev.dot(np.diag(np.sqrt(e)))
            theta = const * affine.dot(all_samp) + mu[:, None]
            theta = theta.dot(all_w)
            prediction.append(theta)

        return prediction

    def infer_y_descent(self, n_samp=10):
        points, weights = hermite.hermgauss(deg=n_samp)
        all_samp = []
        all_w = []
        for i in range(n_samp):
            for j in range(n_samp):
                all_samp.append([points[i], points[j]])
                all_w.append(weights[i] * weights[j])
        all_samp = np.array(all_samp).T  # (2, P)
        all_w = np.array(all_w)
        f_mu = self.f_pred[0].reshape((2, -1)).T
        const = np.sqrt(2)
        prediction = []

        for i in range(len(self.test_data)):
            mu = f_mu[i]
            cov = self.f_pred[1][i]
            e, ev = np.linalg.eig(cov)
            affine = ev.dot(np.diag(np.sqrt(e)))  # (2, 2)
            theta = const * affine.dot(all_samp) + mu[:, None]  # (2, P)
            bss0 = np.array([special.iv(0, norm) for norm in np.linalg.norm(theta, ord=2, axis=0)])
            w = all_w / bss0

            """Perform Gradient Descent to find optimal y"""
            y = np.random.normal(size=(2,))
            for _ in range(100):
                exp = np.exp(theta.T.dot(y))
                coef = w * exp  # (P,)
                grad = theta.dot(coef)
                if np.linalg.norm(grad, ord=2) < 1e-5:
                    break
                hessian = (theta * coef).dot(theta.T)
                y -= np.linalg.solve(hessian, grad)

            else:
                print('Descent not converged for the {:d}-th sample.'.format(i + 1))
            prediction.append(y)

        return np.array(prediction)

    def infer_y_naive(self):
        f_expt = self.f_pred[0].reshape((2, -1)).T
        return f_expt

    def compute_error(self, prediction):
        prediction = prediction.T.tolist()
        t_hat = inverse_trigonometric(prediction[1], prediction[0])
        t_hat = np.array(t_hat) / (2 * pi)

        _, _, t = zip(*self.test_data)
        t = np.array(t)
        rmse = np.sqrt(np.average((t_hat - t) ** 2))

        rdm_asw = np.random.uniform(low=0, high=1, size=(len(self.test_data)))
        rmse2 = np.sqrt(np.average((rdm_asw - t) ** 2))

        self.ans = t_hat
        return rmse, rmse2


def compute_kernel(x1, x2, which, **kwargs):
    """Compute kernel functions between x1 and x2 pairwise.
    :param x1: A collection of samples. Shape (N, D).
    :param x2: The other collection. Shape (M, D).
    :param which: Which kernel function to use.
    :return: A matrix shaped (N, M) whose element K_{i,j} is the kernel between x1[i] and x2[j].
    """
    if which == 'poly':
        return (x1.dot(x2.T) + kwargs['c']) ** kwargs['order']
    elif which == 'rbf':
        diff = x1[:, None, :] - x2[None, :, :]  # shape (N, M, D)
        d_sq = np.sum(diff ** 2, axis=-1)  # shape (N, M)
        exponent = -d_sq / (2 * kwargs['sigma']**2)
        return np.exp(exponent)
    else:
        raise NotImplementedError


def inverse_trigonometric(s, c):
    angles = []
    for si, ci in zip(s, c):
        length = np.sqrt(si**2 + ci**2)
        si, ci = si / length, ci / length
        angles.append(np.arccos(ci) if si >= 0 else 2 * pi - np.arccos(ci))
    return angles
