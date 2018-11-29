import numpy as np
import ass02.part_a.utility as utility
import ass02.part_a.kmeans as kmeans


class FuncPool:
    """FuncPool contains functions that are used calculate the parameters (theta) and posterior distribution
    in EM algorithm. For example, `mean_solver' returns the mean value parameter for Gaussian and Poisson
    distribution, which is a weighted average of sufficient statistics.
    These functions make the program generalize well in the way that it decouples the EM steps with the particular
    distribution assumed for the data. If users would like to use other distributions, they can simply add their
    own customized functions that defines how to calculate the parameters and posterior.

    """
    @staticmethod
    def mean_solver(x, post, theta):
        weighted_sum = post.T.dot(x)
        cluster_size = np.sum(post.T, axis=1, keepdims=True)
        return weighted_sum / cluster_size

    @staticmethod
    def prior_solver(x, post, theta):
        N = x.shape[0]
        cluster_size = np.sum(post, axis=0)
        return cluster_size / N

    @staticmethod
    def variance_solver(x, post, theta):
        mean = theta[1]
        centralized = x[:, None, :] - mean[None, :, :]  # Shape (N, K, D). C[i,k] = x[i] - mu[k].
        var = np.einsum('ikx,iky->ikxy', centralized, centralized)  # Shape (N, K, D, D). V[i,k] = C[i,k] outer C[i,k].
        weighted_var = np.einsum('ikxy,ik->kxy', var, post)  # Shape (K, D, D)
        cluster_size = np.sum(post, axis=0).reshape((-1, 1, 1))

        var = weighted_var / cluster_size
        K, D = var.shape[0], var.shape[1]
        for k in range(K):
            if np.linalg.det(var[k]) < 0.05 ** D:
                var[k] = 0.1 * np.eye(D)
        return var

    @staticmethod
    def _calc_post_and_loss_with_bayesian(likelihood: np.ndarray, prior: np.ndarray):
        """This is a `private' method which is used to calculate the posterior using Bayesian rule.
        The ndarrays `likelihood' and `prior' should have shapes (N, K) and (K,) respectively.

        :param likelihood: L[i,k] = prob(x_i|z_{ik}=1)
        :param prior: P[k] = prob(z_{ik}=1}
        :return: Post[i,k] = prob(z_{ik}=1|x_i}
        """
        joint = likelihood * prior
        marginal = likelihood.dot(prior).reshape((-1, 1))
        post = joint / marginal

        hard_ass = post.argmax(axis=1)
        obj = 0
        for i in range(likelihood.shape[0]):
            ass = hard_ass[i]
            obj += np.log(prior[ass]) + np.log(likelihood[i, ass])
        return post, obj.item()

    @staticmethod
    def poisson_post_calculator(x, theta):
        prior, mean = theta
        x, mean = x.flatten(), mean.flatten()

        likelihood = np.exp(-mean) * (mean ** x[:, None])
        post, obj = FuncPool._calc_post_and_loss_with_bayesian(likelihood, prior)
        # obj = np.sum(post.dot(np.log(prior)) + np.diagonal(post.dot(np.log(likelihood.T))))
        return post, obj

    @staticmethod
    def gaussian_post_calculator(x, theta):
        prior, mean, var = theta
        prec = np.linalg.inv(var)  # Shape (K, D, D)
        prec_det = np.linalg.det(prec)  # Shape (K,)
        assert np.all(prec_det >= 0), 'The determinants of variance matrices should all be positive'

        central = x[:, None, :] - mean[None, :, :]  # Shape (N, K, D). C[i,k] = x[i] - mu[k].
        mahala_dist = np.einsum('ijx,jxy,ijy->ij', central, prec, central)  # Shape (N, K). M[i,k] = MDist(x[i]-mu[k])
        const_den = (2 * np.pi) ** (mean.shape[1] / 2)

        likelihood = np.sqrt(prec_det) * np.exp(-0.5 * mahala_dist) / const_den  # Shape (N, K)
        post, obj = FuncPool._calc_post_and_loss_with_bayesian(likelihood, prior)
        return post, obj


def init_params(args, x):
    K = args.num_cluster

    if args.distribution == 'poisson':
        mean = np.random.uniform(low=1, high=x.flatten().max(), size=(K,))
        mean.sort()
        prior = np.array([1 / K] * K)
        return [prior, mean], [FuncPool.prior_solver, FuncPool.mean_solver], FuncPool.poisson_post_calculator

    elif args.distribution == 'gaussian':
        _, mean = kmeans.receiver(args, x)
        prior = np.array([1 / K] * K)
        var = args.var * np.vstack([np.eye(x.shape[1])[None, :, :] for _ in range(K)])
        return [prior, mean, var], [FuncPool.prior_solver, FuncPool.mean_solver, FuncPool.variance_solver], \
            FuncPool.gaussian_post_calculator

    else:
        raise NotImplementedError


def receiver(args, x):
    """This function is the entry of EM algorithm. It receives arguments and data sent from the main module.
    In this function, `theta' and `solvers' are lists of same length. The former contains the necessary parameters
    for the assumed distribution, which is dynamically determined in function `init_params', with the first element
    always being the prior of each cluster. The latter contains static methods defined in class `FuncPool', with the
    i-th element solving the i-th parameter of `theta' in each iteration (M-Step).
    `post_calc' is also a method in `FuncPool' which is used to calculate the posterior distribution (E-Step).

    For example, if the distribution is Gaussian, `theta' would be (prior, mean, var).
    `prior': Shape (K,).
    `mean': Shape (K, D).
    `var': Shape (K, D, D).
    `post': Shape (N, K).

    :param args: Arguments, such as the distribution assumed for p(X|Z).
    :param x: The data to cluster. Shape (N, D) where N is the number of points and D the dimension.
    :return: The assignments of each data point, and the parameters learned.
    """
    theta, solvers, post_calc = init_params(args, x)
    cur_obj = 0.
    real_iter = 0
    post = None
    stopper = utility.ParamDiffStore()

    for real_iter in range(args.num_epoch):
        post, cur_obj = post_calc(x, theta)
        theta = [solvers[i](x, post, theta) for i in range(len(solvers))]

        if stopper.is_saturate(theta):
            break

    if args.debug:
        print('After {:d} iterations, the objective is {:.5f}\nThe parameters are:'.format(real_iter, cur_obj))
        for i in range(args.num_cluster):
            print('CLUSTER {:03d}'.format(i))
            for param in theta:
                print(param[i])
            print('')

    assign = np.argmax(post, axis=1)
    return assign, theta


def poisson_test():
    # for test only
    x1 = [229, 211, 93, 35, 7, 1]
    x2 = [325, 115, 67, 30, 18, 21]

    def transform_data(x):
        num = len(x)
        realx = []
        for i in range(num):
            realx = realx + [i] * x[i]
        return np.array(realx, dtype=np.float64).reshape((-1, 1))

    x1, x2 = transform_data(x1), transform_data(x2)

    class Foo(object):
        pass

    args = Foo()
    args.distribution = 'poisson'
    args.num_epoch = 10000
    args.debug = True
    for k in range(1, 6):
        args.num_cluster = k
        print('When the number of clusters is {:d}'.format(k))
        receiver(args, x2)
        print('=' * 40 + '\n\n')


if __name__ == '__main__':
    poisson_test()
