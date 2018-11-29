import numpy as np
import random
import ass02.part_a.utility as utility


def init_params(args, x: np.ndarray):
    """Initialize the means of each cluster.

    :param x: The data to cluster.
    :return: The means of each cluster.
    """
    N, D = x.shape
    K = args.num_cluster
    ix = random.sample(range(N), K)
    mean = x[ix].copy()  # type: np.ndarray
    return mean


def update_mean(x, assign):
    """Update the means according to the assignments and data.

    :param x: The data to cluster. Shape (N, D)
    :param assign: One-hot matrix which indicates the assignments. Shape (N, K)
    :return: The new means of the K cluster. Shape (K, D)
    """
    new_mean = assign.T.dot(x)
    den = assign.sum(axis=0).reshape((-1, 1))
    return new_mean / den


def euclidean_dist(x, mean):
    x_sq = np.sum(x * x, axis=1).reshape((-1, 1))  # shape: (N, 1)
    m_sq = np.sum(mean * mean, axis=1).reshape((1, -1))  # shape: (1, K)
    dot_prod = x.dot(mean.T)  # shape: (N, K)

    dist = x_sq + m_sq - 2 * dot_prod
    return dist


def update_assign(x, mean):
    """Update the assignments according to the means of clusters and data.

    :param x: The data to cluster. Shape (N, D).
    :param mean: The means of the K clusters. Shape (K, D)
    :return: The new assignments which is a one-hot matrix of shape (N, K).
        A_{ij} = 1 iff. x_i belongs to cluster j.
    """

    dist = euclidean_dist(x, mean)
    argmin_dist = np.argmin(dist, axis=1)
    N, K = x.shape[0], mean.shape[0]
    one_hot = np.zeros((N, K))  # shape: (N, K)
    one_hot[np.arange(N), argmin_dist] = 1
    return one_hot


def receiver(args, x):
    """This is the entry for kmeans algorithm.

    :param args: The user-defined arguments.
    :param x: The data to cluster. Shape (N, D)
    :return: The assignments of each data point, and the means of each cluster.
    """
    mean = init_params(args, x)
    real_iter = 0
    assign = None
    stopper = utility.ParamDiffStore()

    for real_iter in range(args.num_epoch):
        assign = update_assign(x, mean)
        mean = update_mean(x, assign)

        if stopper.is_saturate([mean]):
            break

    if args.debug:
        print('It took {:d} iterations to converge with KMeans.'.format(real_iter))
    assign = np.argmax(assign, axis=1)  # one-hot matrix back to label vector
    return assign, mean


def kmeans_test():
    # for test only
    class Foo(object):
        pass

    args = Foo()
    args.num_epoch = 10000
    args.num_cluster = 2

    mu1 = np.array([0, 0])
    mu2 = np.array([100, 100])
    sigma = np.array([[1, 0], [0, 1]])
    x1 = np.random.multivariate_normal(mean=mu1, cov=sigma, size=10)
    x2 = np.random.multivariate_normal(mean=mu2, cov=sigma, size=15)
    x = np.vstack((x1, x2))

    receiver(args, x)


if __name__ == '__main__':
    kmeans_test()
