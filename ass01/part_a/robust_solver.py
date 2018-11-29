import numpy as np


def robust_regression(args, x, y):
    """Solve the L1-norm regression using SGD. In experiments, it is found that the algorithm doesn't
    converge well with higher-order polynomials. So momentum as well as decaying learning rate is used,
    to augment SGD.

    :param x: The sampled x.
    :param y: The sampled y.
    :return: The learned weight of L1-norm regression.
    """
    K, N = args.order + 1, x.shape[0]
    phi = np.vstack([x ** i for i in range(K)])  # shape (K, N)
    theta = np.random.normal(loc=0., scale=1., size=(K,))

    lr = 1e-3
    momentum = 0.95
    state = np.zeros(shape=(K,))

    for _ in range(args.epoch):
        indicator = y < phi.T.dot(theta)  # type: np.ndarray
        indicator = -2 * indicator.astype(theta.dtype) + 1

        grad = phi.dot(indicator)
        state = lr * grad + momentum * state
        theta += state

        if (_ + 1) % 600 == 0:
            lr /= 10

        if args.debug:
            print(_, args.arr2str(theta), np.linalg.norm(theta, ord=2))

    return theta
