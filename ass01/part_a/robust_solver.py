import numpy as np


def robust_regression(phi, y, num_epoch):
    """Solve the L1-norm regression using SGD. In experiments, it is found that the algorithm doesn't
    converge well with higher-order polynomials. So momentum as well as decaying learning rate is used,
    to augment SGD.
    """
    dim = phi.shape[0]
    theta = np.random.normal(loc=0., scale=1., size=(dim,))

    lr = 1e-3
    momentum = 0.95
    state = np.zeros(shape=(dim,))

    for _ in range(num_epoch):
        indicator = y < phi.T.dot(theta)  # type: np.ndarray
        indicator = -2 * indicator.astype(theta.dtype) + 1

        grad = phi.dot(indicator)
        state = lr * grad + momentum * state
        theta += state

        if (_ + 1) % 600 == 0:
            lr /= 10

    return theta
