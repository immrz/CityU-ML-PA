import numpy as np


def soft_threshold(args, x, y):
    """Solve the lasso regression where the function is a
    polynomial w.r.t. x, using soft threshold.

    :param x: The sampled x.
    :param y: The sampled y.
    :return: The learned weight of lasso regression.
    """
    K, N = args.order + 1, x.shape[0]
    phi = np.vstack([x ** i for i in range(K)])  # shape (K, N)
    theta = np.random.normal(loc=0., scale=1., size=(K,))
    regu = args.wd * N

    for _ in range(args.epoch):
        # perform coordinate descent
        for k in range(K):
            cur_row = phi[k, :]
            rho = cur_row.dot(y - phi.T.dot(theta) + theta[k] * cur_row)
            z = cur_row.dot(cur_row)

            if rho > regu:
                theta[k] = (rho - regu) / z
            elif rho < -regu:
                theta[k] = (rho + regu) / z
            else:
                theta[k] = 0

        if args.debug:
            print(_, args.arr2str(theta), np.linalg.norm(theta, ord=2))

    return theta
