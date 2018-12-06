import numpy as np


def soft_threshold(phi, y, num_epoch, wd):
    """Solve the lasso regression where the function is a
    polynomial w.r.t. x, using soft threshold.
    """

    dim, num = phi.shape
    theta = np.random.normal(loc=0., scale=1., size=(dim,))
    regu = wd * num

    for _ in range(num_epoch):
        # perform coordinate descent
        for k in range(dim):
            cur_row = phi[k, :]
            rho = cur_row.dot(y - phi.T.dot(theta) + theta[k] * cur_row)
            z = cur_row.dot(cur_row)

            if rho > regu:
                theta[k] = (rho - regu) / z
            elif rho < -regu:
                theta[k] = (rho + regu) / z
            else:
                theta[k] = 0
    return theta
