import numpy as np
from scipy import special, linalg


def approximation(kn_mat, y, num_iter=10):
    n = kn_mat.shape[0]
    mode, g, p = None, None, None  # mode, gradient and precision matrix
    kn_inv = np.linalg.inv(kn_mat)

    """Use Newton-Raphson method to find the mode"""
    for _ in range(num_iter):
        if mode is None:
            mode = np.random.normal(loc=0, scale=1, size=(2 * n,))  # initialization
        else:
            delta = np.linalg.solve(p, g)  # alternative for inv(p).dot(g)
            mode += delta

        norm, bss0, bss1 = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            norm[i] = np.sqrt(mode[i]**2 + mode[i+n]**2)
            bss0[i] = special.iv(0, norm[i])  # Bessel function of order 0
            bss1[i] = special.iv(1, norm[i])  # Bessel function of order 1

        g = get_oracle_gradient(mode, y, kn_inv, bss1 / (bss0 * norm))
        if np.linalg.norm(g, ord=2) < 1e-5:
            break
        p = get_oracle_precision(mode, kn_inv, norm, bss0, bss1)

    return mode, g, p


def get_oracle_gradient(f, y, kn_inv, coef):
    n = coef.shape[0]
    f1, f2 = f[:n], f[n:]
    a = np.concatenate((kn_inv.dot(f1), kn_inv.dot(f2)), axis=0)
    b = np.concatenate((coef * f1, coef * f2), axis=0)
    return -a + y - b


def get_oracle_precision(f, knv_inv, norm, bss0, bss1):
    n = norm.shape[0]
    precision = linalg.block_diag(knv_inv, knv_inv)
    for i in range(n):
        precision[i, i] += (bss0[i]*bss0[i]*f[i]*f[i]
                            - bss1[i]*bss1[i]*f[i]*f[i]
                            + bss0[i]*bss1[i]*norm[i]
                            - 2*bss0[i]*bss1[i]*f[i]*f[i]/norm[i]) / (bss0[i]*bss0[i]*norm[i]*norm[i])
        precision[i+n, i+n] += (bss0[i]*bss0[i]*f[i + n]*f[i + n]
                                - bss1[i]*bss1[i]*f[i + n]*f[i + n]
                                + bss0[i]*bss1[i]*norm[i]
                                - 2*bss0[i]*bss1[i]*f[i + n]*f[i + n]/norm[i]) / (bss0[i]*bss0[i]*norm[i]*norm[i])
        c = ((bss0[i]*bss0[i]-bss1[i]*bss1[i])*f[i]*f[i + n]
             - 2*bss0[i]*bss1[i]*f[i]*f[i + n]/norm[i]) / (bss0[i]*bss0[i]*norm[i]*norm[i])
        precision[i, i + n] += c
        precision[i + n, i] += c

    return precision
