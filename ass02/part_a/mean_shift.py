import numpy as np
import ass02.part_a.utility as utility
import matplotlib.pyplot as plt


def gaussian_kernel_grad(radius):
    """Calculate the gradient of the Gaussian kernel profile.

    :param radius: The radial distance between current point and all the data in x. Shape (N,)
    :return: The gradient w.r.t. `radius' of the Gaussian profile. Shape (N,)
    """
    return 0.5 * np.exp(-0.5 * radius)


def gradient_descent(args, x, init_pos):
    """Calculate the finally converged data point of `init_pos' using mean-shift algorithm.

    :param args: Arguments, like bandwidth.
    :param x: All data. Shape (N, D).
    :param init_pos: The initial data point. Shape (D,)
    :return: The converged data point.
    """
    stopper = utility.ParamDiffStore()
    cur_pos = init_pos
    real_iter = 0

    for real_iter in range(args.num_epoch):
        diff = (x - cur_pos) / args.bandwidth
        radius = np.sum(diff ** 2, axis=1)
        kernel = gaussian_kernel_grad(radius)
        weight_avg = x.T.dot(kernel)
        cur_pos = weight_avg / kernel.sum()

        if stopper.is_saturate([cur_pos]):
            break

    return real_iter, cur_pos


def scatter_compare(x, converged):
    x, converged = x.T, converged.T
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x[0], x[1], s=10.0)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(converged[0], converged[1], s=10.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    plt.show()


def receiver(args, x):
    """This is the entry of mean-shift algorithm. For each data point of `x', use mean-shift to find the mode
    that it converges to.

    :param args: Arguments.
    :param x: Data to cluster. Shape (N,D).
    :return: The assignments of each data point, and the modes of each cluster.
    """
    converged = []
    avg_num_iter = 0

    for i in range(x.shape[0]):
        real_iter, final_pos = gradient_descent(args, x, x[i, :])
        converged.append(final_pos)
        avg_num_iter += real_iter

    converged = np.array(converged)
    avg_num_iter /= x.shape[0]

    precision = 1e-1
    cast_to_int = (converged / precision).astype(np.int64)
    modes, assign = np.unique(cast_to_int, return_inverse=True, axis=0)

    if args.debug:
        print('After meanshift, there are {:d} modes.'.format(modes.shape[0]))
        print('Average number of iterations per sample is {:.3f}.'.format(avg_num_iter))
        if x.shape[1] == 2:
            scatter_compare(x, converged)

    return assign, modes * precision
