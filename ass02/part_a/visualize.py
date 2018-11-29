import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random


def _generate_color_list():
    colors = []
    values = [x / 255 for x in [50, 100, 150, 200]]
    length = len(values)
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            for k in range(length):
                if i == k:
                    continue
                rgb = np.array([values[i], values[j], values[k]])
                colors.append(rgb.reshape((1, -1)))
    random.shuffle(colors)
    return colors


_color_list = _generate_color_list()


def _check_color_list():
    x = np.arange(-100, 100, 0.1)
    for i, _color in enumerate(_color_list):
        plt.plot(x, np.array([i * 10] * x.shape[0]), color=_color.flatten().tolist(), linewidth=4)
    plt.show()


def _partition(x, y):
    """Separate the data into several groups according the labels.

    :param x: The data to cluster. Shape (N, D).
    :param y: The true labels. Shape (N,)
    :return: A tuple of ndarrays, which contains groups of data.
    """
    unq_value, reverse_ix = np.unique(y, return_inverse=True)
    num_groups = unq_value.shape[0]
    groups = [[] for _ in range(num_groups)]

    for i in range(reverse_ix.shape[0]):
        groups[reverse_ix[i]].append(x[i])

    groups = [np.array(g) for g in groups]
    return groups


def _plot_groups(groups, ax, custom_color=False):
    """Plot each group, which is an ndarray.

    :param groups: List of ndarrays. Each group corresponds to a cluster.
    :param ax: The axes to plot on.
    """
    for i, group in enumerate(groups):
        group = group.T
        # in case the default color map is not big enough
        color = _color_list[i] if custom_color else 'C{:d}'.format(i)
        ax.scatter(group[0], group[1], s=10, c=color)


def _plot_cov_ellipse(mean, cov, ax, num_stddev=2, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        mean : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        cov : The 2x2 covariance matrix to base the ellipse on.
        num_stddev : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * num_stddev * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot(x, y, assign, mean=None, var=None, save_fig=None, cur_fig=None):
    """Plot the true and the learned assignments of data within one window.

    :param x: The data to cluster on.
    :param y: The true labels.
    :param assign: The learned assignments, i.e., labels.
    :param mean: If Kmeans or EM is used, plot the mean of each cluster.
    :param var: If EM is used, plot the one-std-dev contour of each cluster.
    :param save_fig: The path of the figure file to save, if specified.
    :param cur_fig: The current figure to plot on.
    """
    is_two_dim = True
    if x.shape[1] != 2:
        is_two_dim = False
    if mean is not None and mean.shape[1] != 2:
        is_two_dim = False
    if var is not None and (var.shape[1] != 2 or var.shape[2] != 2):
        is_two_dim = False
    if not is_two_dim:
        raise ValueError('The data is not 2-dimensional! Cannot plot!')

    if cur_fig is None:
        cur_fig = plt.figure()

    true_groups = _partition(x, y)
    ax = cur_fig.add_subplot(1, 2, 1)
    _plot_groups(true_groups, ax)

    ass_groups = _partition(x, assign)
    ax = cur_fig.add_subplot(1, 2, 2)
    _plot_groups(ass_groups, ax)

    if mean is not None:
        for i in range(mean.shape[0]):
            ax.scatter(mean[i, 0], mean[i, 1], s=50.0, c='C{:d}'.format(i), marker='+')

    if var is not None:
        for i in range(mean.shape[0]):
            _plot_cov_ellipse(mean[i], var[i], ax, num_stddev=2, alpha=0.2, color='C{:d}'.format(i))

    # cur_fig.set_size_inches(8, 3.5)
    if save_fig is not None:
        cur_fig.savefig(save_fig, format='pdf', bbox_inches='tight')
    return cur_fig


def plot_different_params(x, assign, layout, title, mean=None, cur_fig=None):
    if cur_fig is None:
        cur_fig = plt.figure()

    ax = cur_fig.add_subplot(*layout)
    groups = _partition(x, assign)
    _plot_groups(groups, ax, custom_color=True)
    if mean is not None:
        for i in range(mean.shape[0]):
            ax.scatter(mean[i, 0], mean[i, 1], s=50.0, c=_color_list[i], marker='+')

    ax.set_title(title)
    return cur_fig


if __name__ == '__main__':
    _check_color_list()
