import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
from ass01.part_a import lasso_solver, robust_solver


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default=5, help='The order of the polynomial.')
    parser.add_argument('--alg', type=str, choices=['ls', 'lasso', 'rls', 'br', 'rr'], default='ls')
    parser.add_argument('--wd', type=float, default=0., help='The regularization factor lambda.')
    parser.add_argument('--epoch', type=int, default=2000, help='The number of iterations for lasso and rr.')
    parser.add_argument('--debug', action='store_true', help='If set, more information is displayed.')
    parser.add_argument('--alpha', type=float, default=0., help='The prior variance of the parameters to learn.')
    parser.add_argument('--var', type=float, default=0., help='The prior variance of the noise.')
    parser.add_argument('--data-pct', type=float, help='The percentage of training data to use.')
    parser.add_argument('--outlier', action='store_true', help='If set, add outliers to training data randomly.')

    # check whether the arguments are valid
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv.split())
    eps = 1e-3
    if args.order < 1:
        raise ValueError('The order of the polynomial should be at least 1.')
    if args.alg == 'br':
        if args.alpha < eps:
            raise ValueError('The prior variance of parameters should be positive.')
        if args.var < eps:
            raise ValueError('The prior variance of noise should be positive.')
    if (args.alg == 'rls' or args.alg == 'lasso') and args.wd < eps:
        raise ValueError('The regularization hyper-parameter should be positive.')

    # generate a short text describing the config
    conf_txt = '{:<12}:{:>6}\n{:<12}:{:>6}'.format('Poly-order', args.order, 'Alg', args.alg)
    if args.alg == 'rls' or args.alg == 'lasso':
        conf_txt = conf_txt + '\n{:<12}:{:6.2f}'.format('Lambda', args.wd)
    if args.alg == 'lasso' or args.alg == 'rr':
        conf_txt = conf_txt + '\n{:<12}:{:>6}'.format('Epoch', args.epoch)
    if args.alg == 'br':
        conf_txt = conf_txt + '\n{:<12}:{:>6.2f}\n{:<12}:{:>6.2f}'.format('Alpha', args.alpha, 'Sigma', args.var)
    args.conf_txt = conf_txt

    # create a function which transforms an array to a string,
    # with each element rounded to specific precision.
    def ndarray_to_str(arr, precision=3):
        fmt = '{:.' + str(precision) + 'f}'
        to_str = [fmt.format(e) for e in arr.tolist()]
        return '[' + ', '.join(to_str) + ']'
    args.arr2str = ndarray_to_str

    return args


def read_data(sample=True):
    if sample:
        filex = 'polydata_data_sampx.txt'
        filey = 'polydata_data_sampy.txt'
    else:
        filex = 'polydata_data_polyx.txt'
        filey = 'polydata_data_polyy.txt'

    with open(os.path.join(DATA_PATH, filex), 'r') as fr:
        x = list(map(float, fr.readline().split()))

    with open(os.path.join(DATA_PATH, filey), 'r') as fr:
        y = list(map(float, fr.read().split()))

    assert len(x) == len(y)
    return np.array(x), np.array(y)


def data_perturbation(args, x, y):
    """Perturb the training to test the robustness of the algorithms. Methods include adding outliers
    and decreasing data randomly.

    :param x: The original samples of x.
    :param y: The original samples of y.
    :return: The perturbed training data (x*, y*).
    """
    x, y = x.copy(), y.copy()

    if args.data_pct is not None:
        if args.data_pct <= 0 or args.data_pct >= 1:
            raise ValueError('The ratio of the training data should be in the range (0, 1).')
        # pick a percentage of samples for training
        total_len = x.shape[0]
        selected = random.sample(range(total_len), int(total_len * args.data_pct))
        x, y = x[selected], y[selected]
        args.conf_txt = args.conf_txt + '\n{:<12}:{:>6}'.format('Sample-num', x.shape[0])

    if args.outlier:
        num_outlier = random.choice([1, 2, 3])
        outlier_pos = random.sample(range(x.shape[0]), num_outlier)
        for pos in outlier_pos:
            y[pos] += random.choice([-1, 1]) * random.gauss(mu=100, sigma=10)
        args.conf_txt = args.conf_txt + '\n{:<12}:{:>6}'.format('Outlier-num', num_outlier)

    return x, y


def poly_interpolate(args, x, y):
    """Use regression methods to learn weights to interpolate
    the polynomial function given the sampled data.

    :param x: The sampled x.
    :param y: The sampled y.
    :return: The learned weight of regression.
    """
    K = args.order + 1
    phi = np.vstack([x ** i for i in range(K)])

    if args.alg == 'ls':
        theta = np.linalg.inv(phi.dot(phi.T)).dot(phi.dot(y))
    elif args.alg == 'rls':
        theta = np.linalg.inv(phi.dot(phi.T) + args.wd*np.eye(K)).dot(phi.dot(y))
    elif args.alg == 'lasso':
        theta = lasso_solver.soft_threshold(args, x, y)
    elif args.alg == 'br':
        sigma_post = np.linalg.inv(np.eye(K)/args.alpha + phi.dot(phi.T)/args.var)
        mu_post = sigma_post.dot(phi.dot(y)) / args.var
        theta = (mu_post, sigma_post)
    elif args.alg == 'rr':
        theta = robust_solver.robust_regression(args, x, y)
    else:
        raise NotImplementedError

    return theta


def analyze_result(args, samp, real, theta, ax=None):
    """Draw figures to compare the learned polynomial with the real function.
    Also, compute the RMSE between the two functions.

    :param samp: A list containing the sampled x and y.
    :param real: A list containing x and y of the true function.
    :param theta: The weight that needs to be learned.
    :param ax: The axes to plot on.
    """
    phi = np.vstack([real[0] ** i for i in range(args.order + 1)])

    if args.alg == 'br':  # bayesian estimation
        mu_post, sigma_post = theta
        mu_pred = phi.T.dot(mu_post)
        stddev_pred = np.sqrt([phi[:, i].dot(sigma_post).dot(phi[:, i]) for i in range(phi.shape[1])])
        interpolated = mu_pred  # use expectation of y
        print('The posterior mean of the parameters is: ' + args.arr2str(mu_post))  # print the posterior mean of theta
    else:  # point estimation
        interpolated = phi.T.dot(theta)
        print('The parameters are: ' + args.arr2str(theta))  # print the parameters, i.e., theta, for inspection

    # compute and print the RMSE between the true function and the interpolated polynomial
    diff = interpolated - real[1]
    mse = diff.dot(diff) / diff.shape[0]
    print('{}\nThe mean squared error is {:.4f}.'.format(args.conf_txt, mse))

    # draw the figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.plot(real[0], interpolated, 'b-', label='interpolation')
    if args.alg == 'br':  # plot the standard deviation
        ax.fill_between(real[0], interpolated - stddev_pred, interpolated + stddev_pred, label='stddev', color='lightblue')
    ax.plot(real[0], real[1], 'r-.', label='true function')
    ax.scatter(samp[0], samp[1], s=1.0, c='k', label='observation')
    ax.set_title(args.alg)
    ax.legend(loc='upper right', fontsize='x-small')

    return mse


def main(argv=None):
    args = parse_args(argv)
    sampx, sampy = read_data(sample=True)
    sampx, sampy = data_perturbation(args, sampx, sampy)
    realx, realy = read_data(sample=False)

    theta = poly_interpolate(args, sampx, sampy)
    analyze_result(args, [sampx, sampy], [realx, realy], theta)


def run_five_alg():
    common = '--order 5 --wd 5 --alpha 1 --var 5 --data-pct 0.1'
    to_do = [' --alg ' + alg for alg in ['ls', 'rls', 'lasso', 'br', 'rr']]

    sampx, sampy = read_data(sample=True)
    realx, realy = read_data(sample=False)
    fig = plt.figure()

    for i in range(len(to_do)):
        argv = common + to_do[i]
        args = parse_args(argv)

        if i == 0:
            sampx, sampy = data_perturbation(args, sampx, sampy)
        theta = poly_interpolate(args, sampx, sampy)

        ax = fig.add_subplot(2, 3, i + 1)
        analyze_result(args, [sampx, sampy], [realx, realy], theta, ax=ax)
    fig.tight_layout()
    # fig.savefig(r'D:\cityU\Courses\MachineLearning\PA01\submission\figures\question_b.pdf',
    #             format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    DATA_PATH = r'D:\cityU\Courses\MachineLearning\PA01\PA-1-data-text\PA-1-data-text'
    # main('--order 5 --alg rls --wd 5 --debug --alpha 1 --var 5')
    run_five_alg()
