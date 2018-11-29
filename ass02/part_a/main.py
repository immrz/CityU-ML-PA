import numpy as np
from ass02.part_a import kmeans, expect_max, mean_shift
import argparse
import os
import ass02.part_a.visualize as vis
import matplotlib.pyplot as plt


def read_data():
    folder = r'D:\cityU\Courses\MachineLearning\PA02\PA2-cluster-data\cluster_data_text'
    files = ['cluster_data_data{}.txt'.format(f) for f in ['A_X', 'B_X', 'C_X', 'A_Y', 'B_Y', 'C_Y']]
    data = []
    for file in files:
        with open(os.path.join(folder, file), 'r') as fi:
            mat = []
            for line in fi:
                numbers = list(map(float, line.split()))
                mat.append(numbers)
            mat = np.array(mat).T
            if mat.shape[1] == 1:
                mat = mat.flatten().astype(np.int32)
            data.append(mat)
    return data


def parse_args(inst=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cluster', type=int, default=4)
    parser.add_argument('--num-epoch', type=int, default=10000)
    parser.add_argument('--alg', choices=['kmeans', 'em', 'meanshift'], default='kmeans')
    parser.add_argument('--distribution', choices=['gaussian', 'poisson'], default='gaussian')
    parser.add_argument('--var', type=float, default=1)
    parser.add_argument('--bandwidth', type=float, default=1.7)
    parser.add_argument('--tune-bw', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save-fig', type=str, default=None, help='The file name to save for the figure.')
    parser.add_argument('--feat-scale', type=str, default=None, help='Rescale the input features.')
    parser.add_argument('--feat-norm', action='store_true')
    if inst is None:
        arguments = parser.parse_args()
    else:
        arguments = parser.parse_args(inst.split())

    if arguments.feat_scale is not None:
        factors = list(map(float, arguments.feat_scale.strip('()').split(',')))
        assert len(factors) == 2, 'You should only specify two scale factors!'
        arguments.feat_scale = np.array([factors[0], factors[0], factors[1], factors[1]])
    return arguments


def tune_bandwidth(args, x):
    subplot_width = 2
    subplot_height = 2

    candidates = [0.5, 1, 2, 5]
    titles = ['h={}'.format(c) for c in candidates]
    fig = plt.figure()

    for i, (bandwidth, title) in enumerate(zip(candidates, titles)):
        args.bandwidth = bandwidth
        assign, mean = mean_shift.receiver(args, x)
        fig = vis.plot_different_params(x, assign, layout=(subplot_height, subplot_width, i + 1),
                                        title=title, mean=mean, cur_fig=fig)
    plt.tight_layout()
    fig.savefig(r'D:\cityU\Courses\MachineLearning\PA02\submission\figures\ms_sensitivity.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()


def main(inst=None):
    args = parse_args(inst=inst)
    all_data = read_data()

    if args.tune_bw:
        tune_bandwidth(args, all_data[2])
        return

    for i in range(3):
        which_data = i
        x, y = all_data[which_data], all_data[which_data + 3]

        if args.alg == 'kmeans':
            assign, mean = kmeans.receiver(args, x)
            kwargs = {'mean': mean}

        elif args.alg == 'em':
            assign, theta = expect_max.receiver(args, x)
            kwargs = {'mean': theta[1], 'var': theta[2]}

        elif args.alg == 'meanshift':
            assign, mean = mean_shift.receiver(args, x)
            kwargs = {'mean': mean}

        else:
            raise NotImplementedError

        if args.save_fig is not None:
            file_name = '{:s}_{:s}_data{:02d}.pdf'.format(args.alg, args.save_fig, i + 1)
            kwargs['save_fig'] = os.path.join(r'D:\cityU\Courses\MachineLearning\PA02\submission\figures', file_name)

        vis.plot(x, y, assign, **kwargs)
    plt.show()


if __name__ == '__main__':
    main('--alg meanshift --tune-bw')
