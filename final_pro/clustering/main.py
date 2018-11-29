import os
import numpy as np
from ass02.part_a import kmeans, mean_shift
from final_pro import utility
from final_pro.clustering import spectral
import time
from sklearn import metrics
import argparse
import scipy.cluster.vq as vq


def load_feat_and_target(folder):
    feat_file = os.path.join(folder, 'MNIST_TestSet10K_DimReduced.npy')
    target_file = os.path.join(folder, 'MNIST_TestSet10K_GroundTruths.npy')

    features = np.load(feat_file, allow_pickle=False)
    targets = np.load(target_file, allow_pickle=False)
    return features, targets


def main():
    args = parse_args()
    features, targets = load_feat_and_target(args.folder)
    if args.whiten:
        features = vq.whiten(features)
    targets = targets.astype(np.int32)

    num_cluster = 10
    since = time.time()

    if args.alg == 'kmeans':
        args = utility.Config(num_cluster=num_cluster, num_epoch=args.num_epoch, debug=True)
        assign, _ = kmeans.receiver(args, features)

    elif args.alg == 'meanshift':
        args = utility.Config(num_epoch=args.num_epoch, debug=False, bandwidth=args.bandwidth)
        assign, _ = mean_shift.receiver(args, features)

    elif args.alg == 'spectral':
        kw = {'num_epoch': args.num_epoch}
        if args.adj_type == 'knn':
            kw['num_neighbor'] = args.num_neighbor if args.num_neighbor is not None \
                else int(np.sqrt(features.shape[0]))
        elif args.adj_type == 'epsilon':
            kw['epsilon'] = args.epsilon
        elif args.adj_type == 'fully':
            kw['sigma'] = args.sigma
        if args.norm_lap is not None:
            kw['normalize'] = args.norm_lap
        assign, _ = spectral.cluster(features, num_cluster=num_cluster, adj_type=args.adj_type, **kw)

    else:
        raise NotImplementedError

    elapsed = time.time() - since
    adj_rand_idx = metrics.adjusted_rand_score(labels_true=targets, labels_pred=assign)
    print('It takes {:.0f}h {:.0f}m {:.0f}s to finish\nThe Adjusted Random Index is {:.4f}'.format(
        elapsed // 3600, elapsed % 3600 // 60, elapsed % 60, adj_rand_idx))

    print('Do you want to save the assignment?')
    save_path = input().strip()
    if save_path != 'no':
        try:
            np.savetxt(os.path.join(save_path, 'assignment.txt'), assign)
        except:
            print('Saving failed!')
    return elapsed, assign


def parse_args(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='The folder to find the features and ground-truths.')
    parser.add_argument('alg', type=str, choices=['kmeans', 'meanshift', 'spectral'], help='The algorithm to use.')

    parser.add_argument('--whiten', action='store_true', help='If set, normalize the features first.')
    parser.add_argument('--num-epoch', default=100, type=int, help='The maximum iterations.')

    parser.add_argument('--bandwidth', default=0.1, type=float, help='The bandwidth of the kernels for meanshift.')

    parser.add_argument('--adj-type', default='knn', choices=['knn', 'epsilon', 'fully'], type=str,
                        help='The method to compute the adjacency matrix for Spectral Clustering.')
    parser.add_argument('--num-neighbor', default=None, type=int, help='The threshold of the knn adjacency.')
    parser.add_argument('--epsilon', default=0.1, type=float, help='The threshold of the epsilon adjacency.')
    parser.add_argument('--sigma', default=1., type=float, help='The threshold of the fully adjacency.')
    parser.add_argument('--norm-lap', default=None, choices=['sym', 'rw', None], type=str,
                        help='If not None, specify which kind of normalized Laplacian to use.')

    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd.split())

    return args


if __name__ == '__main__':
    main()
