import os
import numpy as np
from ass02.part_a import kmeans, mean_shift
from final_pro import utility
from final_pro.clustering import spectral
import time
from sklearn import metrics
import argparse
import scipy.cluster.vq as vq
import multiprocessing as mp


def load_feat_and_target(folder):
    feat_file = os.path.join(folder, 'MNIST_TestSet10K_DimReduced.npy')
    target_file = os.path.join(folder, 'MNIST_TestSet10K_GroundTruths.npy')

    features = np.load(feat_file, allow_pickle=False)
    targets = np.load(target_file, allow_pickle=False)
    return features, targets


def main(cmd=None):
    args = parse_args(cmd=cmd)
    features, targets = load_feat_and_target(args.folder)
    if args.whiten:
        features = vq.whiten(features)
    targets = targets.astype(np.int32)

    print('Number {:d} started!'.format(args.save_postfix))

    num_cluster = 10
    since = time.time()

    if args.alg == 'kmeans':
        sub_args = utility.Config(num_cluster=num_cluster, num_epoch=args.num_epoch, debug=False)
        assign, _ = kmeans.receiver(sub_args, features)

    elif args.alg == 'meanshift':
        sub_args = utility.Config(num_epoch=args.num_epoch, debug=False, bandwidth=args.bandwidth)
        assign, _ = mean_shift.receiver(sub_args, features)

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
    ret_msg = '{:02d}: It takes {:.0f}h {:.0f}m {:.0f}s to finish. The Adjusted Random Index is {:.4f}'.format(
        args.save_postfix, elapsed // 3600, elapsed % 3600 // 60, elapsed % 60, adj_rand_idx)

    if args.save_path is not None:
        try:
            np.savetxt(os.path.join(args.save_path, 'res_{:02d}.txt'.format(args.save_postfix)), assign)
        except:
            print('Saving failed!')
    return ret_msg, assign


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
    parser.add_argument('--save-path', default=None, type=str, help='The folder to save results.')
    parser.add_argument('--save-postfix', type=int, help='The postfix of the saved result file.')

    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd.split())

    return args


def multi_args():
    choices = ['../ spectral --adj-type knn --num-neighbor 50 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 60 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 70 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 80 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 90 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 100 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 110 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 120 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 130 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 140 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 150 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 160 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 170 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 180 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 190 --norm-lap sym',
               '../ spectral --adj-type knn --num-neighbor 200 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 8 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 9 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 10 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 11 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 12 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 13 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 14 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 15 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 16 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 17 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 18 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 19 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 20 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 21 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 22 --norm-lap sym',
               '../ spectral --adj-type epsilon --epsilon 23 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.1 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.2 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.3 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.4 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.5 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.6 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.7 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.8 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 0.9 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.0 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.1 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.2 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.3 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.4 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.5 --norm-lap sym',
               '../ spectral --adj-type fully --sigma 1.6 --norm-lap sym']
    choices = list([c + ' --save-path ../results --save-postfix {:d}'.format(i) for i, c in enumerate(choices)])
    pool = mp.Pool(processes=48)
    multi_res = pool.map(main, choices)
    pool.close()
    pool.join()

    for msg, _ in multi_res:
        print(msg)


if __name__ == '__main__':
    # main()
    multi_args()
