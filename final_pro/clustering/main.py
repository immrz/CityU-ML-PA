import os
import numpy as np
from ass02.part_a import kmeans
from final_pro import utility
from final_pro.clustering import spectral
import time
from sklearn import metrics


def load_feat_and_target(folder):
    feat_file = os.path.join(folder, 'MNIST_TestSet10K_DimReduced.npy')
    target_file = os.path.join(folder, 'MNIST_TestSet10K_GroundTruths.npy')

    features = np.load(feat_file, allow_pickle=False)
    targets = np.load(target_file, allow_pickle=False)
    return features, targets


def main():
    features, targets = load_feat_and_target(r'D:\cityU\Courses\MachineLearning\Project')
    targets = targets.astype(np.int32)
    args = utility.Config(num_cluster=10, debug=True)

    since = time.time()
    # assign, _ = kmeans.receiver(args, features)
    assign, _ = spectral.cluster(features, num_cluster=10, adj_type='knn', num_neighbor=100, normalize='sym')
    elapsed = time.time() - since

    adj_rand_idx = metrics.adjusted_rand_score(labels_true=targets, labels_pred=assign)
    print('It takes {:.0f}h {:.0f}m {:.0f}s to finish\nThe Adjusted Random Index is {:.4f}'.format(
        elapsed // 3600, elapsed % 3600 // 60, elapsed % 60, adj_rand_idx))
    return elapsed, assign


if __name__ == '__main__':
    main()
