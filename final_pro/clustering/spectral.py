import numpy as np
from scipy.spatial.distance import cosine
import ass02.part_a.kmeans as kmeans
import final_pro.utility as util
import matplotlib.pyplot as plt


def data_to_graph(x, adj_type='knn', **kwargs):
    """Compute an undirected graph representation of the given data, with the
    specific metric of computing similarities.

    :param x: Data of shape (N, D).
    :param adj_type: Which method to compute neighborhood relationships.
        Options are `knn', `epsilon' and `fully'.
    :return: Return the similarity graph, which is a (N, N) shaped matrix, of `x'.
    """
    N, D = x.shape

    feat_norm = np.sum(x**2, axis=1)  # (N,)
    feat_dot = x.dot(x.T)  # (N, N)
    dist = np.sqrt(feat_norm[:, None] - 2 * feat_dot + feat_norm[None, :] + 1e-4)
    arg_sort_dist = [set(row.tolist()) for row in np.argsort(dist, axis=1)[:, :kwargs['num_neighbor']]]
    del feat_dot

    graph = np.zeros(shape=(N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            weight = 0.
            if adj_type == 'epsilon':
                if dist[i, j] < kwargs['epsilon']:
                    weight = 1.
            elif adj_type == 'knn':
                if i in arg_sort_dist[j] or j in arg_sort_dist[i]:
                    weight = 1 - cosine(x[i], x[j])
            elif adj_type == 'fully':
                weight = np.exp(-dist[i, j] ** 2 / kwargs['sigma'])
            else:
                raise NotImplementedError

            graph[i, j] = graph[j, i] = weight

    return graph


def cluster(x, num_cluster, adj_type, num_epoch=100, **kwargs):
    """Apply spectral clustering to the data `x'.
    :param x: Data to cluster. Shape (N, D) where N is the number of samples, and D is the data dimension.
    :param num_cluster: The number of clusters.
    :param adj_type: The method to construct the adjacency matrix. [`knn', `epsilon', `fully'].
    :param num_epoch: The maximum number of epochs for KMeans.
    :keyword normalize: If specified, this arg determines which kind of normalized Laplacian to use. [`sym', `rw'].
    :keyword num_neighbor: If `adj' is `knn', this arg should be the number of nearest neighbors.
    :keyword epsilon: If `adj' is `epsilon', this arg should be the threshold of the distance within which
        two points are regarded adjacent.
    :keyword sigma: If `adj' is `fully', this arg specifies the denominator in the exponent in the RBF.
    :return: The assignment of each data point. Shape (N,). The cluster number starts from 0.
    """
    adj_graph = data_to_graph(x, adj_type=adj_type, **kwargs)
    print('Graph computed!')

    degree = np.sum(adj_graph, axis=0)
    laplacian = np.diag(degree) - adj_graph

    norm_lap = kwargs['normalize'] if 'normalize' in kwargs else None
    if norm_lap == 'sym':
        factor = np.diag(1 / np.sqrt(degree))
        laplacian = factor.dot(laplacian).dot(factor)
    elif norm_lap == 'rw':
        factor = np.diag(1 / degree)
        laplacian = factor.dot(laplacian)

    print('Begin eigen decomposition!')
    eival, eivec = np.linalg.eigh(laplacian)
    eivec = eivec[:, :num_cluster]  # shape (N, K)
    args = util.Config(num_cluster=num_cluster, num_epoch=num_epoch, debug=True)
    assign, _ = kmeans.receiver(args=args, x=eivec)

    return assign, adj_graph


if __name__ == '__main__':
    from ass02.part_a.main import read_data
    from ass02.part_a.visualize import plot

    toy_data = read_data()
    for i in range(3):
        x, y = toy_data[i], toy_data[i+3]
        assign, graph = cluster(x, num_cluster=4, adj_type='knn', num_neighbor=9, epsilon=3, sigma=1, normalize='sym')

        fig = plot(x, y, assign)
        ax = fig.get_axes()[1]
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                if graph[i, j] < 1e-3:
                    continue
                ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], 'k-', linewidth=0.1)
    plt.show()
