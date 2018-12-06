import numpy as np
from ass01.part_a import regression, lasso_solver, robust_solver
import os
import matplotlib.pyplot as plt


def read_data(folder, for_train=False):
    x_file = 'count_data_{}x.txt'.format('train' if for_train else 'test')
    y_file = 'count_data_{}y.txt'.format('train' if for_train else 'test')

    files = [os.path.join(folder, f) for f in [x_file, y_file]]
    ret = []

    for f in files:
        with open(f, 'r') as fi:
            mat = []
            for line in fi:
                vec = list(map(float, line.strip().split()))
                mat.append(vec)
            mat = np.array(mat)
            if mat.shape[1] == 1:
                mat = mat.flatten()
            ret.append(mat)

    return ret


def exp_transform(x):
    x = x.copy()
    x[x >= 6] = 6
    x[x <= -6] = -6
    return np.exp(x)


def quadratic_transform(x):
    D, N = x.shape
    new_D = D * (D + 1) // 2 + D
    new_x = np.zeros(shape=(new_D, N))
    new_x[:D, :] = x.copy()

    idx = [(i, j) for i in range(D) for j in range(i, D)]
    idx = [i * D + j for i, j in idx]
    assert len(idx) == new_D - D

    for i in range(N):
        point = x[:, i].copy()
        outer = np.outer(point, point).flatten()
        new_x[D:, i] = outer[idx]

    return new_x


def calc_err(pred_y, true_y):
    diff = pred_y - true_y
    square = diff ** 2
    mae = np.average(np.fabs(diff))
    mse = np.average(square)
    return mae.item(), mse.item()


def main(wd=0.1, num_epoch=1000, alpha=1, var=10):
    folder = r'D:\cityU\Courses\MachineLearning\PA01\PA-1-data-text\PA-1-data-text'
    train_x, train_y = read_data(folder, for_train=True)
    test_x, test_y = read_data(folder, for_train=False)
    # train_x, test_x = quadratic_transform(train_x), quadratic_transform(test_x)
    train_x, test_x = exp_transform(train_x), exp_transform(test_x)

    train_x = np.vstack([np.ones(shape=(1, train_x.shape[1])), train_x])
    test_x = np.vstack([np.ones(shape=(1, test_x.shape[1])), test_x])

    dim = train_x.shape[0]
    num_data = test_x.shape[1]
    params = []
    algorithms = ['ls', 'rls', 'lasso', 'br', 'rr']
    fig = plt.figure()

    for alg in algorithms:
        if alg == 'ls':
            theta = np.linalg.inv(train_x.dot(train_x.T)).dot(train_x.dot(train_y))
        elif alg == 'rls':
            theta = np.linalg.inv(train_x.dot(train_x.T) + wd * np.eye(dim)).dot(train_x.dot(train_y))
        elif alg == 'lasso':
            theta = lasso_solver.soft_threshold(phi=train_x, y=train_y, num_epoch=num_epoch, wd=wd)
        elif alg == 'br':
            sigma_post = np.linalg.inv(np.eye(dim) / alpha + train_x.dot(train_x.T) / var)
            mu_post = sigma_post.dot(train_x.dot(train_y)) / var
            theta = mu_post
        else:  # rr
            theta = robust_solver.robust_regression(phi=train_x, y=train_y, num_epoch=num_epoch)

        params.append(theta)

    for i, param in enumerate(params):
        pred_y = test_x.T.dot(param)
        pred_y = np.around(pred_y)
        err = calc_err(pred_y, test_y)
        print('{:8s}: MAE: {:.4f}; MSE: {:.4f}'.format(algorithms[i], err[0], err[1]))

        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(range(num_data), pred_y, 'b-', linewidth=1.0, label='prediction')
        ax.plot(range(num_data), test_y, 'r-.', linewidth=1.0, label='true label')
        ax.legend(loc='upper right', fontsize='x-small')
        ax.set_title(algorithms[i])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.28)
    plt.show()


if __name__ == '__main__':
    main()
