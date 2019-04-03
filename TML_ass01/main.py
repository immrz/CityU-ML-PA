import os
import random
import time
from TML_ass01 import model, visualize
import matplotlib.pyplot as plt

DATA_FOLDER = r'D:\cityU\Courses\Topics-in-ML\assignment1\social-data'
# random.seed(0)


def prepare_data(user):
    if type(user) == int:
        to_read = os.path.join(DATA_FOLDER, 'user_{:d}.csv'.format(user))
        data = read_csv(to_read)
        """shuffling and segmentation, 8:2 for training and testing"""
        random.shuffle(data)
        n_test = len(data) // 5
        return data[n_test:], data[:n_test]

    elif user == 'all':
        train_data, test_data = [], []
        for i in range(1, 11):
            to_read = os.path.join(DATA_FOLDER, 'user_{:d}.csv'.format(i))
            data = read_csv(to_read)
            if i <= 8:
                train_data.extend(data)
            else:
                test_data.extend(data)
        return train_data, test_data

    else:
        raise ValueError


def read_csv(fname):
    data = []
    with open(fname, 'r') as fi:
        for line in fi:
            x, y, t = map(float, line.strip().split(','))
            data.append((x, y, t))
    return data


def main():
    t_start = time.time()

    train_data, test_data = prepare_data(user=1)
    vm_reg = model.VonMisesRegression(train_data, 'rbf', sigma=0.05, c=1, order=5, nr_iter=50)
    vm_reg.train_model()
    f_expt = vm_reg.infer_f(test_data)

    # prediction = vm_reg.infer_y(n_samp=10)
    # prediction = vm_reg.infer_y_descent(n_samp=15)
    prediction = vm_reg.infer_y_naive()
    rmse, rmse_rdm = vm_reg.compute_error(prediction)
    print('The RMSE is {:f}. The baseline is {:f}.'.format(rmse, rmse_rdm))

    fig = plt.figure()
    ax = fig.add_subplot(221)
    visualize.plot_labeled_data(train_data, ax=ax, marker='+')
    visualize.plot_labeled_data(test_data, ax=ax, show_map=False, marker='o')

    ax = fig.add_subplot(222)
    visualize.vis_latent(vm_reg.f_post[0], train_data, ax=ax, marker='+')
    visualize.vis_latent(f_expt, test_data, ax=ax, show_map=False, marker='o')

    ax = fig.add_subplot(223)
    visualize.plot_labeled_data(test_data, ax=ax, label=vm_reg.ans, marker='o')

    plt.show()

    print('Program ended within {:f} seconds'.format(time.time() - t_start))


if __name__ == '__main__':
    main()
