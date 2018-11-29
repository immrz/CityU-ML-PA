from PIL import Image
import matplotlib.pyplot as plt
from ass02.part_b.main import segmentation
from ass02.part_a.main import parse_args
import os


def compare_kmeans_em():
    num_of_clusters = [2, 4, 20, 80]
    fig = plt.figure()

    for i, k in enumerate(num_of_clusters):
        _, res1 = segmentation(parse_args('--alg kmeans --num-cluster {:d} --num-epoch 50'.format(k)), img)
        _, res2 = segmentation(parse_args('--alg em --num-cluster {:d} --num-epoch 50'.format(k)), img)

        ax = fig.add_subplot(2, len(num_of_clusters), i + 1)
        ax.imshow(res1)
        ax.set_title('k={:d}'.format(k))
        if i == 0:
            ax.set_ylabel('KMeans', rotation='horizontal')

        ax = fig.add_subplot(2, len(num_of_clusters), i + 1 + len(num_of_clusters))
        ax.imshow(res2)
        if i == 0:
            ax.set_ylabel('EM', rotation='horizontal')

    fig.set_size_inches(6.4, 2.8)
    plt.tight_layout()
    fig.savefig(r'D:\cityU\Courses\MachineLearning\PA02\submission\figures\kmeans_em_cmp.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()


def tune_bandwidth():
    bw = [1, 0.5, 0.1, 0.05]
    fig = plt.figure()

    for i, h in enumerate(bw):
        _, res = segmentation(parse_args('--alg meanshift --bandwidth {:f} --num-epoch 200 --debug'.format(h)), img)

        ax = fig.add_subplot(1, len(bw), i + 1)
        ax.imshow(res)
        ax.set_title('h={:.2f}'.format(h))
        if i == 0:
            ax.set_ylabel('MeanShift', rotation='horizontal')

    plt.tight_layout()
    fig.savefig(r'D:\cityU\Courses\MachineLearning\PA02\submission\figures\meanshift_cmp.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()


def scale_features(alg):
    select_imgs = ['12003.jpg', '370036.jpg']
    folder = r'D:\cityU\Courses\MachineLearning\PA02\PA2-cluster-images\images'
    images = [Image.open(os.path.join(folder, i)) for i in select_imgs]

    fig = plt.figure()
    scale = ' --feat-scale (100,2)'
    inst = '--alg {} --num-epoch 100 --num-cluster 20 --bandwidth 50 --debug'.format(alg)

    for i, image in enumerate(images):
        _, res1 = segmentation(parse_args(inst), image)
        _, res2 = segmentation(parse_args(inst + scale), image)

        ax = fig.add_subplot(2, len(images), i + 1)
        ax.imshow(res1)
        if i == 0:
            ax.set_ylabel('origin', rotation='horizontal')

        ax = fig.add_subplot(2, len(images), i + 1 + len(images))
        ax.imshow(res2)
        if i == 0:
            ax.set_ylabel('scaled', rotation='horizontal')

    plt.tight_layout()
    fig.savefig(r'D:\cityU\Courses\MachineLearning\PA02\submission\figures\{}_feat_scale.pdf'.format(alg),
                format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    img_name = r'D:\cityU\Courses\MachineLearning\PA02\PA2-cluster-images\images\12003.jpg'
    img = Image.open(img_name)
    compare_kmeans_em()
    # tune_bandwidth()
    # scale_features('meanshift')
