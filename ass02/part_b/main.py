from ass02.part_a import kmeans, expect_max, mean_shift
from ass02.part_a.main import parse_args
from ass02.part_b import pa2
from PIL import Image
import os
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq


def segmentation(args, img):
    X, L = pa2.getfeatures(img, stepsize=7)
    if args.feat_norm:
        X = vq.whiten(X.T).T
    elif args.feat_scale is not None:
        X = (X.T * args.feat_scale).T

    if args.alg == 'kmeans':
        assign, _ = kmeans.receiver(args, X.T)
    elif args.alg == 'em':
        assign, _ = expect_max.receiver(args, X.T)
    elif args.alg == 'meanshift':  # bandwidh = 0.05 does well
        assign, _ = mean_shift.receiver(args, X.T)
    else:
        raise ValueError('Unsupported algorithms!')

    assign = assign + 1
    seg = pa2.labels2seg(assign, L)
    color_seg = pa2.colorsegms(seg, img)
    return seg, color_seg


def run_per_image(args, img, num_rows=1, cur_row=0, fig=None):
    if fig is None:
        fig = plt.figure()
    num_cols = 3
    seg, color_seg = segmentation(args, img)

    ax = fig.add_subplot(num_rows, num_cols, cur_row*num_cols+1)
    ax.imshow(img)

    ax = fig.add_subplot(num_rows, num_cols, cur_row*num_cols+2)
    ax.imshow(seg)

    ax = fig.add_subplot(num_rows, num_cols, cur_row*num_cols+3)
    ax.imshow(color_seg)


def main(inst=None, img_names=('56028.jpg',)):
    args = parse_args(inst)
    folder = r'D:\cityU\Courses\MachineLearning\PA02\PA2-cluster-images\images'
    img_paths = [os.path.join(folder, name) for name in img_names]

    fig = plt.figure()
    num_rows = len(img_names)

    for i, path in enumerate(img_paths):
        img = Image.open(path)
        run_per_image(args, img, num_rows=num_rows, cur_row=i, fig=fig)

    plt.tight_layout()
    if args.save_fig:
        save_path = '{:s}_{:s}_img_seg.pdf'.format(args.alg, args.save_fig)
        save_path = os.path.join(r'D:\cityU\Courses\MachineLearning\PA02\submission\figures', save_path)
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    selected_imgs = ['12003.jpg']#, '299086.jpg', '117054.jpg', '370036.jpg']
    main('--alg meanshift --bandwidth 50 --num-epoch 100 --num-cluster 20 --debug --feat-scale (100,2)',
         img_names=selected_imgs)
