import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from PIL import Image
from TML_ass01.model import inverse_trigonometric

_map = Image.open(r'D:\cityU\Courses\Topics-in-ML\assignment1\social-data\map.png').convert('L')


def plot_labeled_data(data, ax=None, show_map=True, marker='+', label=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x, y, t = zip(*data)
    if show_map:
        ax.imshow(_map, zorder=0, cmap='gray')

    if label is not None:
        t = label
    ax.scatter(x, y, c=t, marker=marker, zorder=1, cmap='jet')


def vis_latent(f, raw_data, ax=None, show_map=True, marker='+'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    f = f.reshape((2, -1))
    hue = np.array(inverse_trigonometric(f[1], f[0])) / (2 * np.pi)
    value = np.sqrt(np.sum(f**2, axis=0))
    value = (value - np.min(value)) / np.max(value)
    hsv = np.stack((hue, np.ones(shape=hue.shape), value), axis=-1)  # (N, 3)
    rgb = colors.hsv_to_rgb(hsv)

    x, y, t = zip(*raw_data)
    if show_map:
        ax.imshow(_map, zorder=0, cmap='gray')
    ax.scatter(x, y, c=rgb, zorder=1, marker=marker)
