import numpy as np
import matplotlib as mpl
import itertools

import matplotlib.pyplot as plt

def plot_cmap(labels, data, ax, chart_type='bar', cmap='viridis'):
    scaled_data = np.linspace(0.05, 1, len(data))
    colors = [mpl.colormaps.get_cmap(cmap)(decimal) for decimal in scaled_data]
    fig.patch.set_facecolor('white')

    try:
        getattr(ax, chart_type)(labels, data, color=colors)
    except AttributeError:
        getattr(ax, chart_type)(data, labels=labels, colors=colors)

def plot_confusion(Y, ax, matrix, title='Confusion Matrix', rf='.0f', shrink=0.3):
    im = ax.imshow(matrix, interpolation='nearest', cmap='viridis')
    ax.set_xticks(range(np.unique(Y).shape[0]))
    ax.set_yticks(range(np.unique(Y).shape[0]))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('True label', fontsize=14)
    ax.set_ylabel('Predicted label', fontsize=14)
    cbar = plt.colorbar(im, ax=ax, shrink=shrink)
    for m, n in itertools.product(range(matrix.shape[1]), range(matrix.shape[0])):
        ax.text(n, m, format(matrix[m, n], rf), ha="center", va="center", color="w")

random_state = 7