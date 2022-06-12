# -*- coding: utf-8 -*-
# @Time : 2022/5/17 20:22
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_curve_for_train_and_test(x, y_train, y_test,
                                  save_path: str = None, parameter_dict: dict = None, rc: dict = None):
    """ Plot (accuracy/loss) curves for train/test process.

    :param x: array. x-axis values, usually epochs.
    :param y_train: array. y-axis values for training. If None, not plot training result.
    :param y_test: array. y-axis values for testing. If None, not plot testing result.
    :param save_path: str. the save path for the plot. If None, show the figure in window.
    :param parameter_dict:
    :param rc: dict. rc dict for matplotlib.
    """

    if x is None:
        x = np.arange(len(y_train))

    default_rc = {
        'lines.linewidth': 2,
    }
    if rc is not None:
        default_rc.update(rc)

    default_parameter_dict = {
        'train_label': 'Train',
        'test_label': 'Test',
    }
    default_parameter_dict = defaultdict(lambda: None, **default_parameter_dict)
    if parameter_dict is not None:
        default_parameter_dict.update(parameter_dict)

    with mpl.rc_context(default_rc):
        fig, ax = plt.subplots()
        if y_train is not None:
            line_train = ax.plot(x, y_train, label=default_parameter_dict['train_label'])
        else:
            next(ax._get_lines.prop_cycler)

        if y_test is not None:
            line_test = ax.plot(x, y_test, label=default_parameter_dict['test_label'])
        else:
            next(ax._get_lines.prop_cycler)
        ax.set(title=default_parameter_dict['title'],
               xlabel=default_parameter_dict['xlabel'], ylabel=default_parameter_dict['ylabel'])
        ax.legend()

        if save_path is None:
            fig.show()
        else:
            fig.savefig(save_path)


# Copied from matplotlib document.
# Origin link: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, vmin=0., vmax=1.)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    cbar = plt.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.set_yticks([i / 10. for i in range(0, 11, 2)])
    cbar.ax.set_yticklabels(["{:.0%}".format(i) for i in cbar.get_ticks()])  # set ticks of your format

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("Predicted Categories", fontsize=15, labelpad=8.)
    ax.set_ylabel("Actual Categories", fontsize=15, labelpad=8.)

    return im, cbar


# Copied from matplotlib document.
# Origin link: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_heatmap(data, row_labels, col_labels, save_path: str = None,
                heatmap_kwargs: dict = None, annotator_kwargs: dict = None, rc: dict = None):
    """Get the heatmap for data."""

    default_heatmap_kwargs = {
        "cmap": "Blues",
        "cbarlabel": None
    }
    if heatmap_kwargs is not None:
        default_heatmap_kwargs.update(heatmap_kwargs)
    default_heatmap_kwargs = defaultdict(lambda: None, **default_heatmap_kwargs)

    default_annotator_kwargs = {
        "valfmt": "{x:.2%}",
    }
    if annotator_kwargs is not None:
        default_annotator_kwargs.update(annotator_kwargs)
    default_annotator_kwargs = defaultdict(lambda: None, **default_annotator_kwargs)

    default_rc = {
        'figure.figsize': (10, 10),
    }
    if rc is not None:
        default_rc.update(rc)

    with mpl.rc_context(default_rc):
        fig, ax = plt.subplots()
        im, cbar = heatmap(data, row_labels, col_labels, ax=ax, **default_heatmap_kwargs)

        texts = annotate_heatmap(im, **default_annotator_kwargs)

    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path)


if __name__ == '__main__':
    pass
    # import numpy as np
    #
    # name = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # data = np.random.randn(10, 10)
    # data -= np.min(data)
    # get_heatmap(data, row_labels=name, col_labels=name)
