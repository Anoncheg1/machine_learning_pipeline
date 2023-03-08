import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def rename_columns(p_or_df, columns: dict, len=8):
    """
    fields_dict = {'SRC_BANK': 'Наименование источника',
               'APP_SRC_REF': 'ID источника',
               'DEAL_ID': 'ID сделки ХД',
               'CLIENT_ID': 'ID клиента ХД"',
               'DEAL_SRC_REF': 'ID сделки ЦФТ'}

    :param p:
    :param columns:
    :param len:
    :return:
    """
    from chepelev_pack.common import save, load
    if isinstance(p_or_df, str):
        df = load(p_or_df)
    else:
        df = p_or_df
    new_columns = columns.copy()
    for k, v in columns.items():
        new_columns[k] = v[:len]
    df.rename(columns=new_columns, inplace=True)
    if isinstance(p_or_df, str):
        return save('renamed.pickle', df)
    else:
        return df


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        # group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)] # old
        v00 = cf[0, 0] / np.sum(cf[:,0])
        v10 = cf[1, 0] / np.sum(cf[:, 0])
        v01 = cf[0, 1] / np.sum(cf[:, 1])
        v11 = cf[1, 1] / np.sum(cf[:, 1])
        group_percentages = ["{0:.2%}".format(value) for value in (v00, v01, v10, v11)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def make_confusion_matrix_percent_all(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def histogram_two_in_one(feature_main: str, feature_binary: str, p:str = None, df: pd.DataFrame = None, bins=20, density=True):
    """
    plot two histogram at one for two features
    one of them - binary [0,1]
    :param df:
    :param feature_main: continuous
    :param feature_binary: [0,1]
    :return:
    """
    if p is not None:
        df: pd.DataFrame = pd.read_pickle(p)

    f, ax = plt.subplots(figsize=(6, 4))
    df_1 = df[df[feature_binary] == 1][feature_main]
    df_0 = df[df[feature_binary] == 0][feature_main]
    # print(feature_main, len(df_1), len(df_0))
    # print(df_0.describe())
    # print(df_1.describe())
    # plt.hist(x=df_1, bins=10, color='red', alpha=0.6, normed=True)
    df_0.hist(ax=ax, bins=bins, color='red', alpha=0.6, density=density, label=f'{feature_binary}==0')  # , stacked=True
    df_1.hist(ax=ax, bins=bins, color='green', alpha=0.6, density=density, label=f'{feature_binary}==1')  # , stacked=True
    plt.legend()
    title = f"Гистограмма для {feature_main} по {feature_binary}"
    plt.title(title, {'fontsize': 20})
    # plt.show()
    plt.savefig('hist_norm ' + feature_main)
    # plt.close()


def plot_grid_of_plots(df, title: str = "grid of plots"):
    import math
    cols = 6

    # df: pd.DataFrame = pd.read_pickle(p)

    rows = math.ceil((df.columns.values.shape[0] - 1) / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(19, 10))
    plt.subplots_adjust(left=0.076, right=0.96, bottom=0.04, top=0.96, wspace=0.30, hspace=0.85)  # hspace = wspace ||
    for i in range(rows):
        for j in range(cols):
            n1 = i * cols + j - 1
            if n1 > len(df.columns)-1:
                break
            c = df.columns[n1]
            # print(c)
            # ax[i][j].plot(list(range(df.shape[0])), df[c], lw=1)
            ax[i][j].hist(df[c], 40)
            ax[i][j].set_title(c, pad=15 * (j % 2))
            ax[i][j].tick_params(axis='x', rotation=15)

    plt.savefig(title)
    # plt.show()

