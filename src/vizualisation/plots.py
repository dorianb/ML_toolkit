import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import itertools


def confidence_interval(dataframe, metric=["perc_error"], variables=[], xlim=[1, 24],
                        ylim=800, xlabel=None):
    plt.close('all')
    xlim = list(xlim)

    for var in variables:

        fig, ax = plt.subplots(figsize=(15, 5))
        ax = dataframe.boxplot(column=metric, by=[var], ax=ax)

        if len(xlim) > 0:
            ax.set_xlim(xlim)
            if xlabel:
                plt.xticks(np.arange(xlim[0] + 1, xlim[1]), xlabel)
        else:
            var_values = sorted(dataframe[var].astype(int).unique())
            if xlabel:
                plt.xticks(np.arange(1, len(var_values) + 1), np.array(xlabel)[var_values])

        ax.set_ylim(0, ylim)

    plt.show()


def plot_along(df, y_true="", y_pred="", metrics=["recall"],
               variable="", xlim=24, ylim=1.0):
    plt.close('all')

    fig, ax = plt.subplots(figsize=(20, 5))

    x_range = np.arange(1, xlim + 1)

    recall = np.zeros(len(x_range))
    precision = np.zeros(len(x_range))
    specificity = np.zeros(len(x_range))

    for i, x in enumerate(x_range):
        tp = len(df.loc[(df[variable] == x) & (df[y_true] == df[y_pred]) & (df[y_true] == 1)])
        fp = len(df.loc[(df[variable] == x) & (df[y_true] != df[y_pred]) & (df[y_true] == 0)])
        tn = len(df.loc[(df[variable] == x) & (df[y_true] == df[y_pred]) & (df[y_true] == 0)])
        fn = len(df.loc[(df[variable] == x) & (df[y_true] != df[y_pred]) & (df[y_true] == 1)])

        specificity[i] = float(tn) / (tn + fp) if tn + fp != 0 else np.nan
        recall[i] = float(tp) / (tp + fn) if tp + fn != 0 else np.nan
        precision[i] = float(tp) / (tp + fp) if tp + fp != 0 else np.nan

    for metric in metrics:
        if metric == "recall":
            plt.plot(x_range, recall, label=metric)
        elif metric == "precision":
            plt.plot(x_range, precision, label=metric)
        elif metric == "specificity":
            plt.plot(x_range, specificity, label=metric)

    ax.set_xlim(1, xlim)
    ax.set_ylim(1, ylim)

    plt.xticks(x_range)
    plt.yticks(np.arange(0, ylim + 0.1, step=0.1))
    plt.legend(loc="best")
    plt.xlabel(variable)
    plt.ylabel(u"Metrique de classification")
    plt.title(u"Metrique de classification de l'ecoulement des stocks en fonction de l'heure de vente")

    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def confusion_matrix(y, y_pred, class_names=["Non", "Oui"]):
    # Compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(y, y_pred, labels=[0, 1])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

