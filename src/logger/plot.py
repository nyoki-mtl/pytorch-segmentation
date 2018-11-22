import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def history_ploter(history, path):
    history = np.asarray(history)
    title = path.name[:-4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if history.ndim == 1:
        ax.plot(np.arange(len(history)), history)
    else:
        ax.plot(np.arange(len(history)), history[:, 0], label='train')
        ax.plot(np.arange(len(history)), history[:, 1], label='valid')
        ax.legend()
    ax.set_title(title)
    plt.savefig(str(path))
    plt.close()
