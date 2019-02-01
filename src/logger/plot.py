import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def history_ploter(history, path):
    history = np.asarray(history)
    title = path.name[:-4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(history))
    if history.ndim == 1:
        y = history
        ax.plot(x[y != None], y[y != None])
    else:
        y = history[:, 0]
        ax.plot(x[y != None], y[y != None], label='train')
        y = history[:, 1]
        ax.plot(x[y != None], y[y != None], label='valid')
        ax.legend()
    ax.set_title(title)
    plt.savefig(str(path))
    plt.close()
