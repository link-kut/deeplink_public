import matplotlib.pyplot as plt
import numpy as np

def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a

fig = plt.figure()

ax = []

losses = {}

losses[0] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
losses[1] = [2.0, 2.1, 1.2, 1.3, 1.4, 2.5, 1.6, 1.7, 1.8, 1.9]

for worker_idx in range(2):
    ax.append(fig.add_subplot(2, 1, worker_idx + 1))

fig.subplots_adjust(hspace=0.25)

mean_loss = np.mean(losses[0][-10:])

ax[0].plot(range(len(losses[0])), losses[0], 'b')
ax[0].plot(range(len(losses[0])),
                exp_moving_average(losses[0], 10), 'r')
ax[0].legend(["Loss", "Loss_EMA"])

ax[1].plot(range(len(losses[1])), losses[1], 'b')
ax[1].plot(range(len(losses[1])),
                exp_moving_average(losses[1], 10), 'r')
ax[1].legend(["Loss", "Loss_EMA"])


fig.savefig("loss_worker.png")
