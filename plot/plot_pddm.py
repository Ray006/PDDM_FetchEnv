import numpy as np
import os
import matplotlib.pyplot as plt
# ds = []
# d = np.load('obs.npy', allow_pickle=True)
# ds.append(d)


## Plot
def plot_mean_std(data, label=None, name = 'reward'):


    # from ipdb import set_trace
    # set_trace()
    fig, ax = plt.subplots(1, figsize=(8, 8))

    for i in range(len(data)):

        mean_data = data[i][:, 0]
        std_data = data[i][:, 1]


        xvals = np.arange(len(mean_data))

        ax.plot(xvals, mean_data, label=label[i])
        ax.legend()

        # ax.fill_between(
        #     xvals,
        #     mean_data - std_data,
        #     mean_data + std_data,
        #     color=color,
        #     alpha=0.25)

    fig.savefig(name + '.png', dpi=200, bbox_inches='tight')


def get_rollouts(addr):

    data = []
    file_names = []

    # from ipdb import set_trace
    # set_trace()

    files = os.listdir(addr)
    file_names = files
    for file in files:
        if file.endswith('.npy'):
            file = os.path.join(addr, file)
            data.append(np.load(file, allow_pickle=True))

    plot_mean_std(data, file_names, addr)

    # return 0

if __name__ == '__main__':
    addr = 'pick_and_place'
    get_rollouts(addr)
