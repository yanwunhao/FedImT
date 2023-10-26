import matplotlib.pyplot as plt

data_x_1 = [10, 20, 30, 40, 50]
data_x_2 = [20, 40, 60, 80]

fedavg = [
    [43.9, 70.9, 83.0, 86.4, 89.0],
    [38.5, 50.6, 63.4, 71.2, 78.6],
    [56.8, 57.4, 59.4, 59.8]
]

fedprox = [
    [44.3, 72.5, 83.5, 88.3, 89.4],
    [37.9, 49.2, 63.0, 70.9, 74.3],
    [57.9, 58.8, 59.8, 60.1]
]

fednova = [
    [46.4, 70.9, 82.6, 87.4, 88.1],
    [38.6, 50.6, 62.6, 73.2, 75.4],
    [61.3, 64.0, 66.0, 67.9]
]

fedhealth = [61.7, 64.1, 66.7, 68.5]

fedimt = [
    [42.7, 70.6, 83.4, 89.6, 92.5],
    [37.4, 51.4, 65.3, 77.5, 80.5],
    [61.3, 65.3, 68.9, 71.4]
]

titles = ["(a) MNIST (1:5 downsampled)", "(b) MNIST (1:20 downsampled)", "(c) The Ford Challenge"]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

plt_index = 0
for ax in axs.flat:
    if plt_index != 2:
        ax.plot(data_x_1, fedavg[plt_index], marker="o", label="FedAvg", color="red")
        ax.plot(data_x_1, fedprox[plt_index], marker="v", label="FedProx", color="orange")
        ax.plot(data_x_1, fednova[plt_index], marker="^", label="FedNova", color="purple")
        ax.plot(data_x_1, fedimt[plt_index], marker="s", label="FedImT", color="blue")
        ax.legend(loc='best', ncols=1)
    else:
        ax.plot(data_x_2, fedavg[plt_index], marker="o", label="FedAvg", color="red")
        ax.plot(data_x_2, fedprox[plt_index], marker="v", label="FedProx", color="orange")
        ax.plot(data_x_2, fednova[plt_index], marker="^", label="FedNova", color="purple")
        ax.plot(data_x_2, fedhealth, marker="p", label="FedHealth", color="olive")
        ax.plot(data_x_2, fedimt[plt_index], marker="s", label="FedImT", color="blue")
        ax.set_ylim(30, 72)
        ax.legend(loc='lower right', ncols=1)
    ax.set_title(titles[plt_index])
    ax.set_xlabel("Global rounds")
    ax.set_ylabel("Acc.M %")
    plt_index += 1
plt.tight_layout(w_pad=3)
plt.show()

fig.savefig("comparison_2.eps", dpi=600, format="eps")