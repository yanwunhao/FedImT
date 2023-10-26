import matplotlib.pyplot as plt

data_x_1 = [10, 20, 30, 40, 50]
data_x_2 = [20, 40, 60, 80]

fedavg = [
    [49.5, 84.2, 89.1, 92.9, 93.6],
    [44.3, 76.8, 86.2, 90.0, 91.5],
    [54.6, 75.9, 85.6, 88.7]
]

fedprox = [
    [53.1, 86.6, 90.9, 93.4, 95.4],
    [49.9, 81.4, 89.0, 91.9, 93.0],
    [59.6, 78.8, 89.0, 91.4]
]

fednova = [
    [54.5, 88.4, 90.1, 93.4, 95.6],
    [49.9, 81.9, 89.9, 91.7, 93.3],
    [59.9, 81.6, 90.7, 93.4]
]

fedhealth = [59.8, 89.0, 93.6, 97.0]

fedimt = [
    [46.7, 86.8, 91.4, 94.7, 96.6],
    [45.9, 88.6, 90.9, 94.4, 95.6],
    [57.6, 88.4, 94.6, 97.5]
]

titles = ["(a) MNIST", "(b) MNIST(n_latest)", "(c) UCI-HAR(n_latest)"]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

plt_index = 0
for ax in axs.flat:
    if plt_index != 2:
        ax.plot(data_x_1, fedavg[plt_index], marker="o", label="FedAvg", color="red")
        ax.plot(data_x_1, fedprox[plt_index], marker="v", label="FedProx", color="orange")
        ax.plot(data_x_1, fednova[plt_index], marker="^", label="FedNova", color="purple")
        ax.plot(data_x_1, fedimt[plt_index], marker="s", label="FedImT", color="blue")
    else:
        ax.plot(data_x_2, fedavg[plt_index], marker="o", label="FedAvg", color="red")
        ax.plot(data_x_2, fedprox[plt_index], marker="v", label="FedProx", color="orange")
        ax.plot(data_x_2, fednova[plt_index], marker="^", label="FedNova", color="purple")
        ax.plot(data_x_2, fedhealth, marker="p", label="FedHealth", color="olive")
        ax.plot(data_x_2, fedimt[plt_index], marker="s", label="FedImT", color="blue")
    ax.set_title(titles[plt_index])
    ax.set_xlabel("Global rounds")
    ax.set_ylabel("Acc. %")
    ax.legend(loc='best', ncols=1)
    plt_index += 1
plt.tight_layout(w_pad=3)
plt.show()

fig.savefig("comparison_1.eps", dpi=600, format="eps")