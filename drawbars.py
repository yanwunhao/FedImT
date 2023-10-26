import matplotlib.pyplot as plt
import numpy as np

tasks = ("MNIST(1:5)", "MNIST(1:20)", "The Ford Challenge")
results = {
    'CELoss': (88.7, 73.6, 64.5),
    'FocalLoss': (93.1, 74.6, 70.2),
    'GHMCLoss': (92.6, 68.9, 68.9),
    'BCELoss(FedImT)': (92.5, 80.5, 71.4)
}

x = np.arange(len(tasks))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in results.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('Acc.M %')
ax.set_xticks(x + width, tasks)
ax.legend(loc='upper right', ncols=1)
ax.set_ylim(0, 100)

plt.show()

fig.savefig("loss_comparison.eps", dpi=600, format="eps")