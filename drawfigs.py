import numpy as np
import matplotlib.pyplot as plt

with open("./results/Tj_1.txt", "r") as f1:
    data1 = f1.read().split(" ")

with open("./results/Tj_2.txt", "r") as f2:
    data2 = f2.read().split(" ")

curve1 = []
curve2 = []

for data1_item in data1:
    if data1_item == "":
        continue
    else:
        curve1.append(float(data1_item))

for data2_item in data2:
    if data2_item == "":
        continue
    else:
        curve2.append(float(data2_item))

rounds = np.arange(0, 50, 1)
rounds = rounds + 1

fig, ax = plt.subplots()

ax.plot(rounds, curve1, marker="o", label="standard mode")
ax.plot(rounds, curve2, marker="o", label="n_latest mode")

ax.set_xlim(1, 50)
ax.set_xlabel('Global rounds')
ax.set_ylabel('Tj score')
ax.grid(True)

plt.legend()
plt.show()

fig.savefig("Tj_score.eps", dpi=600, format="eps")