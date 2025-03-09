import matplotlib.pyplot as plt
import numpy as np

# set default font size
plt.rcParams.update({"font.size": 16})

# data
# label, filename
data = [
    ("TFS", "figures/topdown/fromscratch/distances.txt"),
    ("Only PT", "figures/topdown/pretrainedfixed/distances.txt"),
    ("PT + OL", "figures/topdown/pretrainedlearning/distances.txt"),
    ("Using GT", "figures/topdown/realsense/distances.txt"),
]

# read data
for i in range(len(data)):
    with open(data[i][1], "r") as f:
        data[i] = (data[i][0], [float(d) for d in f.readlines()])

# plot boxplots
fig, ax = plt.subplots()
ax.set_title("Distance flown between interventions")
ax.set_ylabel("Distance [m]")
# ax.set_xlabel("Method")
ax.set_xticklabels([d[0] for d in data])
ax.set_yticks([2, 4, 6, 8, 10, 12, 14, 16])
# ax.grid(axis="y")
# ax.set_axisbelow(True)
ax.boxplot([d[1] for d in data])
# add median as text
for i in range(len(data)):
    ax.text(
        i + 1,
        np.median(data[i][1]),
        f"{np.median(data[i][1]):.2f}",
        # sum(data[i][1]) / len(data[i][1]),
        # f"{sum(data[i][1]) / len(data[i][1]):.2f}",
        ha="center",
        va="bottom",
        color="C1",
    )
fig.tight_layout()
plt.savefig("figures/topdown/boxplot.pdf", bbox_inches="tight", transparent=True)
