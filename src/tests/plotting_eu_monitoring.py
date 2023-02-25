import pickle
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

PLOT_TYPE = 'line-eu-monitoring'
FILENAME = 'src/tests/EU_monitoring/experiment'
BEST_EUS, NEW_EUS = pickle.load(open('src/tests/large-func/data/eu_monitoring_data.txt', 'rb'))
x_range = range(len(BEST_EUS))
print(len(BEST_EUS))


def plot():
    plt.figure(figsize=(12, 2))

    plt.plot(x_range, BEST_EUS, c="seagreen", marker="o", markersize=1, label="Best EU", zorder=1, linewidth=1.2)

    plt.ylabel(r"$\mathbb{E}[U_\mathcal{M}(\mathbf{t})]$")
    plt.xlabel("Attempted Swaps")

    axis = plt.gca()

    plt.setp(axis.get_xticklabels(), fontsize=12)
    plt.setp(axis.get_yticklabels(), fontsize=12)

    axis.set_ylim([0, 4.5])
    axis.set_xlim([-40, 3410])
    axis.set_xticks([])
    axis.set_yticks([])

    axis.xaxis.set_label_coords(0.93, 0.15)
    axis.yaxis.set_label_coords(0.021, 0.73)

    plt.tight_layout()
    plt.show()


def main():
    plot()


if __name__ == "__main__":
    main()
