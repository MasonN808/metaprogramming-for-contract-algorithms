import pickle
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['text.usetex'] = True

PLOT_TYPE = 'line-eu-monitoring'
FILENAME = 'src/tests/EU_monitoring/experiment'
BEST_EUS, NEW_EUS = pickle.load(open('src/tests/large-func/data/eu_monitoring_data.txt', 'rb'))
x_range = range(0, len(BEST_EUS))


def plot():
    # Make best EUs green
    plt.plot(x_range, BEST_EUS, c="seagreen",  marker="o", markersize=1, label="Best EU", zorder=1, linewidth=1.2)

    # Make new EUs blue
    plt.plot(x_range, NEW_EUS, c="steelblue", marker="o", markersize=1, label="New EU", zorder=2, linewidth=.5)

    plt.legend(loc='upper right', ncol=1, labelspacing=0.8, fontsize=13)

    plt.ylabel("Expected Utility")
    plt.xlabel("Epochs")

    axis = plt.gca()
    plt.setp(axis.get_xticklabels(), fontsize=14)
    plt.setp(axis.get_yticklabels(), fontsize=14)
    axis.set_ylim([0, 5])

    plt.tight_layout()
    plt.show()


def main():
    plot()


if __name__ == "__main__":
    main()