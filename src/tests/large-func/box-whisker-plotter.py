import pickle
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['text.usetex'] = True

PLOT_TYPE = 'box-whisker'
BUDGET = 10

EU_FILE = 'src/tests/large-func/data/eu_data1.txt'
EUs = pickle.load(open(EU_FILE, 'rb'))
SIMULATIONS = len(EUs)

SAVE_FILENAME = 'src/tests/large-func/plots/{}-simulations{}.png'.format(PLOT_TYPE, SIMULATIONS)

METHODS = [r'\textsc{Equal}', r'\textsc{Pa}($5.0$)', r'\textsc{Pa}($4.0$)', r'\textsc{Pa}($3.0$)', r'\textsc{Pa}($2.0$)', r'\textsc{Pa}($1.0$)', r'\textsc{Pa}($0.8$)', r'\textsc{Pa}($0.6$)', r'\textsc{Pa}($0.5$)', r'\textsc{Pa}($0.1$)', r'\textsc{Pa}($0.0$)', r'\textsc{Rhc}']
# MAKE IT SO THAT THE TODO PARENTS HAVE A HUGE DIFFEREENCE ON CHILDREN


def plot():
    # vectorize and scale
    for i in range(0, len(EUs)):
        # EUs[i] = np.log(np.array(EUs[i])*100)
        EUs[i] = (np.array(EUs[i]) * 1000)

    ordered_EUs = []

    for method in METHODS:
        match method:  # noqa
            case r'\textsc{Pa}($10$)':
                ordered_EUs.append(EUs[0])
            case r'\textsc{Equal}':
                ordered_EUs.append(EUs[11])
            case r'\textsc{Pa}($5.0$)':
                ordered_EUs.append(EUs[1])
            case r'\textsc{Pa}($4.0$)':
                ordered_EUs.append(EUs[2])
            case r'\textsc{Pa}($3.0$)':
                ordered_EUs.append(EUs[3])
            case r'\textsc{Pa}($2.0$)':
                ordered_EUs.append(EUs[4])
            case r'\textsc{Pa}($1.0$)':
                ordered_EUs.append(EUs[5])
            case r'\textsc{Pa}($0.8$)':
                ordered_EUs.append(EUs[6])
            case r'\textsc{Pa}($0.6$)':
                ordered_EUs.append(EUs[7])
            case r'\textsc{Pa}($0.5$)':
                ordered_EUs.append(EUs[8])
            case r'\textsc{Pa}($0.1$)':
                print('Best Baseline Technique')
                print('Max', max(EUs[9]))
                print('Min', min(EUs[9]))
                print('Range', max(EUs[9]) - min(EUs[9]))
                print('Mean', sum(EUs[9]) / len(EUs[9]))
                print('Lower Quartile', np.percentile(EUs[9], 25))
                print('Upper Quartile', np.percentile(EUs[9], 75))
                ordered_EUs.append(EUs[9])
            case r'\textsc{Pa}($0.0$)':
                ordered_EUs.append(EUs[10])
            case r'\textsc{Rhc}':
                print('Rhc')
                print('Max', max(EUs[12]))
                print('Min', min(EUs[12]))
                print('Range', max(EUs[12]) - min(EUs[12]))
                print('Mean', sum(EUs[12]) / len(EUs[12]))
                print('Lower Quartile', np.percentile(EUs[12], 25))
                print('Upper Quartile', np.percentile(EUs[12], 75))
                ordered_EUs.append(EUs[12])
            case _:
                print("Invalid method")
                exit()

    figure = plt.figure(figsize=(10, 5))

    boxplot = plt.boxplot(ordered_EUs, patch_artist=True, vert=True, zorder=100, widths=0.8, sym="D", boxprops=dict(facecolor="red"), medianprops=dict(linewidth=1.2, color='black'))
    x_axis = METHODS

    colors = iter(plt.cm.coolwarm(np.linspace(1, 0, 10)))
    color = ['slategray'] + [next(colors) for _ in range(10)] + ['seagreen']

    i = 0
    for patch in boxplot['boxes']:
        patch.set(facecolor=color[i])
        i += 1

    plt.xticks([i + 1 for i in range(0, len(METHODS))], x_axis)
    plt.ylabel("Expected Utility")
    plt.xlabel("Time Allocation Technique")

    axis = plt.gca()
    plt.setp(axis.get_xticklabels(), fontsize=14)
    plt.setp(axis.get_yticklabels(), fontsize=14)

    plt.tight_layout()
    figure.savefig(SAVE_FILENAME)
    plt.show()


def main():
    plot()


if __name__ == "__main__":
    main()
