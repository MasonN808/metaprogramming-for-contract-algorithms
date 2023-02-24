import pickle
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['text.usetex'] = True

import numpy as np

POSSIBLE_METHODS = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=0.8)', 'PA (ß=0.6)', 'PA (ß=0.5)', 'PA (ß=0.1)', 'PA (ß=0)', r'\textsc{Equal}', r'\textsc{Rhc}']

MAX_NUM_METHODS = len(POSSIBLE_METHODS)

def plot(plot_type, node_indicies, subset_methods, file_eus, file_times, file_c_times, bar_plot_nodes=None, c_list=None, c_node_id=None):
    """
    Plots information pertinent to time allocation and expected utiltiy for various allocation methods

    :param plot_type: string, <box_whisker> => a box and whisker plot of EU for our contract program on differing solution methods
                    or <bar> => a bar graph of the average time allocation over N simulations for particular node(s) on differing solution methods
                    or <scatter> => a scatter plot of average time allocation for a particular node on various solution methods
    :param node_indicies: int[], the node indices of the contract program
    :param subset_methods: string[], the subset of methods that were used during simulation for plotting
    :param file_eus: File, file with EUs for various methods
    :param file_times: File, file with times for various methods on all nodes
    :param file_c_times: File, file with times for various methods on different c values in the performance profiles for a specified node
    :param bar_plot_nodes: int[], a list of node indicies that are specified to be plotted using a bar chart
    :param c_list: float[], a list of c values that were iterated on during simulation
    :param c_node_id: int, the node id specified for experimentation with various c values
    :return: a plot(s) saved to an external file
    """
    # Load the saved embedded lists to append new data
    pickled_eu_list = pickle.load(file_eus)
    pickled_time_list = pickle.load(file_times)
    pickled_c_times = pickle.load(file_c_times)

    simulations = len(pickled_eu_list[0])
    print("Simulations: {}".format(simulations))

    if (plot_type == "box_whisker"):
        # Check if subset of methods is equal to all possible methods by simply comparing lengths
        method_type = "all_methods"
        if (len(subset_methods) != len(POSSIBLE_METHODS)):
            method_type = "subset_methods"
        FILENAME = 'box_whisker_charts/{}-{}-iterations{}.png'.format(plot_type, method_type, simulations)
        logged_eus = []

        # vectorize
        for i in range(0, len(pickled_eu_list)):
            pickled_eu_list[i] = np.array(pickled_eu_list[i]) / 1000

        for method in subset_methods:
            match method:  # noqa
                case r'\textsc{Pa}($10$)':
                    logged_eus.append(pickled_eu_list[0])
                case r'\textsc{Equal}':
                    logged_eus.append(pickled_eu_list[11])
                case r'\textsc{Pa}($5.0$)':
                    logged_eus.append(pickled_eu_list[1])
                case r'\textsc{Pa}($4.0$)':
                    logged_eus.append(pickled_eu_list[2])
                case r'\textsc{Pa}($3.0$)':
                    logged_eus.append(pickled_eu_list[3])
                case r'\textsc{Pa}($2.0$)':
                    logged_eus.append(pickled_eu_list[4])
                case r'\textsc{Pa}($1.0$)':
                    logged_eus.append(pickled_eu_list[5])
                case r'\textsc{Pa}($0.8$)':
                    logged_eus.append(pickled_eu_list[6])
                case r'\textsc{Pa}($0.6$)':
                    logged_eus.append(pickled_eu_list[7])
                case r'\textsc{Pa}($0.5$)':
                    logged_eus.append(pickled_eu_list[8])
                case r'\textsc{Pa}($0.1$)':
                    print('Best Baseline Technique')
                    print('Max', max(pickled_eu_list[9]))
                    print('Min', min(pickled_eu_list[9]))
                    print('Range', max(pickled_eu_list[9]) - min(pickled_eu_list[9]))
                    print('Mean', sum(pickled_eu_list[9]) / len(pickled_eu_list[9]))
                    print('Lower Quartile', np.percentile(pickled_eu_list[9], 25))
                    print('Upper Quartile', np.percentile(pickled_eu_list[9], 75))
                    logged_eus.append(pickled_eu_list[9])
                case r'\textsc{Pa}($0.0$)':
                    logged_eus.append(pickled_eu_list[10])
                case r'\textsc{Rhc}':
                    print('Rhc')
                    print('Max', max(pickled_eu_list[12]))
                    print('Min', min(pickled_eu_list[12]))
                    print('Range', max(pickled_eu_list[12]) - min(pickled_eu_list[12]))
                    print('Mean', sum(pickled_eu_list[12]) / len(pickled_eu_list[12]))
                    print('Lower Quartile', np.percentile(pickled_eu_list[12], 25))
                    print('Upper Quartile', np.percentile(pickled_eu_list[12], 75))
                    logged_eus.append(pickled_eu_list[12])
                case _:
                    print("Invalid method")
                    exit()

        figure = plt.figure(figsize=(10, 5))

        boxplot = plt.boxplot(logged_eus, patch_artist=True, vert=True, zorder=100, widths=0.8, sym="D", boxprops=dict(facecolor="red"), medianprops=dict(linewidth=1.2, color='black'))
        x_axis = subset_methods

        colors = iter(plt.cm.coolwarm(np.linspace(1, 0, 10)))
        color = ['slategray'] + [next(colors) for _ in range(10)] + ['seagreen']

        i = 0
        for patch in boxplot['boxes']:
            patch.set(facecolor=color[i]) 
            i += 1

        plt.xticks([i + 1 for i in range(0, len(subset_methods))], x_axis)
        plt.ylabel("Expected Utility")
        plt.xlabel("Time Allocation Technique")

        axis = plt.gca()
        plt.setp(axis.get_xticklabels(), fontsize=14)
        plt.setp(axis.get_yticklabels(), fontsize=14)

        plt.tight_layout()
        figure.savefig(FILENAME)
        plt.show()

    elif (plot_type == "bar"):
        methods_length = len(pickled_time_list[0])
        total = [[0 for j in range(0, methods_length)] for i in range(0, len(node_indicies))]
        for node_index in range(0, len(node_indicies)):
            for method_index in range(0, methods_length):
                for iteration in range(0, simulations):
                    # TODO: This could be simplified
                    total[node_index][method_index] += pickled_time_list[node_index][method_index][iteration]
        average_times = [[None for j in range(0, methods_length)] for i in range(0, len(node_indicies))]

        # Get the average time across all simulations
        for node_index in range(0, len(node_indicies)):
            for method_index in range(0, methods_length):
                average_times[node_index][method_index] = total[node_index][method_index] / simulations

        # Plot results
        for node_id in bar_plot_nodes:
            FILENAME = 'bar_charts/{}-iterations{}-node{}.png'.format(plot_type, simulations, node_id)
            times = []
            for method in subset_methods:
                match method:  # noqa
                    case r"\textsc{Pa}($10$)":
                        times.append(np.array(average_times[node_id][0]))
                    case r'\textsc{Pa}($5.0$)':
                        times.append(np.array(average_times[node_id][1]))
                    case r'\textsc{Pa}($4.0$)':
                        times.append(np.array(average_times[node_id][2]))
                    case r'\textsc{Pa}($3.0$)':
                        times.append(np.array(average_times[node_id][3]))
                    case r'\textsc{Pa}($2.0$)':
                        times.append(np.array(average_times[node_id][4]))
                    case r'\textsc{Pa}($1.0$)':
                        times.append(np.array(average_times[node_id][5]))
                    case r'\textsc{Pa}($0.8$)':
                        times.append(np.array(average_times[node_id][6]))
                    case r'\textsc{Pa}($0.6$)':
                        times.append(np.array(average_times[node_id][7]))
                    case r'\textsc{Pa}($0.5$)':
                        times.append(np.array(average_times[node_id][8]))
                    case r'\textsc{Pa}($0.1$)':
                        times.append(np.array(average_times[node_id][9]))
                    case r'\textsc{Pa}($0.0$)':
                        times.append(np.array(average_times[node_id][10]))
                    case r'\textsc{Equal}':
                        times.append(np.array(average_times[node_id][11]))
                    case r'\textsc{Rhc}':
                        times.append(np.array(average_times[node_id][12]))
                    case _:
                        print("Invalid method")
                        exit()

            figure = plt.figure(figsize=(12, 6))

            plt.title("Average Time Allocation on Node {}".format(node_id))
            plt.ylabel("Average Time Allocation")
            plt.xlabel("Solution Methods")

            plt.bar(x=subset_methods, height=times)

            axis = plt.gca()
            axis.spines["top"].set_visible(False)

            plt.tight_layout()
            figure.savefig(FILENAME)
            plt.show()

    elif (plot_type == "scatter"):
        simulations = len(pickled_c_times)
        FILENAME = 'scatter_charts/{}-iterations{}.png'.format(plot_type, simulations)

        # Reduce the node id if for and conditionals exist before it
        transformed_node_id = c_node_id
        if transformed_node_id > 7:
            transformed_node_id -= 1
        if transformed_node_id > 11:
            transformed_node_id -= 1

        # Reduce pickled_c_times to having the specified methods and nodes
        print("LENGTH: {}".format(len(pickled_c_times[0])))  # TODO: FIX THIS
        for i in range(0, 13):
            print(pickled_c_times[1][i][2])

        # Truncate with respect to a specified node to choose from
        # pickled_c_times arranged as pickled_c_times[ppv_index][nodes][methods]
        node_reduced_pickled_c_times = []
        for ppv_index in range(0, len(pickled_c_times)):
            # Reverse the pickled_c_times over each node index such that the start is at the end and vice versa
            # Issue in the appending of values in the main file FIXME
            pickled_c_times[ppv_index].reverse()
            temp_allocations = []
            for method_index in range(0, len(pickled_c_times[0][0])):
                # Get the allocations for the specified transformed node_id
                if method_index != 12:
                    temp_allocations.append(pickled_c_times[ppv_index][transformed_node_id][method_index])
                else:
                    temp_allocations.append(pickled_c_times[ppv_index][5][method_index])
            node_reduced_pickled_c_times.append(temp_allocations)

        times = []
        for method in subset_methods:
            subtimes = []
            match method:
                case r"\textsc{Pa}($10$)":
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][0])
                case r'\textsc{Pa}($5.0$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][1])
                case r'\textsc{Pa}($4.0$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][2])
                case r'\textsc{Pa}($3.0$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][3])
                case r'\textsc{Pa}($2.0$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][4])
                case r'\textsc{Pa}($1.0$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][5])
                case r'\textsc{Pa}($0.8$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][6])
                case r'\textsc{Pa}($0.6$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][7])
                case r'\textsc{Pa}($0.5$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][8])
                case r'\textsc{Pa}($0.1$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][9])
                case r'\textsc{Pa}($0.0$)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][10])
                case r'\textsc{Equal}':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][11])
                case r'\textsc{Rhc}':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][12])
                case _:
                    print("Invalid method")
                    exit()
            times.append(subtimes)

        figure = plt.figure(figsize=(10, 5))

        # Make Uniform gray
        plt.plot(c_list, times[0], c="slategray", label=subset_methods[0])

        # Make the PA methods a single heatmap color
        colors = iter(plt.cm.coolwarm(np.linspace(1, 0, 10)))
        for index, method_str in list(enumerate(subset_methods))[1:11]:
            print(index)
            plt.plot(c_list, times[index], c=next(colors), linestyle='dashed', label=method_str)

        # Make RHC green
        plt.plot(c_list, times[11], c="seagreen", marker="o", label=subset_methods[11], zorder=100)

        plt.legend(loc='upper right', ncol=3, labelspacing=0.8, fontsize=13)

        plt.ylabel("Average Time Allocation [sec]")
        plt.xlabel("Growth Rate")

        axis = plt.gca()
        plt.setp(axis.get_xticklabels(), fontsize=14)
        plt.setp(axis.get_yticklabels(), fontsize=14)
        axis.set_ylim([0, 1.65])

        axis.xaxis.set_ticks(np.arange(0.0, 5.1, 0.5))

        plt.tight_layout()
        figure.savefig(FILENAME)
        plt.show()

def print_eu_data(file_eus, subset_methods):
    # Load the saved embedded lists to append new data
    pickled_eu_list = pickle.load(file_eus)

    iterations = len(pickled_eu_list[0])
    print("SIMULATIONS: {}".format(iterations))

    # vectorize
    for i in range(0, len(pickled_eu_list)):
        pickled_eu_list[i] = np.array(pickled_eu_list[i])

    for method in subset_methods:
        eu = 0
        eu_std = 0
        match method:  # noqa
            case 'PA (ß=1)':
                eu = np.mean(pickled_eu_list[0])
                eu_std = np.std(pickled_eu_list[0])
            case 'PA (ß=.5)':
                eu = np.mean(pickled_eu_list[1])
                eu_std = np.std(pickled_eu_list[1])
            case 'PA (ß=.1)':
                eu = np.mean(pickled_eu_list[2])
                eu_std = np.std(pickled_eu_list[2])
            case 'PA (ß=0)':
                eu = np.mean(pickled_eu_list[3])
                eu_std = np.std(pickled_eu_list[3])
            case r'\textsc{Equal}':
                eu = np.mean(pickled_eu_list[4])
                eu_std = np.std(pickled_eu_list[4])
            case r'\textsc{Rhc}':
                eu = np.mean(pickled_eu_list[5])
                eu_std = np.std(pickled_eu_list[5])
            case _:
                print("Invalid method")
                exit()
        print("{} EU --> mean: {} --> std: {}".format(method, eu, eu_std))


if __name__ == "__main__":
    # Get all the node_ids that aren't fors or conditionals
    node_indicies = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]
    c_list = np.arange(.01, 5.11, .1)
    c_node_id = 8
    # c_node_id = 5

    # Pull all the data from the .txt files
    file_eus = open('src/tests/robotics_domain/data/eu_data.txt', 'rb')
    file_times = open('src/tests/robotics_domain/data/time_data.txt', 'rb')
    file_c_times = open('src/tests/robotics_domain/data/time_on_c_data_node8.txt', 'rb')
    subset_methods = [r'\textsc{Equal}', r'\textsc{Pa}($5.0$)', r'\textsc{Pa}($4.0$)', r'\textsc{Pa}($3.0$)', r'\textsc{Pa}($2.0$)', r'\textsc{Pa}($1.0$)', r'\textsc{Pa}($0.8$)', r'\textsc{Pa}($0.6$)', r'\textsc{Pa}($0.5$)', r'\textsc{Pa}($0.1$)', r'\textsc{Pa}($0.0$)', r'\textsc{Rhc}']
    # subset_methods = ['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', r'\textsc{Equal}', r'\textsc{Rhc}']
    # print_eu_data(file_eus=file_eus, subset_methods=subset_methods)

    plot(plot_type="box_whisker", node_indicies=node_indicies, subset_methods=subset_methods, c_list=c_list, c_node_id=c_node_id,
         file_eus=file_eus, file_times=file_times, file_c_times=file_c_times, bar_plot_nodes=[1])
