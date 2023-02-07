import pickle
from matplotlib import pyplot as plt

import numpy as np

POSSIBLE_METHODS = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'RHC']

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

    iterations = len(pickled_eu_list[0])

    if (plot_type == "box_whisker"):
        # Check if subset of methods is equal to all possible methods by simply comparing lengths
        method_type = "all_methods"
        if (len(subset_methods) != len(POSSIBLE_METHODS)):
            method_type = "subset_methods"
        FILENAME = 'box_whisker_charts/{}-{}-iterations{}.png'.format(plot_type, method_type, iterations)
        logged_eus = []

        # Remove 0s in arrays
        for i in range(0, len(pickled_eu_list)):
            pickled_eu_list[i] = [i for i in pickled_eu_list[i] if i != 0]

        for method in subset_methods:
            match method:  # noqa
                case 'PA (ß=10)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[0])))
                case 'PA (ß=5)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[1])))
                case 'PA (ß=4)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[2])))
                case 'PA (ß=3)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[3])))
                case 'PA (ß=2)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[4])))
                case 'PA (ß=1)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[5])))
                case 'PA (ß=.8)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[6])))
                case 'PA (ß=.6)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[7])))
                case 'PA (ß=.5)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[8])))
                case 'PA (ß=.1)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[9])))
                case 'PA (ß=0)':
                    logged_eus.append(np.log(np.array(pickled_eu_list[10])))
                case 'Uniform':
                    logged_eus.append(np.log(np.array(pickled_eu_list[11])))
                case 'RHC':
                    logged_eus.append(np.log(np.array(pickled_eu_list[12])))
                case _:
                    print("Invalid method")
                    exit()

        figure = plt.figure(figsize=(12, 6))

        plt.title("Expected Utility Variation on Solution Methods")
        plt.ylabel("Log(Expected Utility)")
        plt.xlabel("Solution Methods")

        plt.boxplot(logged_eus)
        x_axis = subset_methods

        plt.xticks([i + 1 for i in range(0, len(subset_methods))], x_axis)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 11
        plt.rcParams["grid.linestyle"] = "-"
        plt.grid(True)

        axis = plt.gca()
        axis.spines["top"].set_visible(False)

        plt.tight_layout()
        figure.savefig(FILENAME)
        plt.show()

    elif (plot_type == "bar"):
        total = [[0 for j in range(0, MAX_NUM_METHODS)] for i in range(0, len(node_indicies))]
        for node_index in range(0, len(node_indicies)):
            for method_index in range(0, MAX_NUM_METHODS):
                for iteration in range(0, iterations):
                    # for sublist in pickled_time_list[iteration]:
                    total[node_index][method_index] += pickled_time_list[node_index][method_index][iteration]
        # print("TOTAL: {}".format(total))
        average_times = [[None for j in range(0, MAX_NUM_METHODS)] for i in range(0, len(node_indicies))]
        # Get the average time across all instances
        for node_index in range(0, len(node_indicies)):
            for method_index in range(0, MAX_NUM_METHODS):
                average_times[node_index][method_index] = total[node_index][method_index] / iterations

        # Check if subset of methods is equal to all possible methods by simply comparing lengths
        method_type = "all_methods"
        if (len(subset_methods) != len(POSSIBLE_METHODS)):
            method_type = "subset_methods"

        # Plot results
        for node_id in bar_plot_nodes:
            FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, method_type, iterations, node_id)
            times = []
            for method in subset_methods:
                match method:  # noqa
                    case 'PA (ß=10)':
                        times.append(np.array(average_times[node_id][0]))
                    case 'PA (ß=5)':
                        times.append(np.array(average_times[node_id][1]))
                    case 'PA (ß=4)':
                        times.append(np.array(average_times[node_id][2]))
                    case 'PA (ß=3)':
                        times.append(np.array(average_times[node_id][3]))
                    case 'PA (ß=2)':
                        times.append(np.array(average_times[node_id][4]))
                    case 'PA (ß=1)':
                        times.append(np.array(average_times[node_id][5]))
                    case 'PA (ß=.8)':
                        times.append(np.array(average_times[node_id][6]))
                    case 'PA (ß=.6)':
                        times.append(np.array(average_times[node_id][7]))
                    case 'PA (ß=.5)':
                        times.append(np.array(average_times[node_id][8]))
                    case 'PA (ß=.1)':
                        times.append(np.array(average_times[node_id][9]))
                    case 'PA (ß=0)':
                        times.append(np.array(average_times[node_id][10]))
                    case 'Uniform':
                        times.append(np.array(average_times[node_id][11]))
                    case 'RHC':
                        times.append(np.array(average_times[node_id][12]))
                    case _:
                        print("Invalid method")
                        exit()

            figure = plt.figure(figsize=(12, 6))

            plt.title("Average Time Allocation on Node {}".format(node_id))
            plt.ylabel("Average Time Allocation")
            plt.xlabel("Solution Methods")

            plt.bar(x=subset_methods, height=times)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 11
            plt.rcParams["grid.linestyle"] = "-"
            plt.grid(True)

            axis = plt.gca()
            axis.spines["top"].set_visible(False)

            plt.tight_layout()
            figure.savefig(FILENAME)

    elif (plot_type == "scatter"):
        iterations = len(pickled_c_times)
        # Check if subset of methods is equal to all possible methods by simply comparing lengths
        method_type = "all_methods"
        if (len(subset_methods) != len(POSSIBLE_METHODS)):
            method_type = "subset_methods"
        FILENAME = 'scatter_charts/{}-{}-iterations{}.png'.format(plot_type, method_type, iterations)

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
                temp_allocations.append(pickled_c_times[ppv_index][transformed_node_id][method_index])
            node_reduced_pickled_c_times.append(temp_allocations)

        times = []
        for method in subset_methods:
            subtimes = []
            match method:
                case 'PA (ß=10)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][0])
                case 'PA (ß=5)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][1])
                case 'PA (ß=4)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][2])
                case 'PA (ß=3)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][3])
                case 'PA (ß=2)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][4])
                case 'PA (ß=1)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][5])
                case 'PA (ß=.8)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][6])
                case 'PA (ß=.6)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][7])
                case 'PA (ß=.5)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][8])
                case 'PA (ß=.1)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][9])
                case 'PA (ß=0)':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][10])
                case 'Uniform':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][11])
                case 'RHC':
                    for ppv_index in range(0, len(pickled_c_times)):
                        subtimes.append(node_reduced_pickled_c_times[ppv_index][12])
                case _:
                    print("Invalid method")
                    exit()
            times.append(subtimes)

        figure = plt.figure(figsize=(12, 6))

        plt.title("Time Allocation on C Value on Node {}".format(c_node_id))
        plt.ylabel("Time Allocation")
        plt.xlabel("C value")

        # Cycle through rainbow colors
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(times))))
        for index, method_str in enumerate(subset_methods):
            plt.scatter(x=c_list, y=[times[index]], c=next(colors), marker="o", label=method_str)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 11
        plt.rcParams["grid.linestyle"] = "-"
        plt.grid(True)
        plt.legend(loc='upper right')

        axis = plt.gca()
        axis.spines["top"].set_visible(False)

        plt.tight_layout()
        figure.savefig(FILENAME)
        plt.show()


if __name__ == "__main__":
    # Get all the node_ids that aren't fors or conditionals
    node_indicies = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]
    c_list = np.arange(.01, 5.11, .1)
    # c_list = np.arange(.1, 1.1, .2)
    c_node_id = 6

    # Pull all the data from the .txt files
    file_eus = open('data/eu_data_4.txt', 'rb')
    file_times = open('data/time_data_4.txt', 'rb')
    file_c_times = open('data/time_on_c_data_node6_TEST2.txt', 'rb')
    subset_methods = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'RHC']
    subset_methods = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'RHC']

    plot(plot_type="box_whisker", node_indicies=node_indicies, subset_methods=subset_methods, c_list=c_list, c_node_id=c_node_id,
         file_eus=file_eus, file_times=file_times, file_c_times=file_c_times, bar_plot_nodes=[1])
