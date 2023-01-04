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
    :param node_indicies: the node indices to be plotted
    :param methods: string, <all> => use all solution methods, <subset> => use PA(1), PA(.5), PA(0), Uniform, and RHC
    :param threshold: float, the threshold of the temperature decay during annealing
    :param decay: float, the decay rate of the temperature during annealing
    :return: A stream of optimized time allocations associated with each contract algorithm
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
        FILENAME = 'box_whisker_charts/{}-iterations{}.png'.format(plot_type, method_type, iterations)
        logged_eus = []
        for method in subset_methods:
            match method:
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
                case default:
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
                match method:
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
                    case default:
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
        FILENAME = 'scatter_charts/{}-{}-iterations{}.png'.format(plot_type, methods, iterations)
        if (methods == "all"):
            # Reduce the node id if for and conditionals exist before it
            transformed_node_id = c_node_id
            if transformed_node_id > 7:
                transformed_node_id -= 1
            if transformed_node_id > 12:
                transformed_node_id -= 1

            # Reduce pickled_c_times to having the specified methods and nodes
            node_reduced_pickled_c_times = []

            # Truncate with respect to a specified node to choose from
            # pickled_c_times arranged as pickled_c_times[ppv_index][nodes][methods]
            print(pickled_c_times)
            for ppv_index in range(0, len(pickled_c_times)):
                temp_allocations = []
                for method_index in range(0, len(pickled_c_times[0][0])):
                    # Get the allocations for the specified transformed node_id
                    temp_allocations.append(pickled_c_times[ppv_index][transformed_node_id][method_index])
                node_reduced_pickled_c_times.append(temp_allocations)

            # Get all methods
            truncated_method_indicies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            method_reduced_pickled_c_times = []

            # Get the allocations for the specified methods and create sublist of allocations for each method
            for method_index in truncated_method_indicies:
                temp_allocations = []
                for ppv_index in range(0, len(pickled_c_times)):
                    temp_allocations.append(node_reduced_pickled_c_times[ppv_index][method_index])
                method_reduced_pickled_c_times.append(temp_allocations)

            proportional1 = np.array(method_reduced_pickled_c_times[0])
            proportional2 = np.array(method_reduced_pickled_c_times[1])
            proportional3 = np.array(method_reduced_pickled_c_times[2])
            proportional4 = np.array(method_reduced_pickled_c_times[3])
            proportional5 = np.array(method_reduced_pickled_c_times[4])
            proportional6 = np.array(method_reduced_pickled_c_times[5])
            proportional7 = np.array(method_reduced_pickled_c_times[6])
            proportional8 = np.array(method_reduced_pickled_c_times[7])
            proportional9 = np.array(method_reduced_pickled_c_times[8])
            proportional10 = np.array(method_reduced_pickled_c_times[9])
            proportional11 = np.array(method_reduced_pickled_c_times[10])

            uniform = np.array(method_reduced_pickled_c_times[11])
            RHC = np.array(method_reduced_pickled_c_times[12])

            figure = plt.figure(figsize=(12, 6))

            plt.title("Time Allocation on C Value on Node {}".format(c_node_id))
            plt.ylabel("Time Allocation")
            plt.xlabel("C value")

            # 'PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'RHC'

            # Cycle through rainbow colors
            colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(method_reduced_pickled_c_times))))

            plt.scatter(x=c_list, y=[proportional1], c=next(colors), marker="o", label='PA (ß=10)')
            plt.scatter(x=c_list, y=[proportional2], c=next(colors), marker="o", label='PA (ß=5)')
            plt.scatter(x=c_list, y=[proportional3], c=next(colors), marker="o", label='PA (ß=4)')
            plt.scatter(x=c_list, y=[proportional4], c=next(colors), marker="o", label='PA (ß=5)')
            plt.scatter(x=c_list, y=[proportional5], c=next(colors), marker="o", label='PA (ß=6)')
            plt.scatter(x=c_list, y=[proportional6], c=next(colors), marker="o", label='PA (ß=7)')
            plt.scatter(x=c_list, y=[proportional7], c=next(colors), marker="o", label='PA (ß=.8)')
            plt.scatter(x=c_list, y=[proportional8], c=next(colors), marker="o", label='PA (ß=.6)')
            plt.scatter(x=c_list, y=[proportional9], c=next(colors), marker="o", label='PA (ß=.5)')
            plt.scatter(x=c_list, y=[proportional10], c=next(colors), marker="o", label='PA (ß=.1)')
            plt.scatter(x=c_list, y=[proportional11], c=next(colors), marker="o", label='PA (ß=0)')
            plt.scatter(x=c_list, y=[uniform], c=next(colors), marker="o", label='uniform')
            plt.scatter(x=c_list, y=[RHC], c=next(colors), marker="o", label='RHC')

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

        else:
            # Reduce the node id if for and conditionals exist before it
            transformed_node_id = c_node_id
            if transformed_node_id > 7:
                transformed_node_id -= 1
            if transformed_node_id > 12:
                transformed_node_id -= 1

            # Reduce pickled_c_times to having the specified methods and nodes
            node_reduced_pickled_c_times = []
            # Initilize what methods to choose from
            # pickled_c_times arranged as pickled_c_times[ppv_index][nodes][methods]
            for ppv_index in range(0, len(pickled_c_times)):
                temp_allocations = []
                for method_index in range(0, len(pickled_c_times[0][0])):
                    # Get the allocations for the specified transformed node_id
                    temp_allocations.append(pickled_c_times[ppv_index][transformed_node_id][method_index])
                node_reduced_pickled_c_times.append(temp_allocations)

            # Get only PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'RHC'
            truncated_method_indicies = [5, 8, 9, 11, 12]
            # truncated_method_indicies = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11]
            method_reduced_pickled_c_times = []
            # Get the allocations for the specified methods and create sublist of allocations for each method
            for method_index in truncated_method_indicies:
                temp_allocations = []
                for ppv_index in range(0, len(pickled_c_times)):
                    temp_allocations.append(node_reduced_pickled_c_times[ppv_index][method_index])
                method_reduced_pickled_c_times.append(temp_allocations)

            # Plot results
            proportional1 = np.array(method_reduced_pickled_c_times[0])
            proportional2 = np.array(method_reduced_pickled_c_times[1])
            proportional3 = np.array(method_reduced_pickled_c_times[2])

            uniform = np.array(method_reduced_pickled_c_times[3])
            # RHC = np.array(method_reduced_pickled_c_times[4])

            figure = plt.figure(figsize=(12, 6))

            plt.title("Time Allocation on C Value on Node {}".format(c_node_id))
            plt.ylabel("Time Allocation")
            plt.xlabel("C value")

            # 'PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'RHC'
            plt.scatter(x=c_list, y=[proportional1], c='r', marker="o", label='PA (ß=1)')
            plt.scatter(x=c_list, y=[proportional2], c='g', marker="o", label='PA (ß=.5)')
            plt.scatter(x=c_list, y=[proportional3], c='b', marker="o", label='PA (ß=0)')
            plt.scatter(x=c_list, y=[uniform], c='y', marker="o", label='first')
            # plt.scatter(x=c_list, y=[RHC])
            # plt.xticks([1, 2, 3, 4, 5], x_axis)

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
    # Plot types:
    #   - "box_whisker" => a box and whisker plot of EU for our contract program on differing solution methods
    #   - "bar" => a bar graph of the average time allocation over N simulations for a particular node n_i on differing solution methods
    #   - "scatter" => a scatter plot of average EU

    # Plot methods:
    #   - "all" => use all solution methods
    #   - "subset" => use PA(1), PA(.5), PA(0), Uniform, and RHC

    # Get all the node_ids that aren't fors or conditionals
    node_indicies = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]
    c_list = np.arange(.1, 5.1, .2)
    c_node_id = 6

    # Pull all the data from the .txt files
    file_eus = open('data/eu_data_4.txt', 'rb')
    file_times = open('data/time_data_4.txt', 'rb')
    file_c_times = open('data/time_on_c_data.txt', 'rb')
    subset_methods = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'RHC']
    plot(plot_type="bar", node_indicies=node_indicies, subset_methods=subset_methods, c_list=c_list,
         file_eus=file_eus, file_times=file_times, file_c_times=file_c_times, bar_plot_nodes=[1])
    # plot(plot_type="scatter", node_indicies=node_indicies, subset_methods=methods, c_list=c_list,
    #      file_eus=file_eus, file_times=file_times, file_c_times=file_c_times, bar_plot_nodes=node_indicies)
