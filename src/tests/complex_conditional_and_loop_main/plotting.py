import pickle
from matplotlib import pyplot as plt

import numpy as np


if __name__ == "__main__":
    # Plot types:
    #   - "box_whisker" => a box and whisker plot of EU for our contract program on differing solution methods
    #   - "bar" => a bar graph of the average time allocation over N simulations for a particular node n_i on differing solution methods

    # Plot methods:
    #   - "all" => use all solution methods
    #   - "subset" => use PA(1), PA(.5), PA(0), Uniform, and EHC

    plot_type = "bar"
    # Nodes to plot (only for bar plot types):
    plot_nodes = [4, 8, 11]
    plot_methods = "subset"
    NUM_METHODS = 13

    # Get all the node_ids that aren't fors or conditionals
    node_indicies_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]

    # Pull all the data from the .txt files
    file_eus = open('data/eu_data.txt', 'rb')
    file_times = open('data/time_data.txt', 'rb')
    
    # Load the saved embedded lists to append new data
    pickled_eu_list = pickle.load(file_eus)
    pickled_time_list = pickle.load(file_times)

    iterations = len(pickled_eu_list[0]) # Not the greatest way to do calculate it
    print(iterations)
    # File to create plot
    

    if (plot_type == "box_whisker"):
        FILENAME = 'box_whisker_charts/{}-{}-iterations{}.png'.format(plot_type, plot_methods, iterations)
        if (plot_methods == "all"):
            # Plot results
            proportional1 = np.array(pickled_eu_list[0])
            proportional2 = np.array(pickled_eu_list[1])
            proportional3 = np.array(pickled_eu_list[2])
            proportional4 = np.array(pickled_eu_list[3])
            proportional5 = np.array(pickled_eu_list[4])
            proportional6 = np.array(pickled_eu_list[5])
            proportional7 = np.array(pickled_eu_list[6])
            proportional8 = np.array(pickled_eu_list[7])
            proportional9 = np.array(pickled_eu_list[8])
            proportional10 = np.array(pickled_eu_list[9])
            proportional11 = np.array(pickled_eu_list[10])

            uniform = np.array(pickled_eu_list[11])
            ehc = np.array(pickled_eu_list[12])

            figure = plt.figure(figsize=(12, 6))

            plt.title("Expected Utility Variation on Solution Methods")
            plt.ylabel("Expected Utility")
            plt.xlabel("Solution Methods")

            plt.boxplot([proportional1, proportional2, proportional3, proportional4, proportional5, proportional6, proportional7, proportional8, proportional9, proportional10, proportional11, uniform, ehc])
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ['PA (ß=1)', 'PA (ß=.9)', 'PA (ß=.8)', 'PA (ß=.7)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.4)', 'PA (ß=.3)', 'PA (ß=.2)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC'])

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 11
            plt.rcParams["grid.linestyle"] = "-"
            plt.grid(True)

            axis = plt.gca()
            axis.spines["top"].set_visible(False)

            plt.tight_layout()
            figure.savefig(FILENAME)
            plt.show()

        else:
            # Get only PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC'
            truncated_eu_list = [pickled_eu_list[0], pickled_eu_list[5], pickled_eu_list[10], pickled_eu_list[11], pickled_eu_list[12]]

            # Plot results
            proportional1 = np.array(truncated_eu_list[0])
            proportional2 = np.array(truncated_eu_list[1])
            proportional3 = np.array(truncated_eu_list[2])

            uniform = np.array(truncated_eu_list[3])
            ehc = np.array(truncated_eu_list[4])

            figure = plt.figure(figsize=(12, 6))

            plt.title("Expected Utility Variation on Solution Methods")
            plt.ylabel("Expected Utility")
            plt.xlabel("Solution Methods")

            plt.boxplot([proportional1, proportional2, proportional3, uniform, ehc])
            plt.xticks([1, 2, 3, 4, 5], ['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC'])

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
        if (plot_methods == "all"):
            # Plot results
            for node_index in range(0, len(plot_nodes)):
                FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, plot_methods, iterations, plot_nodes[node_index])
                # Plot results
                proportional1 = np.array(pickled_time_list[0])
                proportional2 = np.array(pickled_time_list[1])
                proportional3 = np.array(pickled_time_list[2])
                proportional4 = np.array(pickled_time_list[3])
                proportional5 = np.array(pickled_time_list[4])
                proportional6 = np.array(pickled_time_list[5])
                proportional7 = np.array(pickled_time_list[6])
                proportional8 = np.array(pickled_time_list[7])
                proportional9 = np.array(pickled_time_list[8])
                proportional10 = np.array(pickled_time_list[9])
                proportional11 = np.array(pickled_time_list[10])

                uniform = np.array(pickled_time_list[11])
                ehc = np.array(pickled_time_list[12])

                figure = plt.figure(figsize=(12, 6))

                plt.title("Expected Utility Variation on Solution Methods")
                plt.ylabel("Expected Utility")
                plt.xlabel("Solution Methods")

                plt.bar([proportional1, proportional2, proportional3, proportional4, proportional5, proportional6, proportional7, proportional8, proportional9, proportional10, proportional11, uniform, ehc])
                plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ['PA (ß=1)', 'PA (ß=.9)', 'PA (ß=.8)', 'PA (ß=.7)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.4)', 'PA (ß=.3)', 'PA (ß=.2)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC'])

                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["font.size"] = 11
                plt.rcParams["grid.linestyle"] = "-"
                plt.grid(True)

                axis = plt.gca()
                axis.spines["top"].set_visible(False)

                plt.tight_layout()
                figure.savefig(FILENAME)

        else:
            # Sum up all the allocations
            total = [[0 for j in range(0, len(node_indicies_list))] for i in range(0, NUM_METHODS)]
            for method_index in range(0, NUM_METHODS):
                for node_index in range(0, len(node_indicies_list)):
                    for iteration in range(0, iterations):
                        # for sublist in pickled_time_list[iteration]:
                        total[method_index][node_index] += pickled_time_list[method_index][node_index][iteration]

            # Get the average time across all instances
            for method_index in range(0, NUM_METHODS):
                for node_index in range(0, len(node_indicies_list)):
                    total[method_index][node_index] = total[method_index][node_index]/iterations
            average_times = total

            truncated_times_list = [average_times[0], average_times[5], average_times[10], average_times[11], average_times[12]]
            # Plot results
            for node_id in plot_nodes:
                FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, plot_methods, iterations, node_id)
                proportional1 = np.array(truncated_times_list[0][node_id])
                proportional2 = np.array(truncated_times_list[1][node_id])
                proportional3 = np.array(truncated_times_list[2][node_id])

                uniform = np.array(truncated_times_list[3][node_id])
                ehc = np.array(truncated_times_list[4][node_id])

                figure = plt.figure(figsize=(12, 6))

                plt.title("Average Time Allocation on Node {}".format(node_id))
                plt.ylabel("Average Time Allocation")
                plt.xlabel("Solution Methods")

                plt.bar(x=['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC'], height=[proportional1, proportional2, proportional3, uniform, ehc])
                # plt.xticks([1, 2, 3, 4, 5], ['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC'])

                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["font.size"] = 11
                plt.rcParams["grid.linestyle"] = "-"
                plt.grid(True)

                axis = plt.gca()
                axis.spines["top"].set_visible(False)

                plt.tight_layout()
                figure.savefig(FILENAME)
