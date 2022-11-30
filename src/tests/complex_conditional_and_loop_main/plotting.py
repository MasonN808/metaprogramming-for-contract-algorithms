import pickle
from matplotlib import pyplot as plt

import numpy as np


if __name__ == "__main__":
    # Plot types:
    #   - "box_whisker" => a box and whisker plot of EU for our contract program on differing solution methods
    #   - "bar" => a bar graph of the average time allocation over N simulations for a particular node n_i on differing solution methods
    #   - "satter" => a scatter plot of average EU

    # Plot methods:
    #   - "all" => use all solution methods
    #   - "subset" => use PA(1), PA(.5), PA(0), Uniform, and EHC

    # plot_type = "bar"
    # plot_type = "box_whisker"
    plot_type = "scatter"
    # Nodes to plot (only for bar plot types):
    # plot_nodes = [4, 8, 11]
    plot_nodes = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]
    plot_methods = "all"
    NUM_METHODS = 13

    # Get all the node_ids that aren't fors or conditionals
    node_indicies_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]

    # Pull all the data from the .txt files
    file_eus = open('data/eu_data_4.txt', 'rb')
    file_times = open('data/time_data_4.txt', 'rb')
    
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
            proportional1 = np.log(np.array(pickled_eu_list[0]))
            proportional2 = np.log(np.array(pickled_eu_list[1]))
            proportional3 = np.log(np.array(pickled_eu_list[2]))
            proportional4 = np.log(np.array(pickled_eu_list[3]))
            proportional5 = np.log(np.array(pickled_eu_list[4]))
            proportional6 = np.log(np.array(pickled_eu_list[5]))
            proportional7 = np.log(np.array(pickled_eu_list[6]))
            proportional8 = np.log(np.array(pickled_eu_list[7]))
            proportional9 = np.log(np.array(pickled_eu_list[8]))
            proportional10 = np.log(np.array(pickled_eu_list[9]))
            proportional11 = np.log(np.array(pickled_eu_list[10]))

            uniform = np.log(np.array(pickled_eu_list[11]))
            ehc = np.log(np.array(pickled_eu_list[12]))

            figure = plt.figure(figsize=(12, 6))

            plt.title("Expected Utility Variation on Solution Methods")
            plt.ylabel("Log(Expected Utility)")
            plt.xlabel("Solution Methods")

            plt.boxplot([proportional1, proportional2, proportional3, proportional4, proportional5, proportional6, proportional7, proportional8, proportional9, proportional10, proportional11, uniform, ehc])
            
            # x_axis = ['PA (ß=1)', 'PA (ß=.9)', 'PA (ß=.8)', 'PA (ß=.7)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.4)', 'PA (ß=.3)', 'PA (ß=.2)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC']
            x_axis = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC']
            
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], x_axis)

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
            proportional1 = np.log(np.array(truncated_eu_list[0]))
            proportional2 = np.log(np.array(truncated_eu_list[1]))
            proportional3 = np.log(np.array(truncated_eu_list[2]))

            uniform = np.log( np.array(truncated_eu_list[3]))
            ehc = np.log(np.array(truncated_eu_list[4]))

            figure = plt.figure(figsize=(12, 6))

            plt.title("Expected Utility Variation on Solution Methods")
            plt.ylabel("Log(Expected Utility)")
            plt.xlabel("Solution Methods")

            # x_axis = ['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC']
            x_axis = ['PA (ß=10)', 'PA (ß=1)', 'PA (ß=0)', 'Uniform', 'EHC']
           
            plt.boxplot([proportional1, proportional2, proportional3, uniform, ehc])
            plt.xticks([1, 2, 3, 4, 5], x_axis)

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

            total = [[0 for j in range(0, NUM_METHODS)] for i in range(0, len(node_indicies_list))]
            for node_index in range(0, len(node_indicies_list)):
                for method_index in range(0, NUM_METHODS):
                    for iteration in range(0, iterations):
                        # for sublist in pickled_time_list[iteration]:
                        total[node_index][method_index] += pickled_time_list[node_index][method_index][iteration]
            # print("TOTAL: {}".format(total))
            average_times = [[None for j in range(0, NUM_METHODS)] for i in range(0, len(node_indicies_list))]
            # Get the average time across all instances
            for node_index in range(0, len(node_indicies_list)):
                for method_index in range(0, NUM_METHODS):
                    average_times[node_index][method_index] = total[node_index][method_index]/iterations
            print("average")
            print(average_times[8])
            # print(average_times)
            # print("total")
            # print(total[0])

            # Plot results
            for node_id in plot_nodes:
                FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, plot_methods, iterations, node_id)

                proportional1 = np.array(average_times[node_id][0])
                proportional2 = np.array(average_times[node_id][1])
                proportional3 = np.array(average_times[node_id][2])
                proportional4 = np.array(average_times[node_id][3])
                proportional5 = np.array(average_times[node_id][4])
                proportional6 = np.array(average_times[node_id][5])
                proportional7 = np.array(average_times[node_id][6])
                proportional8 = np.array(average_times[node_id][7])
                proportional9 = np.array(average_times[node_id][8])
                proportional10 = np.array(average_times[node_id][9])
                proportional11 = np.array(average_times[node_id][10])

                uniform = np.array(average_times[node_id][11])
                ehc = np.array(average_times[node_id][12])

                figure = plt.figure(figsize=(12, 6))

                plt.title("Average Time Allocation on Node {}".format(node_id))
                plt.ylabel("Average Time Allocation")
                plt.xlabel("Solution Methods")
                # x_axis = ['PA (ß=1)', 'PA (ß=.9)', 'PA (ß=.8)', 'PA (ß=.7)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.4)', 'PA (ß=.3)', 'PA (ß=.2)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC']
                x_axis = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC']

                plt.bar(x=x_axis, height=[proportional1, proportional2, proportional3, proportional4, proportional5, proportional6, proportional7, proportional8, proportional9, proportional10, proportional11, uniform, ehc])
            
                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["font.size"] = 11
                plt.rcParams["grid.linestyle"] = "-"
                plt.grid(True)

                axis = plt.gca()
                axis.spines["top"].set_visible(False)

                plt.tight_layout()
                figure.savefig(FILENAME)

        else:
            #TODO: fix these index inconsistencies; it's different from box and whisker plot indexes since EU and TIME are different
            total = [[0 for j in range(0, NUM_METHODS)] for i in range(0, len(node_indicies_list))]
            for node_index in range(0, len(node_indicies_list)):
                for method_index in range(0, NUM_METHODS):
                    for iteration in range(0, iterations):
                        # for sublist in pickled_time_list[iteration]:
                        total[node_index][method_index] += pickled_time_list[node_index][method_index][iteration]

            average_times = [[None for j in range(0, NUM_METHODS)] for i in range(0, len(node_indicies_list))]
            # Get the average time across all instances
            for node_index in range(0, len(node_indicies_list)):
                for method_index in range(0, NUM_METHODS):
                    average_times[node_index][method_index] = total[node_index][method_index]/iterations

            # Truncates the nodes.  The index represents the method index; however, each element in average_times[i] are the nodes
            # Get only PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC'
            truncated_times_list = [average_times[0], average_times[5], average_times[10], average_times[11], average_times[12]]

            # Plot results
            for node_id in plot_nodes:
                # TODO: We need to traverse the DAG to identify when to reduce indices based on functional expression nodes that are not processes
                FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, plot_methods, iterations, node_id)
                if (node_id > 7):
                    node_id -= 1
                proportional1 = np.array(truncated_times_list[0][node_id])
                proportional2 = np.array(truncated_times_list[1][node_id])
                proportional3 = np.array(truncated_times_list[2][node_id])

                uniform = np.array(truncated_times_list[3][node_id])
                ehc = np.array(truncated_times_list[4][node_id])

                figure = plt.figure(figsize=(12, 6))

                if (node_id >= 7):
                    node_id += 1

                plt.title("Average Time Allocation on Node {}".format(node_id))
                plt.ylabel("Average Time Allocation")
                plt.xlabel("Solution Methods")

                # x_axis = ['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC']
                x_axis = ['PA (ß=10)', 'PA (ß=1)', 'PA (ß=0)', 'Uniform', 'EHC']
           
                plt.bar(x=x_axis, height=[proportional1, proportional2, proportional3, uniform, ehc])

                plt.rcParams["font.family"] = "Times New Roman"
                plt.rcParams["font.size"] = 11
                plt.rcParams["grid.linestyle"] = "-"
                plt.grid(True)

                axis = plt.gca()
                axis.spines["top"].set_visible(False)

                plt.tight_layout()

                figure.savefig(FILENAME)
    
    if (plot_type == "box_whisker"):
        FILENAME = 'box_whisker_charts/{}-{}-iterations{}.png'.format(plot_type, plot_methods, iterations)
        if (plot_methods == "all"):
            # Plot results
            proportional1 = np.log(np.array(pickled_eu_list[0]))
            proportional2 = np.log(np.array(pickled_eu_list[1]))
            proportional3 = np.log(np.array(pickled_eu_list[2]))
            proportional4 = np.log(np.array(pickled_eu_list[3]))
            proportional5 = np.log(np.array(pickled_eu_list[4]))
            proportional6 = np.log(np.array(pickled_eu_list[5]))
            proportional7 = np.log(np.array(pickled_eu_list[6]))
            proportional8 = np.log(np.array(pickled_eu_list[7]))
            proportional9 = np.log(np.array(pickled_eu_list[8]))
            proportional10 = np.log(np.array(pickled_eu_list[9]))
            proportional11 = np.log(np.array(pickled_eu_list[10]))

            uniform = np.log(np.array(pickled_eu_list[11]))
            ehc = np.log(np.array(pickled_eu_list[12]))

            figure = plt.figure(figsize=(12, 6))

            plt.title("Expected Utility Variation on Solution Methods")
            plt.ylabel("Log(Expected Utility)")
            plt.xlabel("Solution Methods")

            plt.boxplot([proportional1, proportional2, proportional3, proportional4, proportional5, proportional6, proportional7, proportional8, proportional9, proportional10, proportional11, uniform, ehc])
            
            # x_axis = ['PA (ß=1)', 'PA (ß=.9)', 'PA (ß=.8)', 'PA (ß=.7)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.4)', 'PA (ß=.3)', 'PA (ß=.2)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC']
            x_axis = ['PA (ß=10)', 'PA (ß=5)', 'PA (ß=4)', 'PA (ß=3)', 'PA (ß=2)', 'PA (ß=1)', 'PA (ß=.8)', 'PA (ß=.6)', 'PA (ß=.5)', 'PA (ß=.1)', 'PA (ß=0)', 'Uniform', 'EHC']
            
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], x_axis)

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
            proportional1 = np.log(np.array(truncated_eu_list[0]))
            proportional2 = np.log(np.array(truncated_eu_list[1]))
            proportional3 = np.log(np.array(truncated_eu_list[2]))

            uniform = np.log( np.array(truncated_eu_list[3]))
            ehc = np.log(np.array(truncated_eu_list[4]))

            figure = plt.figure(figsize=(12, 6))

            plt.title("Expected Utility Variation on Solution Methods")
            plt.ylabel("Log(Expected Utility)")
            plt.xlabel("Solution Methods")

            # x_axis = ['PA (ß=1)', 'PA (ß=.5)', 'PA (ß=0)', 'Uniform', 'EHC']
            x_axis = ['PA (ß=10)', 'PA (ß=1)', 'PA (ß=0)', 'Uniform', 'EHC']
           
            plt.boxplot([proportional1, proportional2, proportional3, uniform, ehc])
            plt.xticks([1, 2, 3, 4, 5], x_axis)

            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 11
            plt.rcParams["grid.linestyle"] = "-"
            plt.grid(True)

            axis = plt.gca()
            axis.spines["top"].set_visible(False)

            plt.tight_layout()
            figure.savefig(FILENAME)
            plt.show()

        