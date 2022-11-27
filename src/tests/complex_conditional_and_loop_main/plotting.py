if __name__ == "__main__":
    # Plot types:
    #   - "box_whisker" => a box and whisker plot of EU for our contract program on differing solution methods
    #   - "bar" => a bar graph of the average time allocation over N simulations for a particular node n_i on differing solution methods

    # Plot methods:
    #   - "all" => use all solution methods
    #   - "subset" => use PA(1), PA(.5), PA(0), Uniform, and EHC

    plot_type = "box_whisker"
    # Nodes to plot (only for bar plot types):
    plot_nodes = [4, 8, 11]
    plot_methods = "subset"

    # File to put plot
    FILENAME = '{}-{}-iterations{}.png'.format(plot_type, plot_methods, ITERATIONS)

    if (plot_type == "box_whisker"):
        if (plot_methods == "all"):
            # Plot results
            proportional1 = np.array(eu_list[0])
            proportional2 = np.array(eu_list[1])
            proportional3 = np.array(eu_list[2])
            proportional4 = np.array(eu_list[3])
            proportional5 = np.array(eu_list[4])
            proportional6 = np.array(eu_list[5])
            proportional7 = np.array(eu_list[6])
            proportional8 = np.array(eu_list[7])
            proportional9 = np.array(eu_list[8])
            proportional10 = np.array(eu_list[9])
            proportional11 = np.array(eu_list[10])

            uniform = np.array(eu_list[11])
            ehc = np.array(eu_list[12])

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
            # Plot results
            proportional1 = np.array(eu_list[0])
            proportional2 = np.array(eu_list[1])
            proportional3 = np.array(eu_list[2])

            uniform = np.array(eu_list[3])
            ehc = np.array(eu_list[4])

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
                FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, plot_methods, ITERATIONS, plot_nodes[node_index])
                # Plot results
                proportional1 = np.array(time_list[0])
                proportional2 = np.array(time_list[1])
                proportional3 = np.array(time_list[2])
                proportional4 = np.array(time_list[3])
                proportional5 = np.array(time_list[4])
                proportional6 = np.array(time_list[5])
                proportional7 = np.array(time_list[6])
                proportional8 = np.array(time_list[7])
                proportional9 = np.array(time_list[8])
                proportional10 = np.array(time_list[9])
                proportional11 = np.array(time_list[10])

                uniform = np.array(time_list[11])
                ehc = np.array(time_list[12])

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
            # Get the average time across all instances
            total = [0 for i in range(0, NUM_METHODS)]
            for method_index in range(0, NUM_METHODS):
                for iteration in range(0, ITERATIONS):
                    for sublist in time_list[iteration]:
                        print(sublist[method_index])
                        total[method_index] += sublist[method_index]

            average_times = [total_time / ITERATIONS for total_time in total]

            # Plot results
            for node_index in range(0, len(plot_nodes)):
                FILENAME = 'bar_charts/{}-{}-iterations{}-node{}.png'.format(plot_type, plot_methods, ITERATIONS, plot_nodes[node_index])
                proportional1 = np.array(average_times[node_index][0])
                proportional2 = np.array(average_times[node_index][1])
                proportional3 = np.array(average_times[node_index][2])

                uniform = np.array(average_times[node_index][3])
                ehc = np.array(average_times[node_index][4])

                figure = plt.figure(figsize=(12, 6))

                plt.title("Average Time Allocation on Node {}".format(plot_nodes[node_index]))
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
