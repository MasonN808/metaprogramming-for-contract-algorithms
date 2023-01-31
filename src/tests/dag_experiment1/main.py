import copy
import os
import pickle
import sys
import numpy as np
from os.path import exists  # noqa

# TODO: Try and fix this absolute path
sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.directed_acyclic_graph import DirectedAcyclicGraph  # noqa
from classes.node import Node  # noqa
from classes.contract_program import ContractProgram  # noqa
from classes.generator import Generator  # noqa
from classes import utils  # noqa
from tests.test import Test  # noqa

if __name__ == "__main__":
    # Total budget for the DAG
    BUDGET = 10
    # Number of instances/simulations
    INSTANCES = 10
    # The time upper-bound for each quality mapping
    TIME_LIMIT = BUDGET
    # The step size when producing the quality mapping
    TIME_STEP_SIZE = 0.1
    # The time interval when querying for the performance profile
    TIME_INTERVAL = .01
    TIME_INTERVAL = 0.1
    # The quality interval when querying for the performance profile
    QUALITY_INTERVAL = .05
    NUMBER_OF_LOOPS = 4
    # For type of performance profile (exact or appproximate)
    EXPECTED_UTILITY_TYPE = "exact"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)
    # The number of methods for experimentation
    NUM_METHODS = 13
    # For number of different performance profiles for experiments
    ITERATIONS = 100

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the first-order metareasoning problem
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_2 = Node(2, [], [], expression_type="contract", in_child_contract_program=False)

    # For Node
    node_1= Node(1, [node_2], [], expression_type="contract", in_child_contract_program=False)

    # Root node
    root = Node(0, [node_1], [], expression_type="contract", in_child_contract_program=False)

    # Append the children
    node_2.children = [node_1]
    node_1.children = [root]

    # Nodes
    nodes = [root, node_1, node_2]
    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # ----------------------------------------------------------------------------------------
    # Create a program_dag with expanded subtrees for quality mapping generation
    # ----------------------------------------------------------------------------------------
    # Copy the main dag since we don't need to expand any subtrees and adjust pointers
    program_dag = copy.deepcopy(dag)

    # Use a Dirichlet distribution to generate random ppvs
    performance_profile_velocities = utils.dirichlet_ppv(iterations=ITERATIONS, dag=program_dag, alpha=.9, constant=10)

    # Use an Analysis ppv to test the avaerage time allocations on varying Cs for a given node
    # c_list = np.arange(.1, 5.1, .1)
    c_list = np.arange(.01, 5.11, .1)
    # c_list = np.arange(.1, 1.1, .2)
    c_node_id = 6
    # performance_profile_velocities = utils.ppv_generator(node_id=c_node_id, dag=program_dag, c_list=c_list, constant=1)

    # Initialize the velocities for the quality mappings in a list
    # Need to initialize it after adjusting program_dag
    # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
    # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
    # performance_profile_velocities = [[10, 20, 0.1, 0.1, 0.1, 0.1, 1000, "conditional", 1000, .1, .1, 100, .1, "for", 10],
    #                                   [10, 20, 0.1, 0.1, 0.1, 0.1, 1000, "conditional", 1000, .1, .1, 100, .1, "for", 10]]

    performance_profile_velocities = [[10, 20, 0.1, 0.1, 0.1, 0.1, 1000, "conditional", 1000, .1, .1, 100, .1, "for", 10]]

    # eu_list = [[] for i in range(0, NUM_METHODS)]
    # time_list = [[] for i in range(0, NUM_METHODS)]
    times_on_c = [[] for c in range(0, len(c_list))]

    for ppv_index, ppv in enumerate(performance_profile_velocities):
        # Used to create the synthetic data as instances and a populous file
        generate = True
        if not exists("quality_mappings/populous.json") or generate:
            # Initialize a generator
            generator = Generator(INSTANCES, program_dag=program_dag, time_limit=TIME_LIMIT, time_step_size=TIME_STEP_SIZE,
                                  uniform_low=0.05,
                                  uniform_high=0.9)

            # Adjust the DAG structure that has conditionals for generation
            generator.generator_dag = generator.adjust_dag_with_fors(program_dag)

            # Adjust the DAG structure that has conditionals for generation
            generator.generator_dag = generator.adjust_dag_with_conditionals(generator.generator_dag)

            for i in generator.generator_dag.nodes:
                print("generator_dag (children): {}, {}".format(i.id, [j.id for j in i.children]))
            for i in generator.generator_dag.nodes:
                print("generator_dag (parents): {}, {}".format(i.id, [j.id for j in i.parents]))

            generator.activate_manual_override(ppv)

            # Generate the nodes' quality mappings
            nodes = generator.generate_nodes()  # Return a list of file names of the nodes
            # populate the nodes' quality mappings into one populous file
            generator.populate(nodes, "quality_mappings/populous.json")

        # Create the program with some budget
        program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=dag, child_programs=None, budget=BUDGET, scale=10 ** 6, decimals=3, quality_interval=QUALITY_INTERVAL,
                                        time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=False, generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE,
                                        possible_qualities=POSSIBLE_QUALITIES, performance_profile_velocities=ppv)

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(program_outer)

        # Get all the node_ids that aren't fors or conditionals
        node_indicies_list = [0, 1, 2]

        # TODO: Get rid of None params later
        test = Test(program_outer, ppv, node_indicies_list=node_indicies_list, num_plot_methods=NUM_METHODS, plot_type=None, plot_nodes=None)

        # Outputs embeded list of expected utilities and allocations
        eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=True)

        # Save the time allcoations for C-variation experimenet
        times_on_c[ppv_index] += (eu_time[1])

        save_to_external = True

        if save_to_external:
            file_str_eus = "src/tests/dag_experiment/data/eu_data.txt"
            file_str_times = "src/tests/dag_experimentdata/time_data.txt"
            # Check if data files exist
            if not os.path.isfile(file_str_eus):
                with open(file_str_eus, 'wb') as file_eus:
                    pickle.dump([[] for i in range(0, NUM_METHODS)], file_eus)

            if not os.path.isfile(file_str_times):
                with open(file_str_times, 'wb') as file_times:
                    pickle.dump([[[] for j in range(0, NUM_METHODS)] for i in range(0, len(node_indicies_list))], file_times)

            # Open files in binary mode with wb instead of w
            file_eus = open(file_str_eus, 'rb')
            file_times = open(file_str_times, 'rb')

            # Load the saved embedded lists to append new data
            pickled_eu_list = pickle.load(file_eus)
            pickled_time_list = pickle.load(file_times)

            # Append the EUs appropriately to list in outer scope
            for method_index in range(0, NUM_METHODS):
                pickled_eu_list[method_index].append(eu_time[0][method_index])
                for node in range(0, len(node_indicies_list)):
                    pickled_time_list[node][method_index].append(eu_time[1][node][method_index])

            with open(file_str_eus, 'wb') as file_eus:
                pickle.dump(pickled_eu_list, file_eus)

            with open(file_str_times, 'wb') as file_times:
                pickle.dump(pickled_time_list, file_times)

    save_analysis_to_file = False

    if save_analysis_to_file:
        file = "data/time_on_c_data_node6_TEST2.txt"
        # Check if data files exist
        if not os.path.isfile(file):
            with open(file, 'wb') as file_times:
                pickle.dump([[] for i in range(0, len(performance_profile_velocities))], file_times)

        # Open file in binary mode with wb instead of w
        file_times = open(file, 'rb')

        # Load the saved embedded list to append new data
        pickled_time_list = pickle.load(file_times)

        # Append the EUs appropriately to list in outer scope
        for ppv_index in range(0, len(performance_profile_velocities)):
            pickled_time_list[ppv_index] += (times_on_c[ppv_index])

        print(pickled_time_list)

        with open(file, 'wb') as file_times:
            pickle.dump(pickled_time_list, file_times)
