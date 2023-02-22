import os
import pickle
import sys
import numpy as np
from os.path import exists  # noqa
from tqdm import tqdm

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
    INSTANCES = 1000
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
    EXPECTED_UTILITY_TYPE = "approximate"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)
    # For number of different performance profiles for experiments
    ITERATIONS = 1

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the second-order metareasoning problem (for subtree)
    # ----------------------------------------------------------------------------------------
    # Root Node
    node_inner_1 = Node(13, [], [], expression_type="for", in_child_contract_program=True)
    node_inner_2 = Node(12, [node_inner_1], [], expression_type="contract", in_child_contract_program=True)
    node_inner_3 = Node(11, [node_inner_2], [], expression_type="contract", in_child_contract_program=True)
    node_inner_4 = Node(10, [node_inner_3], [], expression_type="contract", in_child_contract_program=True)
    node_inner_5 = Node(9, [node_inner_4], [], expression_type="contract", in_child_contract_program=True)

    node_inner_2.in_for = True
    node_inner_3.in_for = True
    node_inner_4.in_for = True
    node_inner_5.in_for = True

    node_inner_5.is_last_for_loop = True

    # Add the children
    node_inner_1.children = [node_inner_2]
    node_inner_2.children = [node_inner_3]
    node_inner_3.children = [node_inner_4]
    node_inner_4.children = [node_inner_5]

    # Create a list of the nodes in breadth-first order for the false branch
    nodes_inner = [node_inner_5, node_inner_4, node_inner_3, node_inner_2, node_inner_1]

    # Create and verify the DAG from the node list
    for_subtree = DirectedAcyclicGraph(nodes_inner, node_inner_5)
    for_subtree.number_of_loops = NUMBER_OF_LOOPS

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the second-order metareasoning problem (conditional subtree)
    # Left (True subtree)
    # Make sure all ids are the same for all nodes in multiple DAGs
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_inner_true_4 = Node(7, [], [], expression_type="conditional", in_child_contract_program=True)
    node_inner_true_4.in_true = True
    # Intermediate Nodes
    node_inner_true_3 = Node(5, [node_inner_true_4], [], expression_type="contract", in_child_contract_program=True)
    node_inner_true_3.in_true = True
    node_inner_true_2 = Node(3, [node_inner_true_3], [], expression_type="contract", in_child_contract_program=True)
    node_inner_true_2.in_true = True
    node_inner_true_1 = Node(4, [node_inner_true_3], [], expression_type="contract", in_child_contract_program=True)
    node_inner_true_1.in_true = True
    # Root Node
    node_inner_true_root = Node(1, [node_inner_true_2, node_inner_true_3], [], expression_type="contract",
                                in_child_contract_program=True)
    node_inner_true_root.in_true = True
    # Add the children
    node_inner_true_4.children = [node_inner_true_3]
    node_inner_true_3.children = [node_inner_true_2, node_inner_true_1]
    node_inner_true_2.children = [node_inner_true_root]
    node_inner_true_1.children = [node_inner_true_root]
    # Create a list of the nodes in breadth-first order for the true branch
    nodes_inner_true = [node_inner_true_root, node_inner_true_1, node_inner_true_2, node_inner_true_3,
                        node_inner_true_4]
    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the second-order metareasoning problem (conditional subtree)
    # Right (False subtree)
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_inner_false_2 = Node(7, [], [], expression_type="conditional", in_child_contract_program=True)
    node_inner_false_2.in_false = True
    # Conditional branch nodes
    node_inner_false_1 = Node(6, [node_inner_false_2], [], expression_type="contract", in_child_contract_program=True)
    node_inner_false_1.in_false = True
    # Root nodes
    node_inner_false_root = Node(2, [node_inner_false_1], [], expression_type="contract", in_child_contract_program=True)
    node_inner_false_root.in_false = True
    # Create a list of the nodes in breadth-first order for the false branch
    nodes_inner_false = [node_inner_false_root, node_inner_false_1, node_inner_false_2]

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the first-order metareasoning problem
    # ----------------------------------------------------------------------------------------
    # Leaf nodes
    node_outer_4 = Node(14, [], [], expression_type="contract", in_child_contract_program=False)

    # For Node
    node_outer_3 = Node(13, [node_outer_4], [], expression_type="for", in_child_contract_program=False)
    # node_outer_3.num_loops = NUMBER_OF_LOOPS
    # node_outer_3.for_dag = copy.deepcopy(dag_inner)
    # node_inner_5.subprogram_parent_node = node_outer_3

    for node in [node_inner_2, node_inner_3, node_inner_4, node_inner_5]:
        node.subprogram_parent_node = node_outer_3

    # Intermeditate node from for to conditional
    node_outer_2 = Node(8, [node_outer_3], [], expression_type="contract", in_child_contract_program=False)

    # Append the children

    # Conditional Node
    node_outer_1 = Node(7, [node_outer_2], [], expression_type="conditional", in_child_contract_program=False)
    # Root node
    root_outer = Node(0, [node_outer_1], [], expression_type="contract", in_child_contract_program=False)

    # Append the children
    node_outer_4.children = [node_outer_3]
    node_outer_3.children = [node_outer_2]
    node_outer_2.children = [node_outer_1]
    node_outer_1.children = [root_outer]

    # Nodes
    nodes_outer = [root_outer, node_outer_1, node_outer_2, node_outer_3, node_outer_4]
    # Create and verify the DAG from the node list
    dag_outer = DirectedAcyclicGraph(nodes_outer, root_outer)

    # ----------------------------------------------------------------------------------------
    # Create a program_dag with expanded subtrees for quality mapping generation
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_14 = Node(14, [], [], expression_type="contract", in_child_contract_program=False)

    # Intermediate Nodes
    node_13 = Node(13, [node_14], [], expression_type="for", in_child_contract_program=False)
    node_12 = Node(12, [node_13], [], expression_type="contract", in_child_contract_program=True)
    node_11 = Node(11, [node_12], [], expression_type="contract", in_child_contract_program=True)
    node_10 = Node(10, [node_11], [], expression_type="contract", in_child_contract_program=True)
    node_9 = Node(9, [node_10], [], expression_type="contract", in_child_contract_program=True)

    node_12.in_for = True
    node_11.in_for = True
    node_10.in_for = True
    node_9.in_for = True

    # Transition Node from for to conditional
    node_8 = Node(8, [node_9], [], expression_type="contract", in_child_contract_program=False)

    # Conditional node
    node_7 = Node(7, [node_8], [], expression_type="conditional", in_child_contract_program=False)

    # Conditional branch nodes
    node_6 = Node(6, [node_7], [], expression_type="contract", in_child_contract_program=False)
    node_5 = Node(5, [node_7], [], expression_type="contract", in_child_contract_program=False)

    node_6.in_false = True
    node_5.in_true = True

    # Conditional subtrees
    node_4 = Node(4, [node_5], [], expression_type="contract", in_child_contract_program=False)
    node_3 = Node(3, [node_5], [], expression_type="contract", in_child_contract_program=False)
    node_2 = Node(2, [node_6], [], expression_type="contract", in_child_contract_program=False, is_conditional_root=True)
    node_1 = Node(1, [node_3, node_4], [], expression_type="contract", in_child_contract_program=False, is_conditional_root=True)

    node_4.in_true = True
    node_3.in_true = True
    node_2.in_false = True
    node_1.in_true = True

    # Root node
    root = Node(0, [node_1, node_2], [], expression_type="contract", in_child_contract_program=False)

    # Add the children
    node_1.children = [root]
    node_2.children = [root]
    node_3.children = [node_1]
    node_4.children = [node_1]
    node_5.children = [node_3, node_4]
    node_6.children = [node_2]
    node_7.children = [node_5, node_6]
    node_8.children = [node_7]
    node_9.children = [node_8]
    node_10.children = [node_9]
    node_11.children = [node_10]
    node_12.children = [node_11]
    node_13.children = [node_12]
    node_14.children = [node_13]

    # For a list of nodes for the DAG creation
    nodes = [root, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_9, node_10, node_11, node_12, node_13, node_14]
    program_dag = DirectedAcyclicGraph(nodes, root)

    # Use a Dirichlet distribution to generate random ppvs
    performance_profile_velocities = utils.dirichlet_ppv(iterations=ITERATIONS, dag=program_dag, alpha=.9, constant=10)

    # Use an Analysis ppv to test the avaerage time allocations on varying Cs for a given node
    # c_list = np.arange(.1, 5.1, .1)
    c_list = np.arange(.01, 5.11, .1)
    # c_list = np.arange(.1, 1.1, .2)
    c_node_id = 8
    # performance_profile_velocities = utils.ppv_generator(node_id=c_node_id, dag=program_dag, c_list=c_list, constant=1)

    # Initialize the velocities for the quality mappings in a list
    # Need to initialize it after adjusting program_dag
    # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
    # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
    # performance_profile_velocities = [[10, 20, 0.1, 0.1, 0.1, 0.1, 1000, "conditional", 1000, .1, .1, 100, .1, "for", 10],
    #                                   [10, 20, 0.1, 0.1, 0.1, 0.1, 1000, "conditional", 1000, .1, .1, 100, .1, "for", 10]]

    # performance_profile_velocities = [[1, 20, 0.1, 0.1, 0.1, 0.1, 1000, "conditional", 1000, .1, .1, 100, .1, "for", 10]]

    # eu_list = [[] for i in range(0, NUM_METHODS)]
    # time_list = [[] for i in range(0, NUM_METHODS)]
    times_on_c = [[] for c in range(0, len(c_list))]

    for ppv in tqdm(performance_profile_velocities, desc='Progress Bar'):
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
        program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=dag_outer, child_programs=None, budget=BUDGET, scale=10 ** 6, decimals=3, quality_interval=QUALITY_INTERVAL,
                                        time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=False, generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE,
                                        possible_qualities=POSSIBLE_QUALITIES, performance_profile_velocities=ppv)

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(program_outer)

        # Add the subtree contract programs to the conditional node
        # Add the left subtree
        true_subtree = DirectedAcyclicGraph(nodes_inner_true, root=node_inner_true_root)

        # Convert to a contract program
        node_outer_1.true_subprogram = ContractProgram(program_id=1, parent_program=program_outer, child_programs=None, program_dag=true_subtree, budget=0, scale=10 ** 6, decimals=3,
                                                       quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=True,
                                                       generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, subprogram_expression_type="true")

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(node_outer_1.true_subprogram)

        # Add the right subtree
        false_subtree = DirectedAcyclicGraph(nodes_inner_false, root=node_inner_false_root)

        # Convert to a contract program
        node_outer_1.false_subprogram = ContractProgram(program_id=2, parent_program=program_outer, child_programs=None, program_dag=false_subtree, budget=0, scale=10 ** 6, decimals=3,
                                                        quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=True,
                                                        generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, subprogram_expression_type="false")

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(node_outer_1.false_subprogram)

        # Convert to a contract program
        node_outer_3.for_subprogram = ContractProgram(program_id=3, parent_program=program_outer, child_programs=None, program_dag=for_subtree, budget=0, scale=10 ** 6, decimals=3,
                                                      quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=True, generator_dag=program_dag,
                                                      expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, number_of_loops=NUMBER_OF_LOOPS, subprogram_expression_type="for")

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(node_outer_3.for_subprogram)

        program_outer.child_programs = [node_outer_1.true_subprogram, node_outer_1.false_subprogram, node_outer_3.for_subprogram]

        # Add the pointers from the parent program to the subprograms
        node_outer_1.true_subprogram.parent_program = program_outer
        node_outer_1.false_subprogram.parent_program = program_outer
        node_outer_3.for_subprogram.parent_program = program_outer

        # Add the pointers from the generator dag to the subprograms
        node_outer_1.true_subprogram.generator_dag = program_dag
        node_outer_1.false_subprogram.generator_dag = program_dag
        node_outer_3.for_subprogram.generator_dag = program_dag

        node_outer_1.true_subprogram.subprogram_expression_type = "conditional"
        node_outer_1.false_subprogram.subprogram_expression_type = "conditional"
        node_outer_3.for_subprogram.subprogram_expression_type = "for"

        # Get all the node_ids that aren't fors or conditionals
        node_indicies_list = utils.find_non_meta_indicies(dag=program_dag)

        # TODO: Get rid of None params later
        test = Test(program_outer, ppv, node_indicies_list=node_indicies_list, plot_type=None, plot_nodes=None)
        test.contract_program = program_outer

        # Outputs embeded list of expected utilities and allocations
        eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=True)

        print("PPV: {}".format(ppv))

        # Save the time allcoations for C-variation experimenet
        # times_on_c[ppv_index] += (eu_time[1])

        # Check if any of the EUs are 0
        for eu in eu_time[0]:
            if eu == 0:
                print("Found 0 in EU: {}".format(eu_time[0]))
                exit()

        # Save the EU and Time data to an external files
        test.save_eu_time_data(eu_time_list=eu_time, eu_file_path="src/tests/robotics_domain/data/eu_data_difGenerator.txt", time_file_path="src/tests/robotics_domain/data/time_data_difGenerator.txt", node_indicies=node_indicies_list)

    save_analysis_to_file = False

    if save_analysis_to_file:
        file = "src/tests/robotics_domain/data/time_on_c_data_node8.txt"
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

        with open(file, 'wb') as file_times:
            pickle.dump(pickled_time_list, file_times)
