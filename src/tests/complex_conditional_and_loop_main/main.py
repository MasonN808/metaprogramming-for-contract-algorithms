# import copy
import sys
from matplotlib import pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt

sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.directed_acyclic_graph import DirectedAcyclicGraph  # noqa
from classes.node import Node  # noqa
from classes.contract_program import ContractProgram  # noqa
from classes.generator import Generator  # noqa
from classes import utils  # noqa
from tests.test import Test  # noqa
from os.path import exists  # noqa

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
    EXPECTED_UTILITY_TYPE = "approximate"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)

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

    # Rollout the for loop in a seperate DAG
    # dag_inner_rolled_out = Generator.rollout_for_loops(dag_inner)

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
    node_inner_true_root = Node(1, [node_inner_true_1, node_inner_true_2], [], expression_type="contract",
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
    # node_13.num_loops = NUMBER_OF_LOOPS
    # for_dag = copy.deepcopy(dag_inner)
    # for_dag.nodes = for_dag.nodes[0:len(for_dag.nodes) - 1]
    # node_13.for_dag = copy.deepcopy(for_dag)
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

    node_6.in_true = True
    node_5.in_true = True

    # Conditional subtrees
    node_4 = Node(4, [node_5], [], expression_type="contract", in_child_contract_program=False)
    node_3 = Node(3, [node_5], [], expression_type="contract", in_child_contract_program=False)
    node_2 = Node(2, [node_6], [], expression_type="contract", in_child_contract_program=False, is_conditional_root=True)
    node_1 = Node(1, [node_3, node_4], [], expression_type="contract", in_child_contract_program=False, is_conditional_root=True)

    node_4.in_true = True
    node_3.in_true = True
    node_2.in_true = True
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

    iterations = 1

    # Use a Dirichlet distribution to generate random ppvs
    performance_profile_velocities = utils.dirichlet_ppv(iterations=iterations, dag=program_dag, alpha=.9, constant=10)

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

    eu_list = [[] for i in range(0, 13)]
    time_list = [[] for i in range(0, 13)]

    for ppv in performance_profile_velocities:
        # Used to create the synthetic data as instances and a populous file
        generate = True
        if not exists("populous.json") or generate:
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

            # Initialize the velocities for the quality mappings in a list
            # Need to initialize it after adjusting program_dag
            # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
            # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
            generator.activate_manual_override(ppv)

            # Generate the nodes' quality mappings
            nodes = generator.generate_nodes()  # Return a list of file names of the nodes
            # populate the nodes' quality mappings into one populous file
            generator.populate(nodes, "populous.json")

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
                                                       generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES)

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(node_outer_1.true_subprogram)

        # Add the right subtree
        false_subtree = DirectedAcyclicGraph(nodes_inner_false, root=node_inner_false_root)

        # Convert to a contract program
        node_outer_1.false_subprogram = ContractProgram(program_id=2, parent_program=program_outer, child_programs=None, program_dag=false_subtree, budget=0, scale=10 ** 6, decimals=3,
                                                        quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=True,
                                                        generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES)

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(node_outer_1.false_subprogram)

        # Convert to a contract program
        node_outer_3.for_subprogram = ContractProgram(program_id=3, parent_program=program_outer, child_programs=None, program_dag=for_subtree, budget=0, scale=10 ** 6, decimals=3,
                                                      quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=True, generator_dag=program_dag,
                                                      expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, number_of_loops=NUMBER_OF_LOOPS)

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

        # Verify we have valid plot params
        if (plot_type != "box_whisker" and plot_type != "bar"):
            ValueError("Invalid plot type")
        if (plot_methods != "all" and plot_methods != "subset"):
            ValueError("Invalid plot methods value")

        # The input should be the outermost program
        test = Test(program_outer, ppv, plot_type, plot_methods, plot_nodes)

        if (plot_type == "box_whisker"):
            if (plot_methods == "all"):
                # Test solution method expected utilities and allocations
                eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=False)
                # Append the EUs appropriately to list in outer scope
                for index in range(0, 13):
                    eu_list[index].append(eu_time[0][index])
            elif (plot_methods == "subset"):
                eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=False)
                for index in range(0, 5):
                    eu_list[index].append(eu_time[0][index])

        elif (plot_type == "bar"):
            if (plot_methods == "all"):
                # Test solution method expected utilities and allocations
                eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=False)
                # Append the EUs appropriately to list in outer scope
                for index in range(0, 13):
                    time_list[index].append(eu_time[1][index])
                pass
            elif (plot_methods == "subset"):
                eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=False)
                for index in range(0, 5):
                    time_list[index].append(eu_time[1][index])

    # print(eu_list)

    FILENAME = '{}-{}-{}.png'.format(plot_type, plot_methods, iterations)

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
