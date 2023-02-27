import copy
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
    TIME_INTERVAL = 0.1
    # The quality interval when querying for the performance profile
    QUALITY_INTERVAL = .05
    NUMBER_OF_LOOPS = 4
    # For type of performance profile (exact or appproximate)
    EXPECTED_UTILITY_TYPE = "approximate"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)
    # The number of methods for experimentation
    NUM_METHODS = 13
    # For number of different performance profiles for experiments
    ITERATIONS = 47

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the first-order metareasoning problem
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_3 = Node(3, [], [], expression_type="contract", in_child_contract_program=False)

    # Leaf node
    node_2 = Node(2, [], [], expression_type="contract", in_child_contract_program=False)

    # For Node
    node_1 = Node(1, [node_2, node_3], [], expression_type="contract", in_child_contract_program=False)

    # Root node
    root = Node(0, [node_1], [], expression_type="contract", in_child_contract_program=False)

    # Append the children
    node_3.children = [node_1]
    node_2.children = [node_1]
    node_1.children = [root]

    # Nodes
    nodes = [root, node_1, node_2, node_3]
    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # ----------------------------------------------------------------------------------------
    # Create a program_dag with expanded subtrees for quality mapping generation
    # ----------------------------------------------------------------------------------------
    # Copy the main dag since we don't need to expand any subtrees and adjust pointers
    program_dag = copy.deepcopy(dag)

    # Use a Dirichlet distribution to generate random ppvs
    performance_profile_velocities = utils.dirichlet_ppv(iterations=ITERATIONS, dag=program_dag, alpha=.9, constant=10)

    # performance_profile_velocities = [[10, 20, 0.1]]
    for ppv in tqdm(performance_profile_velocities, desc='Progress Bar'):
        # Used to create the synthetic data as instances and a populous file
        generate = True
        if not exists("quality_mappings/populous.json") or generate:
            # Initialize a generator
            generator = Generator(INSTANCES, program_dag=program_dag, time_limit=TIME_LIMIT, time_step_size=TIME_STEP_SIZE,
                                  uniform_low=0.05,
                                  uniform_high=0.9)

            # Adjust the DAG structure that has conditionals for generation
            generator.full_dag = generator.adjust_dag_with_fors(program_dag)

            # Adjust the DAG structure that has conditionals for generation
            generator.full_dag = generator.adjust_dag_with_conditionals(generator.full_dag)

            for i in generator.full_dag.nodes:
                print("full_dag (children): {}, {}".format(i.id, [j.id for j in i.children]))
            for i in generator.full_dag.nodes:
                print("full_dag (parents): {}, {}".format(i.id, [j.id for j in i.parents]))

            generator.activate_manual_override(ppv)

            # Generate the nodes' quality mappings
            nodes = generator.generate_nodes()  # Return a list of file names of the nodes
            # populate the nodes' quality mappings into one populous file
            generator.populate(nodes, "quality_mappings/populous.json")

        # Create the program with some budget
        program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=dag, child_programs=None, budget=BUDGET, scale=10, decimals=3, quality_interval=QUALITY_INTERVAL,
                                        time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_child_contract_program=False, full_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE,
                                        possible_qualities=POSSIBLE_QUALITIES, performance_profile_velocities=ppv)

        # Initialize the pointers of the nodes to the program it is in
        utils.initialize_node_pointers_current_program(program_outer)

        # Get all the node_ids that aren't fors or conditionals
        node_indicies_list = utils.find_non_meta_indicies(program_dag)

        # TODO: Get rid of None params later
        test = Test(program_outer, ppv, node_indicies_list=node_indicies_list, num_plot_methods=NUM_METHODS, plot_type=None, plot_nodes=None)
        test.contract_program = program_outer

        # Outputs embeded list of expected utilities and allocations
        eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=True)

        # Check if any of the EUs are 0
        found_zero = False
        for eu in eu_time[0]:
            if eu == 0:
                performance_profile_velocities.extend(utils.dirichlet_ppv(iterations=1, dag=program_dag, alpha=.9, constant=10))
                found_zero = True
                print("Found 0 in EU")
                exit()

        if not found_zero:
            # Save the EU and Time data to an external files
            test.save_eu_time_data(eu_time_list=eu_time, eu_file_path="src/tests/med-func/data/eu_data.txt", time_file_path="src/tests/med-func/data/time_data.txt", node_indicies=node_indicies_list, num_methods=NUM_METHODS)
