import sys
import numpy as np
from os.path import exists  # noqa
from tqdm import tqdm

# TODO: Try and fix this absolute path
sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.directed_acyclic_graph import DirectedAcyclicGraph  # noqa
from classes.node import Node  # noqa
from classes.contract_program import ContractProgram  # noqa
from classes import utils  # noqa
from tests.test import Test  # noqa

if __name__ == "__main__":
    np.seterr(all='raise')
    # Total budget for the DAG
    BUDGET = 10
    # The time upper-bound for each quality mapping
    TIME_LIMIT = BUDGET
    # The step size when producing the quality mapping
    TIME_STEP_SIZE = 0.1
    # The time interval when querying for the performance profile
    TIME_INTERVAL = 0.1
    # The quality interval when querying for the performance profile
    QUALITY_INTERVAL = .05
    # For type of performance profile (exact or appproximate)
    EXPECTED_UTILITY_TYPE = "approximate"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the first-order metareasoning problem
    # ----------------------------------------------------------------------------------------
    node_7 = Node(7, [], [], expression_type="contract")
    node_6 = Node(6, [], [], expression_type="contract")
    node_5 = Node(5, [], [], expression_type="contract")
    node_4 = Node(4, [], [], expression_type="contract")
    # Leaf node
    node_3 = Node(3, [node_6, node_7], [], expression_type="contract")

    # Leaf node
    node_2 = Node(2, [node_4, node_5], [], expression_type="contract")

    # For Node
    node_1 = Node(1, [node_2, node_3], [], expression_type="contract")

    # Root node
    root = Node(0, [node_1], [], expression_type="contract")

    # Append the children
    node_7.children = [node_3]
    node_6.children = [node_3]
    node_5.children = [node_2]
    node_4.children = [node_2]
    node_3.children = [node_1]
    node_2.children = [node_1]
    node_1.children = [root]

    # Nodes
    nodes = [root, node_1, node_2, node_3, node_4, node_5, node_6, node_7]
    # Create and verify the DAG from the node list
    program_dag = DirectedAcyclicGraph(nodes, root)

    # ----------------------------------------------------------------------------------------
    # Run Simulations
    # ----------------------------------------------------------------------------------------
    SIMULATIONS = 150
    for _ in tqdm(range(0, SIMULATIONS), desc='Progress Bar', position=0, leave=True):
        # Use a Dirichlet distribution to generate random ppvs
        # growth_factors = utils.dirichlet_growth_factor_generator(dag=program_dag, alpha=.9, lower_bound=.05, upper_bound=10)
        growth_factors = utils.uniform_growth_factor_generator(dag=program_dag, lower_bound=1, upper_bound=4)
        # sum_growth_factors = math.e** (-sum(growth_factors))
        sum_growth_factors = 1
        # Get the meta nodes
        try:
            meta_conditional_index = utils.find_conditional_indices(program_dag, include_meta=True)[-1]
            meta_for_index = utils.find_for_indices(program_dag, include_meta=True)[-1]
        except IndexError:
            meta_conditional_index = -1
            meta_for_index = -1

        # Append a growth factor (c) to each node in the contract program for online performance profile querying
        # This loops through each list in parallel
        for node, generated_c in zip(nodes, growth_factors):
            # Skip any meta/placeholder nodes
            if node.id == meta_conditional_index or node.id == meta_for_index:
                node.c = None
            else:
                # Append the growth rate value to the node object
                node.c = generated_c

        SCALE = 10**3
        # Create the program with some budget
        program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=program_dag, child_programs=None, budget=BUDGET, scale=SCALE, decimals=3, quality_interval=QUALITY_INTERVAL,
                                        time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, full_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE,
                                        possible_qualities=POSSIBLE_QUALITIES, sum_growth_factors=sum_growth_factors)

        # Get all the node_ids that aren't fors or conditionals
        node_indicies_list = utils.find_non_meta_indicies(program_dag)

        test = Test(program_outer, node_indicies_list=node_indicies_list)
        test.contract_program = program_outer

        # Outputs embeded list of expected utilities and allocations
        eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, test_phis=[10, 5, 4, 3, 2, 1, .8, .6, .5, .1, 0], verbose=True)

        # Save the EU and Time data to an external files
        test.save_eu_time_data(eu_time_list=eu_time, eu_file_path="src/tests/large-func/data/eu-data-truncated.txt", time_file_path="src/tests/large-func/data/time-data-truncated.txt", node_indicies=node_indicies_list)
