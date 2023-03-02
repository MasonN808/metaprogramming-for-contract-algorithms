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
    NUMBER_OF_LOOPS = 4
    # For type of performance profile (exact or appproximate)
    EXPECTED_UTILITY_TYPE = "approximate"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the second-order metareasoning problem (conditional subtree)
    # Left (True subtree)
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_inner_true_1 = Node(3, [], [], expression_type="conditional", program_id=1)
    # node_inner_true_1.in_true = True

    # Root Node
    node_inner_true_root = Node(1, [node_inner_true_1], [], expression_type="contract", program_id=1, is_conditional_root=True)
    # node_inner_true_root.in_true = True

    # Create a list of the nodes in breadth-first order for the true branch
    nodes_inner_true = [node_inner_true_root, node_inner_true_1]

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the second-order metareasoning problem (conditional subtree)
    # Right (False subtree)
    # ----------------------------------------------------------------------------------------
    # Leaf node
    node_inner_false_1 = Node(3, [], [], expression_type="conditional", program_id=2)
    # node_inner_false_1.in_false = True

    # Root node
    node_inner_false_root = Node(2, [node_inner_false_1], [], expression_type="contract", program_id=2, is_conditional_root=True)
    # node_inner_false_root.in_false = True
    # Create a list of the nodes in breadth-first order for the false branch
    nodes_inner_false = [node_inner_false_root, node_inner_false_1]

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the first-order metareasoning problem
    # ----------------------------------------------------------------------------------------
    # conditional nodes
    node_outer_1 = Node(3, [], [], expression_type="conditional", program_id=0)

    # Root node
    root_outer = Node(0, [node_outer_1], [], expression_type="contract", program_id=0)

    # Append the children
    # node_outer_1.children = [root_outer]

    nodes_outer = [root_outer, node_outer_1]
    # Create and verify the DAG from the node list
    dag_outer = DirectedAcyclicGraph(nodes_outer, root_outer)

    # ----------------------------------------------------------------------------------------
    # Create the expanded or full DAG
    # ----------------------------------------------------------------------------------------
    # Conditional node
    node_3 = Node(3, [], [], expression_type="conditional")

    # false node
    node_2 = Node(2, [node_3], [], expression_type="contract", in_false=True, is_conditional_root=True)

    # true Node
    node_1 = Node(1, [node_3], [], expression_type="contract", in_true=True, is_conditional_root=True)

    # Root node
    root = Node(0, [node_1], [], expression_type="contract")

    # Append the children
    # node_3.children = [node_1, node_2]
    # node_2.children = [root]
    # node_1.children = [root]

    # Nodes
    nodes = [root, node_1, node_2, node_3]
    # Create and verify the DAG from the node list
    program_dag = DirectedAcyclicGraph(nodes, root)

    scale = 10**2
    # Create the program with so budget
    program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=dag_outer, child_programs=None, budget=BUDGET, scale=scale, decimals=3, quality_interval=QUALITY_INTERVAL,
                                    time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, full_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE,
                                    possible_qualities=POSSIBLE_QUALITIES)

    # ----------------------------------------------------------------------------------------
    #   Define the subprograms
    # ----------------------------------------------------------------------------------------
    # Add the subtree contract programs to the conditional node
    # Add the left subtree
    true_subtree = DirectedAcyclicGraph(nodes_inner_true, root=node_inner_true_root)
    # Convert to a contract program
    node_outer_1.true_subprogram = ContractProgram(program_id=1, parent_program=program_outer, child_programs=None, program_dag=true_subtree, budget=0, scale=scale, decimals=3,
                                                   quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE,
                                                   full_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, subprogram_expression_type="true")
    node_3.true_subprogram = node_outer_1.true_subprogram

    # Add the right subtree
    false_subtree = DirectedAcyclicGraph(nodes_inner_false, root=node_inner_false_root)
    # Convert to a contract program
    node_outer_1.false_subprogram = ContractProgram(program_id=2, parent_program=program_outer, child_programs=None, program_dag=false_subtree, budget=0, scale=scale, decimals=3,
                                                    quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE,
                                                    full_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, subprogram_expression_type="false")
    node_3.false_subprogram = node_outer_1.false_subprogram

    # Define a hashmap to pull contract subprograms during recursive optimization
    program_outer.subprogram_map = {
        # Here they key is "{node id} [optional]-0{false branch} {or} 1 {true branch}"
        "3-1": node_outer_1.true_subprogram,
        "3-0": node_outer_1.false_subprogram
    }

    # ----------------------------------------------------------------------------------------
    # Run Simulations
    # ----------------------------------------------------------------------------------------
    SIMULATIONS = 1
    for _ in tqdm(range(0, SIMULATIONS), desc='Progress Bar', position=0, leave=True):
        # Use a Dirichlet distribution to generate random ppvs
        # growth_factors = utils.uniform_growth_factor_generator(dag=program_dag, lower_bound=.05, upper_bound=10)
        growth_factors = utils.dirichlet_growth_factor_generator(dag=program_dag, alpha=.9, lower_bound=.05, upper_bound=10)
        # growth_factors = [6, .2, .2, 1]
        # sum_growth_factors = math.e** sum(growth_factors)
        # Get the meta nodes
        try:
            meta_conditional_index = utils.find_conditional_indices(program_dag, include_meta=True)[-1]
            meta_for_index = utils.find_for_indices(program_dag, include_meta=True)[-1]
        except IndexError:
            meta_conditional_index = -1
            meta_for_index = -1

        # Append a growth factor (c) to each node in the contract program for online performance profile querying
        # This loops through each list in parallel
        for node, generated_c in zip(program_dag.nodes, growth_factors):
            # Skip any meta/placeholder nodes
            if node.id == meta_conditional_index or node.id == meta_for_index:
                node.c = None
            else:
                # Append the growth rate value to the node object
                node.c = generated_c

        # Append the growth factors to the subprograms
        program_outer.append_growth_factors_to_subprograms()

        # Get all the node_ids that aren't fors or conditionals
        node_indicies_list = utils.find_non_meta_indicies(program_dag)

        # TODO: Get rid of None params later
        test = Test(program_outer, node_indicies_list=node_indicies_list, plot_type=None, plot_nodes=None)
        test.contract_program = program_outer

        # Outputs embeded list of expected utilities and allocations
        eu_time = test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, test_phis=[10, 5, 4, 3, 2, 1, .8, .6, .5, .1, 0], verbose=True)
        # Save the EU and Time data to an external files
        test.save_eu_time_data(eu_time_list=eu_time, eu_file_path="src/tests/small-conditional/data/eu-data.txt", time_file_path="src/tests/small-conditional/data/time-data.txt", node_indicies=node_indicies_list)
