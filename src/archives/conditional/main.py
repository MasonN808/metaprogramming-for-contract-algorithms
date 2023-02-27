from os.path import exists
from tests.test import Test
from classes.generator import Generator
from classes.contract_program import ContractProgram
from classes.node import Node
from classes.directed_acyclic_graph import DirectedAcyclicGraph
import sys

sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")


if __name__ == "__main__":
    BUDGET = 10
    INSTANCES = 5
    TIME_LIMIT = BUDGET
    STEP_SIZE = 0.1
    QUALITY_INTERVAL = .05
    VERBOSE = False

    # Create a DAG manually for testing
    # Leaf nodes
    node_4 = Node(4, [], [], expression_type="contract", in_child_contract_program=False)
    node_5 = Node(5, [], [], expression_type="contract", in_child_contract_program=False)

    # Conditional Node
    node_3 = Node(3, [node_4, node_5], [], expression_type="conditional", in_child_contract_program=True)

    # Intermediate nodes
    node_1 = Node(1, [node_3], [], expression_type="contract", in_child_contract_program=True)
    node_2 = Node(2, [node_3], [], expression_type="contract", in_child_contract_program=True)

    # Root node
    root = Node(0, [node_1, node_2], [], expression_type="contract", in_child_contract_program=False)

    # Add the children
    node_1.children = [root]
    node_2.children = [root]
    node_3.children = [node_1, node_2]
    node_4.children = [node_3]
    node_5.children = [node_3]

    # Nodes
    nodes = [root, node_1, node_2, node_3, node_4, node_5]

    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # Used to create the synthetic data as instances and a populous file
    generate = False
    if not exists("populous.json") or generate:
        # Initialize a generator
        generator = Generator(INSTANCES, dag, time_limit=TIME_LIMIT, time_step_size=STEP_SIZE, uniform_low=.05,
                              uniform_high=.9)

        # Adjust the DAG structure that has conditionals for generation
        generator.full_dag = generator.adjust_dag_with_conditionals(dag)

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # Create the program with some budget
    program = ContractProgram(program_dag=dag, budget=BUDGET, scale=10**6, decimals=3, quality_interval=QUALITY_INTERVAL, time_interval=.1, time_step_size=STEP_SIZE, child_programs=None, full_dag=dag,
                              in_child_contract_program=False, parent_program=None, program_id=0)

    # Adjust allocations (hardcode)
    test = Test(program)

    # Print the tree
    # print(test.print_tree(program_dag.root))

    # Test a random distribution on the initial allocations
    # print(test.test_initial_allocations(iterations=500, initial_is_random=True, verbose=False))

    # Test initial vs optimal expected utility and allocations
    test.find_utility_and_allocations(initial_allocation="uniform", verbose=False, outer_program=program)
    test.find_utility_and_allocations(initial_allocation="Dirichlet", verbose=False, outer_program=program)
