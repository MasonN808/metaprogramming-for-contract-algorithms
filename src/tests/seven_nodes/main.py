import sys

sys.path.append("/Users/masonnakamura/Local-Git/mca/src")

from classes.directed_acyclic_graph import DirectedAcyclicGraph  # noqa
from classes.nodes.node import Node  # noqa
from classes.contract_program import ContractProgram  # noqa
from classes.generator import Generator  # noqa
from tests.test import Test  # noqa
from os.path import exists  # noqa

if __name__ == "__main__":
    BUDGET = 10
    INSTANCES = 5
    TIME_LIMIT = BUDGET
    STEP_SIZE = 0.1
    QUALITY_INTERVAL = .05
    VERBOSE = False

    # Create a DAG manually for testing
    # Leaf nodes
    node_3 = Node(3, [], [], expression_type="contract", in_subtree=False)
    node_4 = Node(4, [], [], expression_type="contract", in_subtree=False)
    node_5 = Node(5, [], [], expression_type="contract", in_subtree=False)
    node_6 = Node(6, [], [], expression_type="contract", in_subtree=False)

    # Intermediate nodes
    node_1 = Node(1, [node_3, node_4], [], expression_type="contract", in_subtree=False)
    node_2 = Node(2, [node_5, node_6], [], expression_type="contract", in_subtree=False)

    # Root node
    root = Node(0, [node_1, node_2], [], expression_type="contract", in_subtree=False)

    # Add the children
    node_1.children = [root]
    node_2.children = [root]
    node_3.children = [node_1]
    node_4.children = [node_1]
    node_5.children = [node_2]
    node_6.children = [node_2]

    # Nodes
    nodes = [root, node_1, node_2, node_3, node_4, node_5, node_6]

    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # Used to create the synthetic data as instances and a populous file
    generate = True
    if not exists("populous.json") or generate:
        # Initialize a generator
        generator = Generator(INSTANCES, program_dag=dag, time_limit=TIME_LIMIT, time_step_size=STEP_SIZE, uniform_low=.05,
                              uniform_high=.9, generator_dag=dag)

        # Initialize the velocities for the quality mappings in a list
        # Need to initialize it after adjusting program_dag
        # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
        # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
        generator.manual_override = [10000, 0.1, 0.1, 0.1, 0.1, 10000, 10000]

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # Create the program with some budget
    program = ContractProgram(program_dag=dag, budget=BUDGET, scale=10**6, decimals=3, quality_interval=QUALITY_INTERVAL, time_interval=.1, time_step_size=STEP_SIZE, child_programs=None, generator_dag=dag,
                              in_subtree=False, parent_program=None, program_id=0)

    test = Test(program)

    # Test initial vs optimal expected utility and allocations
    test.find_utility_and_allocations(initial_allocation="uniform", verbose=False, outer_program=program)
    test.find_utility_and_allocations(initial_allocation="uniform with noise", verbose=False, outer_program=program)
    test.find_utility_and_allocations(initial_allocation="Dirichlet", verbose=False, outer_program=program)
