from src.classes.directed_acyclic_graph import DirectedAcyclicGraph
from src.classes.node import Node
from src.classes.contract_program import ContractProgram
from src.classes.generator import Generator
from src.tests.test import Test
from os.path import exists

if __name__ == "__main__":
    # Total budget for the DAG
    BUDGET = 10
    # Number of instances/simulations
    INSTANCES = 5
    # The time upper-bound for each quality mapping
    TIME_LIMIT = BUDGET
    # The step size when producing the quality mapping
    TIME_STEP_SIZE = 0.1
    # The time interval when querying for the performance profile
    TIME_INTERVAL = .01
    # The quality interval when querying for the performance profile
    QUALITY_INTERVAL = .05
    # For debugging
    VERBOSE = False

    # Create a DAG manually for testing

    # Leaf nodes
    node_6 = Node(6, [], [], expression_type="contract")
    node_4 = Node(4, [], [], expression_type="contract")

    # Intermediate Nodes
    node_5 = Node(5, [node_6], [], expression_type="contract")

    # Conditional Node
    node_3 = Node(3, [node_4, node_5], [], expression_type="conditional")

    # Conditional branch nodes
    node_1 = Node(1, [node_3], [], expression_type="contract")
    node_2 = Node(2, [node_3], [], expression_type="contract")

    # Root node
    root = Node(0, [node_1, node_2], [], expression_type="contract")

    # Add the children
    node_1.children = [root]
    node_2.children = [root]
    node_3.children = [node_1, node_2]
    node_4.children = [node_3]
    node_5.children = [node_3]
    node_6.children = [node_5]

    # Nodes
    nodes = [root, node_1, node_2, node_3, node_4, node_5, node_6]

    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # Used to create the synthetic data as instances and a populous file
    generate = True
    if not exists("populous.json") or generate:
        # Initialize a generator
        generator = Generator(INSTANCES, program_dag=dag, time_limit=TIME_LIMIT, time_step_size=TIME_STEP_SIZE, uniform_low=0.05,
                              uniform_high=0.9)

        # Let the root be trivial and not dependent on parents
        # generator.trivial_root = True

        # Adjust the DAG structure that has conditionals for generation
        generator.full_dag = generator.adjust_dag_with_conditionals(dag)

        # Initialize the velocities for the quality mappings in a list
        # Need to initialize it after adjusting program_dag
        # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
        # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
        generator.manual_override = [0.1, 0.1, 0.1, "conditional", 10000, 10000, 10]

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # Create the program with some budget

    program = ContractProgram(dag, BUDGET, scale=10**6, decimals=3, quality_interval=QUALITY_INTERVAL,
                              time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE)

    # Adjust allocations (hardcode)
    test = Test(program)

    # Test a random distribution on the initial allocations
    # print(test.test_initial_allocations(iterations=500, initial_is_random=True, verbose=False))

    # Test initial vs optimal expected utility and allocations
    test.find_utility_and_allocations(initial_allocation="uniform", verbose=False)
    test.find_utility_and_allocations(initial_allocation="uniform with noise", verbose=False)
    test.find_utility_and_allocations(initial_allocation="Dirichlet", verbose=False)
