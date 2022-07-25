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

    # Create a DAG manually for the conditional subtree
    # Leaf nodes
    node_inner_7 = Node(7, [], [], expression_type="conditional")

    # Conditional branch nodes
    node_inner_6 = Node(6, [node_inner_7], [], expression_type="contract")
    node_inner_5 = Node(5, [node_inner_7], [], expression_type="contract")

    # Conditional subtrees
    node_inner_4 = Node(4, [node_inner_5], [], expression_type="contract")
    node_inner_3 = Node(3, [node_inner_5], [], expression_type="contract")
    node_inner_2 = Node(2, [node_inner_6], [], expression_type="contract")
    node_inner_1 = Node(1, [node_inner_3, node_inner_4], [], expression_type="contract")

    # Root node
    # Add the evaluation node to compose the two branches together for the expected utility of the contract expression
    # Note: this node will not be included in the 1st-level metareasoning process
    root_inner = Node(0, [node_inner_1, node_inner_2], [], expression_type="contract")

    # Assign it as trivial for proper quality mapping generation
    root_inner.trivial = True

    # Add the children
    node_inner_1.children = [root_inner]
    node_inner_2.children = [root_inner]
    node_inner_3.children = [node_inner_1]
    node_inner_4.children = [node_inner_1]
    node_inner_5.children = [node_inner_3, node_inner_4]
    node_inner_6.children = [node_inner_2]
    node_inner_7.children = [node_inner_5, node_inner_6]

    # For a list of nodes for the DAG creation
    nodes_inner = [root_inner, node_inner_1, node_inner_2, node_inner_3, node_inner_4, node_inner_5, node_inner_6, node_inner_7]

    # Create a DAG manually for the first-order metareasoning problem
    # Leaf nodes
    node_outer_2 = Node(2, [], [], expression_type="contract")

    # Conditional Node
    node_outer_1 = Node(1, [node_outer_2], [], expression_type="conditional")

    # Add the subtree to the conditional node
    node_outer_1.subtree = DirectedAcyclicGraph(nodes_inner, root=root_inner)

    # Root node
    root_outer = Node(0, [node_outer_1, node_outer_2], [], expression_type="contract")

    # Nodes
    nodes = [root_outer, node_outer_1, node_outer_2]

    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root_outer)

    # Used to create the synthetic data as instances and a populous file
    generate = True
    if not exists("populous.json") or generate:
        # Initialize a generator
        generator = Generator(INSTANCES, program_dag=dag, time_limit=TIME_LIMIT, time_step_size=TIME_STEP_SIZE, uniform_low=0.05,
                              uniform_high=0.9)

        # Let the root be trivial and not dependent on parents
        # generator.trivial_root = True

        # Adjust the DAG structure that has conditionals for generation
        generator.generator_dag = generator.adjust_dag_with_conditionals(dag)

        # Initialize the velocities for the quality mappings in a list
        # Need to initialize it after adjusting dag
        # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
        # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
        generator.manual_override = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, "conditional", 10000]

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # Create the program with some budget

    program = ContractProgram(dag, BUDGET, scale=10**6, decimals=3, quality_interval=QUALITY_INTERVAL,
                              time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE)

    # Adjust allocations (hardcode)
    test = Test(program)

    # Print the tree for verification
    # print(test.print_tree(dag.root))

    # Test a random distribution on the initial allocations
    # print(test.test_initial_allocations(iterations=500, initial_is_random=True, verbose=False))

    # Test initial vs optimal expected utility and allocations
    test.find_utility_and_allocations(initial_allocation="uniform", verbose=False)
    test.find_utility_and_allocations(initial_allocation="uniform with noise", verbose=False)
    test.find_utility_and_allocations(initial_allocation="Dirichlet", verbose=False)
