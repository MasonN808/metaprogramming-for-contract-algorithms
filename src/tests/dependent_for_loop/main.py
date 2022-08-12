# from src.classes.directed_acyclic_graph import DirectedAcyclicGraph
# from src.classes.nodes.node import Node
# from src.classes.contract_program import ContractProgram
# from src.classes.generator import Generator
# from src.tests.test import Test
# from os.path import exists
#
#
# def initialize_node_pointers_current_program(contract_program):
#     for node in contract_program.program_dag.nodes:
#         node.current_program = contract_program
#
#
# if __name__ == "__main__":
#     # Total budget for the DAG
#     BUDGET = 10
#     # Number of instances/simulations
#     INSTANCES = 5
#     # The time upper-bound for each quality mapping
#     TIME_LIMIT = BUDGET
#     # The step size when producing the quality mapping
#     TIME_STEP_SIZE = 0.1
#     # The time interval when querying for the performance profile
#     TIME_INTERVAL = 0.1
#     # The quality interval when querying for the performance profile
#     QUALITY_INTERVAL = .05
#     # For debugging
#     VERBOSE = False
#
#     # ----------------------------------------------------------------------------------------
#     # Create a DAG manually for the second-order metareasoning problem (simple_conditional subtree)
#     # Make sure all ids are the same for all nodes in multiple DAGs
#     # ----------------------------------------------------------------------------------------
#
#     # ----------------------------------------------------------------------------------------
#     # Create a DAG manually for the second-order metareasoning problem (inner subtree)
#     # ----------------------------------------------------------------------------------------
#
#     # Root nodes
#     node_1 = Node(1, [], [], expression_type="contract", in_subtree=True)
#
#     # Create a list of the nodes in breadth-first order for the false branch
#     nodes_inner = [node_1]
#
#     # ----------------------------------------------------------------------------------------
#     # Create a DAG manually for the first-order metareasoning problem
#     # ----------------------------------------------------------------------------------------
#
#     # Leaf nodes
#     node_outer_3 = Node(3, [], [], expression_type="contract", in_subtree=False)
#
#     # Conditional Node
#     node_outer_2 = Node(2, [node_outer_3], [], expression_type="for", in_subtree=False)
#
#     # Root node
#     root_outer = Node(0, [node_outer_2, node_outer_3], [], expression_type="contract", in_subtree=False)
#
#     # Nodes
#     nodes_outer = [root_outer, node_outer_2, node_outer_3]
#
#     # Create and verify the DAG from the node list
#     dag_outer = DirectedAcyclicGraph(nodes_outer, root_outer)
#
#     # ----------------------------------------------------------------------------------------
#     # Create a program_dag with expanded subtrees for quality mapping generation
#     # ----------------------------------------------------------------------------------------
#
#     # Leaf node
#     node_3 = Node(3, [], [], expression_type="contract", in_subtree=False)
#
#     # Intermediate Nodes
#     # TODO: make a pointer from the for-node to a DAG that represents the internals of the loop
#     node_2 = Node(2, [node_3], [], expression_type="for", in_subtree=False)
#     node_1 = Node(1, [node_2], [], expression_type="contract", in_subtree=False)
#
#     # Root Node
#     node_root = Node(0, [node_1], [], expression_type="contract", in_subtree=False)
#
#     # Add the children
#     node_3.children = [node_2]
#     node_1.children = [node_root]
#
#
#     # For a list of nodes for the DAG creation
#     nodes = [node_root, node_1, node_2, node_3]
#
#     program_dag = DirectedAcyclicGraph(nodes, node_root)
#
#     # Used to create the synthetic data as instances and a populous file
#     generate = False
#     if not exists("populous.json") or generate:
#         # Initialize a generator
#         generator = Generator(INSTANCES, program_dag=program_dag, time_limit=TIME_LIMIT, time_step_size=TIME_STEP_SIZE,
#                               uniform_low=0.05,
#                               uniform_high=0.9)
#
#         # Let the root be trivial and not dependent on parents
#         # generator.trivial_root = True
#
#         # Adjust the DAG structure that has conditionals for generation
#         generator.generator_dag = generator.adjust_dag_with_conditionals(program_dag)
#
#         # Initialize the velocities for the quality mappings in a list
#         # Need to initialize it after adjusting program_dag
#         # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
#         # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
#         generator.manual_override = [10000, 0.1, 0.1, 0.1, 0.1, 0.1, 10000, "simple_conditional", 10000]
#
#         # Generate the nodes' quality mappings
#         nodes = generator.generate_nodes()  # Return a list of file names of the nodes
#
#         # populate the nodes' quality mappings into one populous file
#         generator.populate(nodes, "populous.json")
#
#     # Create the program with some budget
#
#     program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=dag_outer, child_programs=None, budget=BUDGET, scale=10 ** 6, decimals=3, quality_interval=QUALITY_INTERVAL,
#                                     time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_subtree=False, generator_dag=program_dag)
#
#     # Initialize the pointers of the nodes to the program it is in
#     initialize_node_pointers_current_program(program_outer)
#
#     # Add the subtree contract programs to the simple_conditional node
#     # Add the left subtree
#     true_subtree = DirectedAcyclicGraph(nodes_inner_true, root=node_inner_true_root)
#     # Convert to a contract program
#     node_outer_1.true_subprogram = ContractProgram(program_id=1, parent_program=program_outer, child_programs=None, program_dag=true_subtree, budget=0, scale=10 ** 6, decimals=3,
#                                                    quality_interval=QUALITY_INTERVAL,
#                                                    time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_subtree=True, generator_dag=program_dag)
#
#     # Initialize the pointers of the nodes to the program it is in
#     initialize_node_pointers_current_program(node_outer_1.true_subprogram)
#
#     # Add the right subtree
#     false_subtree = DirectedAcyclicGraph(nodes_inner_false, root=node_inner_false_root)
#     # Convert to a contract program
#     node_outer_1.false_subprogram = ContractProgram(program_id=2, parent_program=program_outer, child_programs=None, program_dag=false_subtree, budget=0, scale=10 ** 6, decimals=3,
#                                                     quality_interval=QUALITY_INTERVAL,
#                                                     time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_subtree=True, generator_dag=program_dag)
#
#     program_outer.child_programs = [node_outer_1.true_subprogram, node_outer_1.false_subprogram]
#
#     # Initialize the pointers of the nodes to the program it is in
#     initialize_node_pointers_current_program(node_outer_1.false_subprogram)
#
#     # Add the pointers from the parent program to the subprograms
#     node_outer_1.true_subprogram.parent_program = program_outer
#     node_outer_1.false_subprogram.parent_program = program_outer
#
#     # Add the pointers from the generator dag to the subprograms
#     node_outer_1.true_subprogram.generator_dag = program_dag
#     node_outer_1.false_subprogram.generator_dag = program_dag
#
#     # The input should be the outermost program
#     test = Test(program_outer)
#
#     # Print the tree for verification
#     # print(test.print_tree(program_dag.root))
#
#     # Test a random distribution on the initial allocations
#     # print(test.test_initial_allocations(iterations=500, initial_is_random=True, verbose=False))
#
#     # Test initial vs optimal expected utility and allocations
#     test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=True)
#     # test.find_utility_and_allocations(initial_allocation="uniform with noise", outer_program=program_outer, verbose=False)
#     # test.find_utility_and_allocations(initial_allocation="Dirichlet", outer_program=program_outer, verbose=False)
