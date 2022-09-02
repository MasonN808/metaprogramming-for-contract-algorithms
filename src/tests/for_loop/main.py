import copy
import sys
import numpy as np

sys.path.append("/Users/masonnakamura/Local-Git/mca/src")

from classes.directed_acyclic_graph import DirectedAcyclicGraph  # noqa
from classes.nodes.node import Node  # noqa
from classes.contract_program import ContractProgram  # noqa
from classes.generator import Generator  # noqa
from tests.test import Test  # noqa
from os.path import exists  # noqa


def initialize_node_pointers_current_program(contract_program):
    for node in contract_program.program_dag.nodes:
        node.current_program = contract_program


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
    TIME_INTERVAL = 0.1
    # The quality interval when querying for the performance profile
    QUALITY_INTERVAL = .05
    # For debugging
    VERBOSE = False
    NUMBER_OF_LOOPS = 5
    # For type of performance profile (exact or appproximate)
    EXPECTED_UTILITY_TYPE = "exact"
    # Initialize a list of all possible qualities
    POSSIBLE_QUALITIES = np.arange(0, 1 + QUALITY_INTERVAL, QUALITY_INTERVAL)

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the second-order metareasoning problem (inner subtree)
    # ----------------------------------------------------------------------------------------

    # Root Node
    node_inner_1 = Node(6, [], [], expression_type="for", in_subtree=True)
    root_inner = Node(1, [node_inner_1], [], expression_type="contract", in_subtree=True)
    root_inner.in_for = True

    # Create a list of the nodes in breadth-first order for the false branch
    nodes_inner = [root_inner, node_inner_1]

    # Create and verify the DAG from the node list
    dag_inner = DirectedAcyclicGraph(nodes_inner, root_inner)
    dag_inner.number_of_loops = NUMBER_OF_LOOPS

    # Rollout the for loop in a seperate DAG
    dag_inner_rolled_out = Generator.rollout_for_loops(dag_inner)

    for i in dag_inner_rolled_out.nodes:
        print("dag_inner_rolled_out: {}, {}".format(i.id, [j.id for j in i.parents]))

    # ----------------------------------------------------------------------------------------
    # Create a DAG manually for the first-order metareasoning problem
    # ----------------------------------------------------------------------------------------

    # Leaf nodes
    node_outer_2 = Node(7, [], [], expression_type="contract", in_subtree=False)

    # Conditional Node
    node_outer_1 = Node(6, [node_outer_2], [], expression_type="for", in_subtree=False)
    node_outer_1.num_loops = NUMBER_OF_LOOPS
    node_outer_1.for_dag = copy.deepcopy(dag_inner)
    root_inner.subprogram_parent_node = node_outer_1

    for node in dag_inner_rolled_out.nodes:
        node.subprogram_parent_node = node_outer_1

    # Root node
    root_outer = Node(0, [node_outer_1], [], expression_type="contract", in_subtree=False)

    # Append the children
    node_outer_2.children = [node_outer_1]
    node_outer_1.children = [root_outer]

    # Nodes
    nodes_outer = [root_outer, node_outer_1, node_outer_2]

    # Create and verify the DAG from the node list
    dag_outer = DirectedAcyclicGraph(nodes_outer, root_outer)

    # ----------------------------------------------------------------------------------------
    # Create a program_dag with expanded subtrees for quality mapping generation
    # ----------------------------------------------------------------------------------------

    # Leaf node
    node_2 = Node(2, [], [], expression_type="contract", in_subtree=False)

    # Intermediate Nodes
    node_1 = Node(1, [node_2], [], expression_type="for", in_subtree=False)
    node_1.num_loops = NUMBER_OF_LOOPS
    for_dag = copy.deepcopy(dag_inner)
    for_dag.nodes = for_dag.nodes[0:len(for_dag.nodes) - 1]
    node_1.for_dag = copy.deepcopy(for_dag)

    # Root Node
    node_root = Node(0, [node_1], [], expression_type="contract", in_subtree=False)

    # Append the children
    node_2.children = [node_1]
    node_1.children = [node_root]

    # For a list of nodes for the DAG creation
    nodes = [node_root, node_1, node_2]

    program_dag = DirectedAcyclicGraph(nodes, node_root)

    # Rollout the for loop in a seperate DAG
    program_dag = Generator.adjust_dag_structure_with_for_loops(program_dag)

    for i in program_dag.nodes:
        print("program_dag: {}, {}".format(i.id, [j.id for j in i.parents]))

    # ----------------------------------------------------------------------------------------
    # Generate the performance profiles
    # ----------------------------------------------------------------------------------------

    # Used to create the synthetic data as instances and a populous file
    generate = True
    if not exists("populous.json") or generate:
        # Initialize a generator
        generator = Generator(INSTANCES, program_dag=program_dag, time_limit=TIME_LIMIT, time_step_size=TIME_STEP_SIZE,
                              uniform_low=0.05,
                              uniform_high=0.9)

        # Let the root be trivial and not dependent on parents
        # generator.trivial_root = True

        # Adjust the DAG structure that has conditionals for generation
        generator.generator_dag = generator.adjust_dag_with_fors(program_dag)

        # Initialize the velocities for the quality mappings in a list
        # Need to initialize it after adjusting program_dag
        # A higher number x indicates a higher velocity in f(x)=1-e^{-x*t}
        # Note that the numbers can't be too small; otherwise the qualities converge to 0, giving a 0 utility
        generator.manual_override = [.1, .6, .5, .4, .3, .2, "for", .1]

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # ----------------------------------------------------------------------------------------
    # Initialize the contract programs
    # ----------------------------------------------------------------------------------------

    # Create the outer program with some budget
    # TODO FIX THIS (8/18)

    # print([i.id for i in program_dag.nodes])
    # for i in program_dag.nodes:
    #     if i.is_last_for_loop:
    #         print("last loop id: {}".format(i.id))
    #     if i.expression_type == "for":
    #         print("for id: {}".format(i.id))
    #     if i.id == 0:
    #         print("parents of 0: {}".format([j.id for j in i.parents]))

    program_outer = ContractProgram(program_id=0, parent_program=None, program_dag=dag_outer, child_programs=None, budget=BUDGET, scale=10 ** 6, decimals=3, quality_interval=QUALITY_INTERVAL,
                                    time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_subtree=False, generator_dag=program_dag, expected_utility_type=EXPECTED_UTILITY_TYPE,
                                    possible_qualities=POSSIBLE_QUALITIES)

    # Initialize the pointers of the nodes to the program it is in
    initialize_node_pointers_current_program(program_outer)

    # Convert to a contract program
    node_outer_1.for_subprogram = ContractProgram(program_id=1, parent_program=program_outer, child_programs=None, program_dag=dag_inner_rolled_out, budget=0, scale=10 ** 6, decimals=3,
                                                  quality_interval=QUALITY_INTERVAL, time_interval=TIME_INTERVAL, time_step_size=TIME_STEP_SIZE, in_subtree=True, generator_dag=program_dag,
                                                  expected_utility_type=EXPECTED_UTILITY_TYPE, possible_qualities=POSSIBLE_QUALITIES, number_of_loops=NUMBER_OF_LOOPS)

    # Initialize the pointers of the nodes to the program it is in
    initialize_node_pointers_current_program(node_outer_1.for_subprogram)

    program_outer.child_programs = [node_outer_1.for_subprogram]

    # Add the pointers from the parent program to the subprograms
    node_outer_1.for_subprogram.subprogram_expression_type = "for"
    node_outer_1.for_subprogram.parent_program = program_outer
    node_outer_1.for_subprogram.generator_dag = program_dag

    # The input should be the outermost program
    test = Test(program_outer)

    # Test initial vs optimal expected utility and allocations
    test.find_utility_and_allocations(initial_allocation="uniform", outer_program=program_outer, verbose=True)

    # TODO: For now work on calculating the exact EU of the module (8/25)
