from src.classes.directed_acyclic_graph import DirectedAcyclicGraph
from src.classes.node import Node
from src.classes.contract_program import ContractProgram
from src.profiles.generator import Generator
from os.path import exists
# import seaborn as sns

if __name__ == "__main__":
    BUDGET = 10
    INSTANCES = 5
    TIME_LIMIT = BUDGET
    STEP_SIZE = 0.1

    # Create a DAG manually for testing
    # Leaf nodes
    node_1 = Node(1, [], expression_type="contract")

    # Root node
    root = Node(0, [node_1], expression_type="contract")

    # Nodes
    nodes = [root, node_1]

    # Create and verify the DAG from the node list
    dag = DirectedAcyclicGraph(nodes, root)

    # Used to create the synthetic data as instances and a populous file
    if not exists("populous.json"):
        # Initialize a generator
        generator = Generator(INSTANCES, dag, time_limit=TIME_LIMIT,
                              step_size=STEP_SIZE, uniform_low=.05, uniform_high=.9)

        # Generate the nodes' quality mappings
        nodes = generator.generate_nodes()  # Return a list of file names of the nodes

        # populate the nodes' quality mappings into one populous file
        generator.populate(nodes, "populous.json")

    # Create the program with some budget
    program = ContractProgram(dag, BUDGET, scale=10, decimals=3)

    # Check the decimal count for rounding
    initial_time_allocations = [i.time for i in program.allocations]
    eu_initial = program.global_expected_utility(program.allocations) * program.scale
    if program.decimals is not None:
        initial_time_allocations = [round(i.time, program.decimals) for i in program.allocations]
        eu_initial = round(eu_initial, program.decimals)
    # The initial time allocations for each contract algorithm
    print("Initial Time Allocations: {}".format(initial_time_allocations))
    print("Initial Expected Utility: {}".format(eu_initial))

    # Check the decimal count for rounding
    # This is a list of TimeAllocation objects
    optimal_allocations = program.naive_hill_climbing()
    optimal_time_allocations = [i.time for i in program.naive_hill_climbing()]
    eu_optimal = program.global_expected_utility(optimal_allocations) * program.scale
    if program.decimals is not None:
        optimal_time_allocations = [round(i.time, program.decimals) for i in program.allocations]
        eu_optimal = round(eu_optimal, program.decimals)
    print("Naive Hill Climbing Search --> Time Allocations: {}".format(optimal_time_allocations))
    print("Naive Hill Climbing Search --> Expected Utility: {}".format(eu_optimal))

    # probability = 1
    # average_qualities = []
    # for (id, time) in enumerate(program.allocations):
    #     qualities = program.query_quality_list(time.time, id)
    #     average_quality = program.average_quality(qualities)
    #     average_qualities.append(average_quality)
    #     probability = probability * program.query_probability(time.time, id, average_quality)
    #
    # sns.distplot(subset['arr_delay'], hist=False, kde=True,
    #              kde_kws={'shade': True, 'linewidth': 3},
    #              label=airline)
