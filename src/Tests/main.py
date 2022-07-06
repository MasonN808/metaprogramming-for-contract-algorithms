from src.Classes.dag import Dag
from src.Classes.node import Node
from src.Classes.program import Program
from src.Classes.performance_profile import PerformanceProfile
from src.profiles.generator import Generator

if __name__ == "__main__":
    BUDGET = 10  # The budget of the contract program
    INSTANCES = 5  # The number of instances generated for the performance profiles
    TIME_LIMIT = BUDGET  # The time limit for the synthetic performance profiles
    STEP_SIZE = .1  # The time step sizes in the performance profiles
    # Create a DAG manually for testing
    # Leaf node
    node_3 = Node(3, [])

    # Leaf node
    node_4 = Node(4, [])

    # Leaf node
    node_5 = Node(5, [])

    # Leaf node
    node_6 = Node(6, [])

    # Leaf node
    node_1 = Node(1, [node_3, node_4])

    # Intermediate node
    node_2 = Node(2, [node_5, node_6])

    # Intermediate node
    root = Node(0, [node_1, node_2])

    # Node list
    node_list = [root, node_1, node_2, node_3, node_4, node_5, node_6]

    # Create and verify the DAG from the node list
    dag = Dag(node_list)

    # Initialize a generator
    generator = Generator(INSTANCES, dag, time_limit=TIME_LIMIT, step_size=STEP_SIZE)

    # Generate the nodes quality mappings
    nodes = generator.generate_nodes()  # Return a list of file names of the nodes
    print(nodes)
    # populate the nodes' quality mappings into one populous file
    populous_file_name = "populous.json"
    generator.populate(nodes, populous_file_name)

    # Initialize the performance profiles from the JSON file
    performance_profiles = PerformanceProfile(
        populous_file_name, time_interval=1, time_limit=TIME_LIMIT, step_size=STEP_SIZE)

    # Test the query method
    print(performance_profiles.query_quality_list(time=5.4, id=0))

    print(performance_profiles.query_probability(time=5.4, id=0, queried_quality=1))

    # Create the program with some budget
    program = Program(dag, BUDGET)

    # The initial time allocations for each contract algorithm
    # print(program.allocations)
