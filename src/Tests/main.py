from src.Classes.dag import Dag
from src.Classes.node import Node
from src.Classes.program import Program
from src.Classes.performance_profile import PerformanceProfile
from src.profiles.generator import Generator

if __name__ == "__main__":
    BUDGET = 10
    INSTANCES = 5

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
    generator = Generator(INSTANCES, dag)

    # Generate the instances
    instances = generator.generate_instances()  # Return a list of file names of the instances

    # populate the instances into one populous file
    populous_file_name = "populous.json"
    generator.populate(instances, populous_file_name)

    performance_profiles = PerformanceProfile(populous_file_name)
    print(performance_profiles.query_quality_list(20.0, 0))

    # Create the program with some budget
    program = Program(dag, BUDGET)

    # The initial time allocations for each contract algorithm
    print(program.allocations)
