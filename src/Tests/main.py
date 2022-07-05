from src.Classes.dag import Dag
from src.Classes.node import Node
from src.Classes.program import Program
from src.Classes.performance_profile import PerformanceProfile
from src.profiles.simulate import create_dictionary
import json

if __name__ == "__main__":
    BUDGET = 10

    # Create a DAG manually for testing
    # Leaf node
    performance_profile_3 = PerformanceProfile(3)
    node_3 = Node(3, [], performance_profile_3)

    # Leaf node
    performance_profile_4 = PerformanceProfile(4)
    node_4 = Node(4, [], performance_profile_4)

    # Leaf node
    performance_profile_5 = PerformanceProfile(5)
    node_5 = Node(5, [], performance_profile_5)

    # Leaf node
    performance_profile_6 = PerformanceProfile(6)
    node_6 = Node(6, [], performance_profile_6)

    # Leaf node
    performance_profile_1 = PerformanceProfile(1)
    node_1 = Node(1, [node_3, node_4], performance_profile_1)

    # Intermediate node
    performance_profile_2 = PerformanceProfile(2)
    node_2 = Node(2, [node_5, node_6], performance_profile_2)

    # Intermediate node
    performance_profile_0 = PerformanceProfile(0)
    root = Node(0, [node_1, node_2], performance_profile_0)

    # Node list
    node_list = [root, node_1, node_2, node_3, node_4, node_5, node_6]

    # Create and verify the DAG from the node list
    dag = Dag(node_list)

    # Create the program with some budget
    program = Program(dag, BUDGET)

    # The initial time allocations for each contract algorithm
    print(program.allocations)

    dictionary_synthetic = create_dictionary(dag)
    with open('data.json', 'w') as f:
        json.dump(dictionary_synthetic, f, indent=2)
        print("New json file is created from data.json file")

    dag.import_performance_profiles("data.json")
