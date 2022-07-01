import src.Classes.dag as dag
import src.Classes.node as node
import src.Classes.program as pgr
import src.Classes.performance_profile as pp

if __name__ == "__main__":
    BUDGET = 10

    # Create a DAG manually for testing
    # Leaf node
    performance_profile_3 = pp.PerformanceProfile(3)
    node_3 = node.Node(3, [], performance_profile_3)

    # Leaf node
    performance_profile_4 = pp.PerformanceProfile(4)
    node_4 = node.Node(4, [], performance_profile_4)

    # Leaf node
    performance_profile_5 = pp.PerformanceProfile(5)
    node_5 = node.Node(5, [], performance_profile_5)

    # Leaf node
    performance_profile_6 = pp.PerformanceProfile(6)
    node_6 = node.Node(6, [], performance_profile_6)

    # Leaf node
    performance_profile_1 = pp.PerformanceProfile(1)
    node_1 = node.Node(1, [node_3, node_4], performance_profile_1)

    # Intermediate node
    performance_profile_2 = pp.PerformanceProfile(2)
    node_2 = node.Node(2, [node_5, node_6], performance_profile_2)

    # Intermediate node
    performance_profile_0 = pp.PerformanceProfile(0)
    root = node.Node(0, [node_1, node_2], performance_profile_0)

    # Node list
    node_list = [root, node_1, node_2, node_3, node_4, node_5, node_6]

    # Create and verify the DAG from the node list
    dag = dag.Dag(node_list)

    # Create the program with some budget
    pgr.Program(dag, BUDGET)
