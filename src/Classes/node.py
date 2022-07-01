class Node:
    """
    A node in our DAG that represents a contract algorithm with all inherited properties of a contract algorithm

    :param id: non-negative int, required
        A unique index appended to the node
    :param expr_type: string, optional
        The type of the expression (e.g., functional, contract, or conditional)
    :param time: non-negative int
        The time allocation given to the contract algorithm in seconds (one-second intervals)
    :param parents: Node[], required for non-leaf nodes
        The parent nodes that are directed into the current node
    :param performance_profile: dictionary, required
        the performance profile of a node in the DAG
    """

    def __init__(self, id, parents, performance_profile, expr_type=None, time=None):
        self.id = id  # id of the node in the tree
        self.traversed = False  # Used in checking for connectedness in the DAG
        self.expr_type = expr_type
        self.time = time
        self.parents = parents
        self.performance_profile = performance_profile  # This will be a an dictionary in the embedded dictionary of
        # performance profiles for the contract program, likely in a JSON file

    def adjust_time(self, time):
        self.time = time
