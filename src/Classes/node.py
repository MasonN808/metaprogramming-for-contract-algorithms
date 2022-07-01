class Node:
    """
    A node in our DAG that represents a contract algorithm with all inherited properties of a contract algorithm

    Parameters
    ----------
    id : non-negative int, required
        A unique index appended to the node

    expr_type : string, optional
        The type of the expression (e.g., functional, contract, or conditional)

    time : non-negative int
        The time allocation given to the contract algorithm in seconds (one-second intervals)

    parents : Node[], required for non-leaf nodes
        The parent nodes that are directed into the current node

    children : Node[], required for non-root node
        The children nodes that are directed from the current node

    pp : dictionary, required
        the performance profile of a node in the DAG
    """

    def __init__(self, id, parents, children, performance_profile, expr_type=None, time=None):
        self.id = id  # id of the node in the tree
        self.traversed = False  # Used in checking for connectedness in the DAG
        self.expr_type = expr_type
        self.time = time
        self.parents = parents
        self.children = children
        self.performance_profile = performance_profile  # This will be a an dictionary in the embedded dictionary of
        # performance profiles for the contract program

    def query_pp(self, time, current_quality, previous_qualities):
        """
        Queries the performance profile at a specific time given the previous qualities of the contract algorithms
        parents
        :param time: the time allocation by which the contract algorithm stops
        :param current_quality: the potential quality of the current contract algorithm
        :param previous_qualities: the qualities outputted from the parent nodes
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time allocation
        """

        return self.performance_profile[time]

    def adjust_time(self, time):
        self.time = time
