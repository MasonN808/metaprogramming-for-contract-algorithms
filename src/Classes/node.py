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
    """

    def __init__(self, id, parents, expr_type=None, time=None):
        self.id = id  # id of the node in the tree
        self.traversed = False  # Used in checking for connectedness in the DAG
        self.expr_type = expr_type
        self.time = time
        self.parents = parents

    def __check_time(self):
        """
        Checks that the time allocation is valid
        :return:
        """
        if not self.time >= 0 | self.time is None:
            raise ValueError("Time allocation must be positive")

    def allocate_time(self, time):
        """
        Adjusts the time of the contract algorithm
        :param time: non-negative int, required
        :return: None
        """
        self.time = time

    def local_joint_probability_distribution(self):
        """
        Queries the conditional performance profiles to create a joint probability distribution of the subtree.
        This will be used later in the expected utility function in conjunction with the utility function
        :return:
        """
        if self.time is None:
            raise ValueError("Node has no time allocation")
        # TODO: Finish this
