class Node:
    """
    A node in our DAG that represents a contract algorithm or conditional.

    :param id: non-negative int, required
        A unique index appended to the node
    :param expression_type: string, required
        The type of the expression (e.g., functional, contract, or conditional)
    :param time: non-negative int
        The time allocation given to the contract algorithm in seconds (one-second intervals)
    :param parents: Node[], required for non-leaf nodes
        The parent nodes that are directed into the current node
    :param children: Node[], required for non-root nodes
        Used directly for contract conditionals
    """

    def __init__(self, id, parents, children, expression_type, in_subtree, time=None):
        # id of the node in the tree
        self.id = id
        self.parents = parents
        self.children = children
        self.expression_type = expression_type

        # subtree for the conditional expression
        self.subtree = None
        # true subtree for the conditional expression
        self.true_subtree = None
        # false subtree for the conditional expression
        self.false_subtree = None
        self.in_subtree = in_subtree
        # Used for the subtree that doesn't have access to parents
        self.parent_qualities = []

        self.time = time
        self.trivial = False
        # Used in checking for connectedness in the DAG
        self.traversed_connectedness = False
        self.traversed = False

    def local_joint_probability_distribution(self):
        """
        Queries the conditional performance profiles to create a joint probability distribution of the subtree.
        This will be used later in the expected utility function in conjunction with the utility function
        :return:
        """
        if self.time is None:
            raise ValueError("Node has no time allocation")
