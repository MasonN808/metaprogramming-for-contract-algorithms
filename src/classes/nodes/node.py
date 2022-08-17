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

    def __init__(self, id, parents, children, expression_type, in_subtree, is_conditional_root=False, time=None):
        # id of the node in the tree
        self.id = id
        self.parents = parents
        self.children = children
        self.expression_type = expression_type

        # pointer to the contract program that it's in (This should be initialized after creating the contract program
        self.current_program = None

        # true subtree for the conditional expression
        self.true_subprogram = None
        # false subtree for the conditional expression
        self.false_subprogram = None
        self.in_subtree = in_subtree
        self.in_true = None
        self.in_false = None

        # subtree for the for loop
        self.for_dag = None
        self.in_for = None
        self.num_loops = 0

        # Used for the subtree that doesn't have access to parents
        self.parent_qualities = []
        self.is_conditional_root = is_conditional_root

        self.time = time

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

    @staticmethod
    def is_conditional_node(node, family_type=None) -> bool:
        """
        Checks whether the parents or children are a conditional node

        :param node: Node object
        :param family_type: The "children" or "parents"
        :return: bool
        """
        if family_type is None:
            if node.expression_type == "conditional":
                return True
            return False
        if family_type == "parents":
            for parent in node.parents:
                if parent.expression_type == "conditional":
                    return True
            return False
        elif family_type == "children":
            for child in node.parents:
                if child.expression_type == "conditional":
                    return True
            return False
        else:
            raise ValueError("Invalid family_type")
