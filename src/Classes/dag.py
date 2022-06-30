class Dag:
    """
    Creates a DAG from a list of Node objects and a root node

    Parameters
    ----------
    root : Node, optional
        The root of the DAG

    node_list : Nodes[], required
        The list of nodes in the DAG, including the root
    """

    def __init__(self, root, node_list):
        if root is None:  # Check if root is unknown
            self.root = self.find_root()
        else:
            self.root = root
        self.node_list = node_list

    # def _check_structure(self):
    #     return None

    def find_root(self):
        possible_roots = []
        for node in self.node_list:
            if len(node.children) == 0:  # Check if node has no children, a property of the root
                possible_roots.append(node)
        if len(possible_roots) == 1:
            return possible_roots[0]  # Return the only possible root
        else:
            raise ValueError("More than one possible root found, restructure the DAG")  # More than one possible root
            # found; DAG must be restructured with one root
