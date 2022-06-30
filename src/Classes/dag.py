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
        self.check_structure(self.root, self.node_list)

    def check_structure(self, root, node_list):
        """
        Checks the following properties in order:
            1. A unique root
            2. Connectedness
            3. No self-loops
            4. No directed cycles
        :return: Error; else, continue
        """
        # Check for a unique root
        possible_roots = []
        for node in self.node_list:
            if len(node.children) == 0:  # Check if node has no children, a property of the root
                possible_roots.append(node)
        if len(possible_roots) > 1:
            # More than one possible root found; DAG must be restructured with one root
            raise ValueError("More than one possible root found: restructure the DAG")
        if len(possible_roots) == 0:  # This catches only some cycles
            raise ValueError("No possible roots found: a cycle is likely present, restructure the DAG")
        # Check for connectedness
        # TODO: Finish this
        # Check for self-loops
        # TODO: Finish this
        # Check for directed cycles
        # TODO: Finish this

    def is_cyclic_util(self, v, visited, rec_stack):
        """
        An adaptation from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/#:~:text=To%20detect%20cycle%2C
        %20check%20for,a%20cycle%20in%20the%20tree to detect a cycle in the given DAG

        :param v: The current node v being evaluated
        :param visited: The visited Nodes
        :param rec_stack: The recursion stack
        :return: True, if v has been visited; else False
        """
        # Mark current node as visited and add to recursion stack
        visited[v.index] = True
        rec_stack[v.index] = True

        # Recur for all children if any child is visited and in recStack then graph is cyclic
        for child in v.children:
            if not visited[child.index]:
                if self.is_cyclic_util(child, visited, rec_stack):
                    return True
            elif rec_stack[child.index]:
                return True

        # The node needs to be popped from recursion stack before function ends
        rec_stack[v.index] = False
        return False

    # Returns true if graph is cyclic; else, false
    def is_cyclic(self):
        """
        An adaptation from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/#:~:text=To%20detect%20cycle%2C
        %20check%20for,a%20cycle%20in%20the%20tree to detect a cycle in the given DAG

        :Assumption: The DAG is connected
        :return: True, if the DAG has a cycle; else False
        """
        visited = [False] * (len(self.node_list) + 1)
        rec_stack = [False] * (len(self.node_list) + 1)
        for node in self.node_list:
            if not visited[node.index]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True
        return False

    def find_root(self):
        """
        If the root isn't provided, an exhaustive search over the nodes is used
        :return: Node (root) or Error
        """
        possible_roots = []
        for node in self.node_list:
            if len(node.children) == 0:  # Check if node has no children, a property of the root
                possible_roots.append(node)
        if len(possible_roots) == 1:
            return possible_roots[0]  # Return the only possible root
        else:
            raise ValueError("More than one possible root found, restructure the DAG")  # More than one possible root
            # found; DAG must be restructured with one root
