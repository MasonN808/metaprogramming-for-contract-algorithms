import json


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

    def __init__(self, node_list, root=None):
        self.node_list = node_list
        self.order = len(self.node_list)
        if root is None:  # Checks if root is unknown
            self.root = self.__find_root()
        else:
            self.root = root
            if self.__find_root() != self.root:  # Checks if the input root is valid
                raise ValueError("Inputted root is not a root")
        self.__unique_id("list")  # Checks that all nodes have a unique id
        self.check_structure()  # Checks that the structure of the DAG is valid
        # self.performance_profiles = self.__import_performance_profiles()  # import the performance profiles

    @staticmethod
    def __import_performance_profiles():
        """
        Imports the performance profiles via an external JSON file.
        The JSON will have the following embedded format:
        # TODO: make sure this is correct
            * List of contact algorithms/nodes in the DAG
                * List of possible discretized intervals for parent node 1
                    * ...
                        * List of possible discretized intervals for parent node n
                            * List of possible discretized time intervals
                                * List of possible qualities for current node

        :return: an embedded dictionary of instances and conditional performance profiles
        """
        # JSON file
        f = open("src/profiles/data.json", "r")
        # Reading from file
        return json.loads(f.read())

    def check_structure(self):
        """
        Checks the following properties in order:
            1. A unique root
            2. Connectedness
            3. No self-loops
            4. No directed cycles
        :return: Error; else, continue
        """
        # Check for a unique root
        # Done in constructor
        # Check for connectedness
        if self.__is_disconnected(self.root):
            raise ValueError("Provided DAG is invalid, has disconnectedness")
        self.__reset_traversed()  # Reset the traversed pointers for the nodes in node_list to False
        # Check for self-loops
        if self.__has_self_loops():
            raise ValueError("Provided DAG is invalid, has self-loops")
        # Check for directed cycles
        if self.__is_cyclic():
            raise ValueError("Provided DAG is invalid, has directed cycles")

    def __is_disconnected(self, node):
        """
        Checks for disconnectedness in the provided DAG by doing a DFS from the root and checks that all nodes have been
        traversed in the node_list

        :assumption: The unique root has been found and pointed to self.root
        :param: node: a Node object
        :return: True, if the DAG is disconnected; else False
        """
        node.traversed = True
        if len(node.parents) == 0:  # Hit a leaf node or trivial DAG with one root node
            if node == self.root:
                return False  # Return False since trivially connected
            return  # Pop the call stack
        else:
            for parent in node.parents:  # Do a recursive call on all the parents
                self.__is_disconnected(parent)
        for temp_node in self.node_list:
            if not temp_node.traversed:
                return True  # Found a node that wasn't traversed, so the DAG is disconnected
        return False

    def __reset_traversed(self):
        """
        Reset the traversed attribute/pointer in all the nodes to False for future traversals
        """
        for node in self.node_list:
            node.traversed = False

    def __has_self_loops(self):
        """
        Checks for self-loops in the provided DAG by doing a DFS from the root

        :assumption: The unique root has been found and pointed to self.root
                     and the DAG is connected
        :param: node: a Node object
        :return: True, if the DAG has self-loops; else False
        """
        for node in self.node_list:
            if node in node.parents:
                return True  # Has self-loops
        return False  # No self-loops

    def __is_cyclic_util(self, v, visited, rec_stack):
        """
        An adaptation from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/#:~:text=To%20detect%20cycle%2C
        %20check%20for,a%20cycle%20in%20the%20tree to detect a cycle in the given DAG

        :param v: The current node v being evaluated
        :param visited: The visited Nodes
        :param rec_stack: The recursion stack
        :return: True, if v has been visited; else False
        """
        # Mark current node as visited and add to recursion stack
        visited[v.id] = True
        rec_stack[v.id] = True

        # Recur for all parents if any parent is visited and in rec_stack then graph is cyclic
        for parent in v.parents:
            if not visited[parent.id]:
                if self.__is_cyclic_util(parent, visited, rec_stack):
                    return True
            elif rec_stack[parent.id]:
                return True

        # The node needs to be popped from recursion stack before function ends
        rec_stack[v.id] = False
        return False

    def __is_cyclic(self):
        """
        An adaptation from https://www.geeksforgeeks.org/detect-cycle-in-a-graph/#:~:text=To%20detect%20cycle%2C
        %20check%20for,a%20cycle%20in%20the%20tree to detect a cycle in the given DAG

        :Assumption: The DAG is connected
        :return: True, if the DAG has a cycle; else False
        """
        visited = [False] * (len(self.node_list) + 1)
        rec_stack = [False] * (len(self.node_list) + 1)
        for node in self.node_list:
            if not visited[node.id]:
                if self.__is_cyclic_util(node, visited, rec_stack):
                    return True
        return False

    def __find_root(self):
        """
        If the root isn't provided, an exhaustive search over the nodes is used
        :return: Node (root) or Error
        """
        possible_roots = []
        # This double for-loop checks to see if the node is a parent of any other nodes for potential roots
        for node in self.node_list:
            is_parent = False
            for inner_node in self.node_list:
                if node in inner_node.parents:  # Check if node is a parent of any nodes, a property not of the root
                    is_parent = True
            if not is_parent:
                possible_roots.append(node)
        if len(possible_roots) == 1:
            return possible_roots[0]  # Return the only possible root
        if len(possible_roots) > 1:
            # More than one possible root found; DAG must be restructured with one root
            raise ValueError("More than one possible root found: restructure the DAG")
        if len(possible_roots) == 0:  # This catches only some cycles
            raise ValueError("No possible roots found: a cycle is likely present, restructure the DAG")

    def add_node(self, node):
        """
        Adds a node to the node_list and verifies the structure of the DAG

        :param node: a Node object to be appended to the self.node_list
        :return: None
        """
        self.__unique_id("node", node)  # Checks that the node has a unique id relative to node_list
        self.node_list.append(node)
        self.check_structure()  # Checks that adding the node doesn't ruin the DAG's structure

    def __unique_id(self, data_type, data=None):
        """
        Checks to see if the given list or element either has all unique elements or if the element appended
        to the node_list is unique, where, if not unique, will infringe on other functions

        :param data_type: string ("list" or "node")
        :param data: Node object or None
        :return: True if valid; else, error
        """
        if data_type == "list":  # Check that all nodes in the list have unique ids
            for node in self.node_list:
                for inner_node in self.node_list:
                    if node.id == inner_node.id:  # Check that the ids are different
                        if node != inner_node:  # Check that the inner node and outer node are different
                            raise ValueError("The same id is applied to more than one node")
            return True
        elif data_type == "node":  # Check that the appended node has a unique id relative to the other nodes
            for node in self.node_list:
                if node.id == data.id:
                    raise ValueError("The same id is applied to more than one node")
            return True
        else:
            raise ValueError("Invalid data_type given")
