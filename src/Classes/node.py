class Node:
    """
    A node in our DAG that represents a contract algorithm with all inherited properties of a contract algorithm

    Parameters
    ----------
    expr_type : string, required
        The type of the expression (e.g., functional, contract, or conditional)

    time : int, required
        The time allocation to the contract algorithm in seconds

    parents : Node[], required for non-leaf nodes
        The parent nodes that are directed into the current node

    children : Node[], required for non-root node
        The children nodes that are directed from the current node

    pp : dictionary, optional the performance profile of the
    """

    def __init__(self, expr_type, time, parents, children, pp):
        self.expr_type = expr_type
        self.time = time
        self.parents = parents
        self.children = children
        self.pp = pp

    def query_pp(self, time):
        return self.pp[time]

    def adjust_time(self, time):
        self.time = time
