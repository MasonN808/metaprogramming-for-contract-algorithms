class Node:
    def __init__(self, expr_type, time, parents, children, pp):
        self.expr_type = expr_type
        self.time = time  # Integer seconds
        self.parents = parents
        self.children = children
        self.pp = pp

    def query_pp(self, time):
        return self.pp[time]
