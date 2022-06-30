class Program:
    """
    Structures a directed-acyclic graph (DAG) as a contract program by applying a budget on a DAG of
    contract algorithms.  The edges are directed from the leaves to the root

    Parameters
    ----------
    budget : int, required
        The budget of the contract program represented as seconds

    dag : DAG, required
        The DAG that the contract program inherits
    """

    def __init__(self, dag, budget):
        self.budget = budget
        self.dag = dag

# if __name__ == "__main__":
