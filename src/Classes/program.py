class Program:
    """Structures a directed-acyclic graph (DAG) as a contract program

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
