class Program:
    """
    Structures a directed-acyclic graph (DAG) as a contract program by applying a budget on a DAG of
    contract algorithms.  The edges are directed from the leaves to the root

    Parameters
    ----------
    budget : non-negative int, required
        The budget of the contract program represented as seconds

    dag : DAG, required
        The DAG that the contract program inherits
    """

    def __init__(self, dag, budget):
        self.budget = budget
        self.allocations = self.__discretize_budget()
        self.dag = dag

    def naive_hill_climbing(self):
        """
        Does naive hill climbing search by randomly replacing a set amount of time s between two different contract
        algorithms. If the expected value of the root node of the contract algorithm increases, we commit to the
        replacement; else, we divide s by 2 and repeat the above until s reaches some threshold epsilon by which we
        terminate

        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        # TODO: Finish this
        return

    def __discretize_budget(self):
        """
        Discretizes the budget into equal partitions relative to the order of the DAG
        :return:
        """
        allocation = self.budget / self.dag.order  # Divide the budget into equal allocations for every contract algo
        # TODO: Take into account for no nice division (i.e. Floats)
        return [allocation] * self.dag.order
