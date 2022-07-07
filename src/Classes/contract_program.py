from src.Classes.performance_profile import PerformanceProfile


class ContractProgram(PerformanceProfile):
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
    STEP_SIZE = 0.1
    POPULOUS_FILE_NAME = "populous.json"

    def __init__(self, dag, budget):
        PerformanceProfile.__init__(self, file_name=self.POPULOUS_FILE_NAME, time_interval=1, time_limit=budget,
                                    step_size=self.STEP_SIZE)
        self.budget = budget
        self.dag = dag
        self.allocations = self.__partition_budget()

    @staticmethod
    def global_utility(qualities):
        """
        Gives a utility given the qualities of the parents of the current node

        :param qualities: Qualities[], required
                The qualities that were outputted for each contract algorithm in the DAG
        :return: float
        """
        return sum(qualities)

    def global_expected_utility(self, time_allocations):
        """
        Gives the expected utility of the contract program given the performance profiles of the nodes
        (i.e., the probability distribution of each contract program's conditional performance profile) and the
        global utility

        Assumption(s): 1) A time-allocation is given to each node in the contract program

        :param time_allocations: float[], required
                The time allocations for each contract algorithm
        :return: float
        """
        # TODO: make sure that node ids are non-negative integers (for our example, it is)
        probability = 1
        average_qualities = []
        for (id, time) in enumerate(time_allocations):
            qualities = self.query_quality_list(time, id)
            average_quality = self.average_quality(qualities)
            average_qualities.append(average_quality)
            probability = probability * self.query_probability(time, id, average_quality)
        expected_utility = probability * self.global_utility(average_qualities)
        return expected_utility

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

    def __partition_budget(self):
        """
        Discretizes the budget into equal partitions relative to the order of the DAG
        :return:
        """
        allocation = self.budget / self.dag.order  # Divide the budget into equal allocations for every contract algo
        return [allocation] * self.dag.order
