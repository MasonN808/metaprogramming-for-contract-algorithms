import copy
from itertools import permutations
from src.classes.performance_profile import PerformanceProfile
from src.classes.time_allocation import TimeAllocation


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

    def __init__(self, dag, budget, scale):
        PerformanceProfile.__init__(self, file_name=self.POPULOUS_FILE_NAME, time_interval=1, time_limit=budget,
                                    step_size=self.STEP_SIZE)
        self.budget = budget
        self.dag = dag
        self.allocations = self.__partition_budget()
        self.scale = scale

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
            qualities = self.query_quality_list(time.time, id)
            average_quality = self.average_quality(qualities)
            average_qualities.append(average_quality)
            probability = probability * self.query_probability(time.time, id, average_quality)

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
        allocation = self.budget / self.dag.order
        time_switched = allocation / 1.1
        while time_switched > .05:
            # print("_________________")
            possible_local_max = []

            for combination in permutations(self.allocations, 2):
                # Avoids exchanging time with itself
                if combination[0].node_id == combination[1].node_id:
                    continue
                # print("Combination: {}".format([i.node_id for i in combination]))

                # Make a deep copy to avoid pointers to the same list
                adjusted_allocations = copy.deepcopy(self.allocations)

                if adjusted_allocations[combination[0].node_id].time - time_switched < 0:
                    continue
                else:
                    adjusted_allocations[combination[0].node_id].time = adjusted_allocations[
                        combination[0].node_id].time - time_switched
                    adjusted_allocations[combination[1].node_id].time = adjusted_allocations[
                        combination[1].node_id].time + time_switched
                    if self.global_expected_utility(adjusted_allocations) > self.global_expected_utility(
                            self.allocations):
                        possible_local_max.append(adjusted_allocations)
                    print("Amount of time switched: {:<21} => EU(adjusted): {:<18}      EU(original): {}".format(time_switched,
                                                                                                                 self.global_expected_utility(
                                                                                                                     adjusted_allocations) * self.scale,
                                                                                                                 self.global_expected_utility(
                                                                                                                     self.allocations) * self.scale))

            # arg max here
            if possible_local_max:
                best_allocation = max([self.global_expected_utility(j) for j in possible_local_max])
                for j in possible_local_max:
                    if self.global_expected_utility(j) == best_allocation:
                        # Make a deep copy to avoid pointers to the same list
                        self.allocations = copy.deepcopy(j)
                # print("Allocations: {}\nEU: {}".format([i.time for i in self.allocations],
                #                                        self.global_expected_utility(self.allocations) * self.scale))
            else:
                time_switched = time_switched / 2
        return self.allocations

    def __partition_budget(self):
        """
        Discretizes the budget into equal partitions relative to the order of the DAG
        :return:
        """
        allocation = self.budget / self.dag.order  # Divide the budget into equal allocations for every contract algo
        return [TimeAllocation(allocation, node_id) for node_id in range(0, self.dag.order)]
