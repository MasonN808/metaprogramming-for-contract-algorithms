from random import random

import numpy as np

from src.classes import utils
from src.classes.time_allocation import TimeAllocation


class InitializeAllocations:
    """
    A class that holds all methods to initialize the time allocations of a contract program using various methods

    :param: budget : non-negative float
        The budget of the contract program represented as seconds
    :param: program_dag : DAG object
        The DAG that the contract program inherits
    :param: generator_dag : DAG object
        The DAG that unions all inner and outer DAGS
    :param: performance_profile : PerformanceProfile
        The performance_profile of the contract program
    :param: in_subtree : bool
        Determines whether the contract program is a child of another contract program
    """

    def __init__(self, budget, program_dag, generator_dag, performance_profile, in_subtree):
        self.budget = budget
        self.program_dag = program_dag
        self.generator_dag = generator_dag
        self.performance_profile = performance_profile
        self.in_subtree = in_subtree

    def uniform_budget(self) -> [TimeAllocation]:
        """
        Partitions the budget into equal partitions relative to the order of the DAG

        :return: TimeAllocation[]
        """
        # Initialize a list of time allocations of the entire dag that includes the subdags from the conditionals
        time_allocations = []
        for i in range(self.generator_dag.order):
            time_allocations.append(TimeAllocation(i, None))

        budget = float(self.budget)

        # Initialize a list to properly loop through the nodes given that the node ids are not sequenced
        list_of_ordered_ids = [node.id for node in self.program_dag.nodes]

        # Do an initial pass to find the conditionals to adjust the budget
        for node_id in list_of_ordered_ids:
            if utils.find_node(node_id, self.program_dag).expression_type == "conditional" and utils.find_node(node_id,
                                                                                                               self.program_dag).in_subtree:
                # Assume every conditional takes tau time
                tau = self.performance_profile.calculate_tau()
                # Subtract tau from the budget
                budget -= tau
                # Add the time allocation at a specified index
                time_allocations[node_id] = TimeAllocation(node_id, tau)

        # Do a second pass to add in the rest of the allocations wrt a uniform allocation
        for node_id in list_of_ordered_ids:

            # Continue since we already did the initial pass
            if utils.find_node(node_id, self.program_dag).expression_type == "conditional" and utils.find_node(node_id,
                                                                                                               self.program_dag).in_subtree:
                continue

            allocation = self.find_uniform_allocation(budget)
            time_allocations[node_id] = TimeAllocation(node_id, allocation)

        # print("DEBUG-ALLOCATIONS-{}".format(utils.print_allocations(time_allocations)))
        return time_allocations

    def dirichlet_budget(self) -> [TimeAllocation]:
        """
        Partitions the budget into random partitions such that they add to the budget using a Dirichlet distribution

        :return: TimeAllocation
        """
        number_of_conditionals = self.count_conditionals()

        # Remove the one of the branches and the conditional node before applying the Dirichlet distribution
        allocations_array = np.random.dirichlet(np.ones(self.program_dag.order - (2 * number_of_conditionals)),
                                                size=1).squeeze()

        allocations_list = allocations_array.tolist()

        # Multiply all elements by the budget and remove tau times if conditionals exist
        # TODO: Later make this a list, if multiple conditionals exist
        tau = self.performance_profile.calculate_tau()

        # Transform the list wrt the budget
        allocations_list = [time * (self.budget - (number_of_conditionals * tau)) for time in allocations_list]

        # Insert the conditional nodes into the list with tau time and
        # Search for conditional branches and append a neighbor since we removed it prior to using Dirichlet
        index = 0
        while index < len(allocations_list):
            # We insert a conditional branch and the conditional node since they were omitted before
            if utils.child_of_conditional(utils.find_node(index)):
                # Insert the neighbor branch with same time allocation
                allocations_list.insert(index, allocations_list[index])

                index += 1

                # Insert the conditional node with tau time allocation
                allocations_list.insert(index + 1, tau)
            index += 1

        return [TimeAllocation(node_id=id, time=time) for (id, time) in enumerate(allocations_list)]

    def uniform_budget_with_noise(self, perturbation_bound=.1, iterations=10) -> [TimeAllocation]:
        """
        Partitions the budget into a uniform distribution with added noise

        :return: TimeAllocation[]
        """
        time_allocations = self.uniform_budget()
        i = 0

        while i <= iterations:
            # Initialize a random number to be used as a perturbation
            random_number = random.uniform(0, perturbation_bound)

            # Get two random indexes from the list of time allocations
            random_index_0 = random.randint(0, self.program_dag.order - 1)
            random_index_1 = random.randint(0, self.program_dag.order - 1)

            # Do some checks to ensure the properties of conditional expressions are held
            # Avoid all exchanges that include the conditional node
            if utils.find_node(random_index_0).expression_type == "conditional" or utils.find_node(
                    random_index_1).expression_type == "conditional":
                continue

            # Avoids exchanging time between two branch nodes of a conditional
            elif utils.child_of_conditional(utils.find_node(random_index_0)) and utils.child_of_conditional(
                    utils.find_node(random_index_1)):
                continue

            # Avoids exchanging time with itself
            elif random_index_0 == random_index_1:
                continue

            elif time_allocations[random_index_0].time - random_number < 0:
                continue

            else:
                i += 1

                # Check if is child of conditional so that both children of the conditional are allocated same time
                if utils.child_of_conditional(utils.find_node(random_index_0)):
                    # find the neighbor node
                    neighbor = utils.find_neighbor_branch(utils.find_node(random_index_0))

                    # Adjust the allocation to the traversed node under the conditional
                    time_allocations[random_index_0].time -= random_number
                    # Adjust allocation to the neighbor in parallel
                    time_allocations[neighbor.id].time -= random_number

                    # Adjust allocation to then non-child of a conditional
                    time_allocations[random_index_1].time += random_number

                elif utils.child_of_conditional(utils.find_node(random_index_1)):
                    # find the neighbor node
                    neighbor = utils.find_neighbor_branch(utils.find_node(random_index_1))

                    # Adjust the allocation to the traversed node under the conditional
                    time_allocations[random_index_1].time += random_number
                    # Adjust allocation to the neighbor in parallel
                    time_allocations[neighbor.id].time += random_number

                    # Adjust allocation to then non-child of a conditional
                    time_allocations[random_index_0].time -= random_number

                else:
                    time_allocations[random_index_0].time -= random_number
                    time_allocations[random_index_1].time += random_number

        return time_allocations

    def find_uniform_allocation(self, budget) -> float:
        """
        Finds the allocation that can uniformly be distributed given the budget

        :param budget: float
        :return: uniformed allocation
        """
        number_of_conditionals = self.count_conditionals()
        # multiply by two since the branches get an equivalent time allocation

        allocation = budget / (self.program_dag.order - number_of_conditionals)
        return allocation

    def reset_traversed(self) -> None:
        """
        Resets the traversed pointers to Node objects

        :return: None
        """
        for node in self.program_dag.nodes:
            node.traversed = False

    def count_conditionals(self) -> int:
        """
        Counts the number of conditionals in the contract program

        :return: number of conditionals:
        """
        if self.in_subtree:
            number_of_conditionals = 0

            # Initialize a list to properly loop through the nodes given that the node ids are not sequenced
            list_of_ordered_ids = [node.id for node in self.program_dag.nodes]

            for node_id in list_of_ordered_ids:
                if utils.find_node(node_id, self.program_dag).expression_type == "conditional":
                    number_of_conditionals += 1
            return number_of_conditionals
        else:
            # Since the conditionals don't affect the outer metareasoning allocations
            return 0
