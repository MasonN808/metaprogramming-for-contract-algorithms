import math
import sys
from typing import List
import json
import numpy as np
import scipy.stats as st

sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes import utils  # noqa
from classes.node import Node  # noqa


class PerformanceProfile:
    """
    A performance profile is attached to a node in the DAG via an id associated with the node.

    :param: file_name: the file name of the JSON file of performance profiles to be used
    :param time_interval: the interval w.r.t. time to query from in the quality mapping
    :param time_limit: the time limit for each quality mapping
    :param time_step_size: the step size for each time step
    :param quality_interval: the interval w.r.t. qualities to query from in the quality mapping
    """

    def __init__(self, program_dag, full_dag, file_name, time_interval, time_limit, time_step_size,
                 quality_interval, expected_utility_type):
        self.program_dag = program_dag
        self.full_dag = full_dag
        # self.dictionary = self.import_quality_mappings(file_name)
        self.time_interval = time_interval
        self.time_limit = time_limit
        self.time_step_size = time_step_size
        self.quality_interval = quality_interval
        self.expected_utility_type = expected_utility_type

    def calculate_quality_sd(self, node):
        # TODO: test with sqrt and powers < 1. Possibly make it a function of the mean quality
        if node.time == 0:
            return 0
        else:
            return min(.2 * np.log(node.time + 1) * node.quality_sd, 0.2)

    # TODO: change name to calculate_expected_quality
    def query_expected_quality(self, node) -> float:
        return 1 - math.e ** (-node.c * node.time)

    # TODO: 1) Qualities over time should be monotonic increasing
    # 2) Qualities should never be negative --> make the sd a function of C?
    # Solution: Assume the starting quality is 0 for now. Pull from a normal distribution

    def query_quality(self, node) -> float:
        return self.query_expected_quality(node)
        #     # Generate noise from a guassian distribution with a standard deviation that is dependent on time
        #     # Need to also squeeze it to reduce from list to float
        #     # noise = np.random.normal(loc=0, scale=self.calculate_quality_sd(node), size=1).squeeze()

        #     # return self.query_expected_quality(node) + noise
        #     return self.query_expected_quality(node)

    def query_probability_contract_expression(self, quality, node) -> float:
        if node.time == 0:
            return 1
        else:
            # Calculate the z-score
            z_score = (quality - self.query_expected_quality(node)) / self.calculate_quality_sd(node)
            delta = .2
            probability = 1 - st.norm.cdf(z_score + delta) + st.norm.cdf(z_score - delta)
            return probability

    def query_probability_and_quality_from_conditional_expression(self, conditional_node) -> List[float]:
        """
        The performance profile (conditional expression): Queries the quality mapping at a specific time given the
        previous qualities of the contract algorithm's parents

        :param conditional_node: Node object, the conditional nodddde being evaluated
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time
        allocation
        """
        found_embedded_if = False
        # A list of the root qualities from the branches
        root_qualities = []
        # Query the probability of the condition (this is static for now)
        rho = self.estimate_rho()
        for child in conditional_node.children:
            # Take into account branched if statements
            if child.expression_type == "conditional":
                found_embedded_if = True

        if not found_embedded_if:
            # Create a list with the joint probability distribution of the conditional branch and the last quality of the branch
            # print(conditional_node.id)
            # print(conditional_node.true_subprogram)
            true_probability_quality = self.conditional_contract_program_probability_quality(conditional_node.true_subprogram)
            false_probability_quality = self.conditional_contract_program_probability_quality(conditional_node.false_subprogram)
            # print(true_probability_quality)
            # print(false_probability_quality)
            # exit()
            performance_profile_true = true_probability_quality[0]
            performance_profile_false = false_probability_quality[0]
            true_quality = true_probability_quality[1]
            false_quality = false_probability_quality[1]

            root_qualities.extend([true_quality, false_quality])

            conditional_quality = rho * true_quality + (1 - rho) * false_quality

            probability = (rho * performance_profile_true) + ((1 - rho) * performance_profile_false)

        else:
            # TODO: take into account embedded later
            raise ValueError("Found an embedded conditional")

        return [probability, conditional_quality]

    def conditional_contract_program_probability_quality(self, subprogram):
        # Note we are in the subprogram here
        # Assume for now that a contract_program is a conditional contract program
        probability = 1.0
        last_quality = None
        for node in subprogram.program_dag.nodes:
            # Skip the identifier node
            if node.expression_type != "conditional":
                # Get the parents' qualities given their time allocations
                parent_qualities = self.find_parent_qualities(utils.find_node_in_full_dag(node, full_dag=self.full_dag), depth=0)
                # print(parent_qualities)
                expected_quality = self.query_expected_quality(node)
                # Use the mean of the parent qualities to alter quality of current node
                if parent_qualities:
                    expected_quality *= np.mean(parent_qualities)
                # Keep only the average quality of the last node in the program
                if node.id == min([node.id for node in subprogram.program_dag.nodes]):
                    last_quality = expected_quality
                # Here we query the quality that is conditioned on the parent qualities TODO: double check with Justin or Samer
                probability *= self.query_probability_contract_expression(expected_quality, node)
        return [probability, last_quality]

    def query_probability_and_quality_from_for_expression(self, subprogram):
        probability = 1.0
        last_quality = None
        for node in subprogram.program_dag.nodes:
            # Skip the identifier node
            if node.expression_type != "for":
                # Get the parents' qualities given their time allocations
                parent_qualities = self.find_parent_qualities(utils.find_node_in_full_dag(node, full_dag=self.full_dag), depth=0)
                expected_quality = self.query_expected_quality(node)
                # Use the mean of the parent qualities to alter quality of current node
                if parent_qualities:
                    expected_quality *= np.mean(parent_qualities)
                # Keep only the average quality of the last node in the program
                if node.id == min([node.id for node in subprogram.program_dag.nodes]):
                    last_quality = expected_quality
                # Here we query the quality that is conditioned on the parent qualities TODO: double check with Justin or Samer
                probability *= self.query_probability_contract_expression(expected_quality, node)
        return [probability, last_quality]

    @staticmethod
    def estimate_rho() -> float:
        """
        Returns the probability of the condition in the conditional being true
        :return: float
        """
        # Assume it's constant for now
        return 0.01

    @staticmethod
    def calculate_tau() -> float:
        """
        Returns the amount of time to obtain rho
        :return: float
        """
        # Assume it takes constant time
        return 0.1

    def find_parent_qualities(self, node, depth) -> List[float]:
        """
        Returns the parent qualities given the time allocations and node

        :param: depth: The depth of the recursive call
        :param: node: Node object, finding the parent qualities of this node
        :param: time_allocations: float[] (order matters), for the entire DAG
        :return: A list of parent qualities
        """
        # Recur down the DAG
        depth += 1
        if node.parents:
            # Check if all the parents are conditional roots
            #TODO: make this more general
            if self.are_conditional_roots(node.parents):
                conditional_node = self.find_conditional_node(self.full_dag)
                parent_quality = self.query_probability_and_quality_from_conditional_expression(conditional_node)[1]
                # print(parent_quality)
                # exit()
                return [parent_quality]
            # Check if any of the parents are the last node of a for loop
            elif self.has_last_for_loop(node.parents):
                # print([node.id for node in self.program_dag.nodes])
                for_node = self.find_for_node(self.full_dag)
                parent_quality = self.query_probability_and_quality_from_for_expression(for_node.for_subprogram)[1]
                return parent_quality
            # Check that none of the parents are conditional expressions or for expressions
            elif not Node.is_conditional_node(node, "parents") and not Node.is_for_node(node, "parents"):
                parent_qualities = []
                for parent in node.parents:
                    quality = self.find_parent_qualities(parent, depth)
                    # Reset the parent qualities for the next node
                    parent_qualities.append(quality)
                if depth == 1:
                    return parent_qualities
                else:
                    # Return a list of parent-dependent qualities (not a leaf or root)
                    # quality = self.query_expected_quality(node) * np.mean(parent_qualities)
                    quality = self.query_expected_quality(node)
                    return quality
            elif Node.is_for_node(node, "parents"):
                # Assumption: Node only has one parent (the for)
                # Skip the for node since no relevant mapping exists
                node_for = node.parents[0]

                parent_qualities = []
                for parent in node_for.parents:
                    quality = self.find_parent_qualities(parent, depth)
                    # Reset the parent qualities for the next node_for
                    parent_qualities.append(quality)
                if depth == 1:
                    return parent_qualities
                else:
                    # Return a list of parent-dependent qualities (not a leaf or root)
                    # quality = self.query_expected_quality(node) * np.mean(parent_qualities)
                    quality = self.query_expected_quality(node)
                    return quality
            elif Node.is_conditional_node(node, "parents"):
                # Assumption: Node only has one parent (the conditional)
                # Skip the conditional node since no relevant mapping exists
                node_conditional = node.parents[0]
                parent_qualities = []
                for parent in node_conditional.parents:
                    quality = self.find_parent_qualities(parent, depth)
                    # print(quality)
                    # Reset the parent qualities for the next node_conditional
                    parent_qualities.append(quality)
                if depth == 1:
                    return parent_qualities
                else:
                    # Return a list of parent-dependent qualities (not a leaf or root)
                    quality = self.query_expected_quality(node)
                    # quality = self.query_expected_quality(node) * np.mean(parent_qualities)
                    return quality
        # Base Case (Leaf Nodes in a functional expression)
        else:
            # Leaf Node as a trivial functional expression
            if depth == 1 and (Node.is_conditional_node(node) or Node.is_for_node(node)):
                # exit()
                return [1]
            elif Node.is_conditional_node(node) or Node.is_for_node(node):
                return 1
            else:
                quality = self.query_expected_quality(node)
                if depth == 1:
                    return [quality]
                else:
                    return quality

    @staticmethod
    def round_nearest(number, step) -> float:
        """
        Finds the nearest element with respect to the step size

        :param number: A float
        :param step: A float
        :return: A float
        """
        return round(number / step) * step

    @staticmethod
    def find_number_of_decimals(number) -> int:
        """
        Finds the number of decimals in a float
        :param number: float
        :return: int
        """
        string_number = str(number)
        return string_number[::-1].find('.')

    @staticmethod
    def find_node(node_id, dag) -> Node:
        """
        Finds the node in the node list given the id

        :param: node_id: The id of the node
        :return Node object
        """
        # print([i.id for i in dag.nodes])
        # print(node_id)
        nodes = dag.nodes
        for node in nodes:
            if node.id == node_id:
                return node
        print([node.id for node in dag.nodes])
        raise IndexError("Node not found with given id --> {}".format(node_id))

    @staticmethod
    def find_conditional_node(dag) -> Node:
        nodes = dag.nodes
        for node in nodes:
            if node.expression_type == "conditional":
                return node
        raise IndexError("Conditional node not found in DAG")

    @staticmethod
    def find_for_node(dag) -> Node:
        nodes = dag.nodes
        for node in nodes:
            if node.expression_type == "for":
                return node
        raise IndexError("For node not found in DAG")

    @staticmethod
    def are_conditional_roots(nodes) -> bool:
        result = all(node.is_conditional_root is True for node in nodes)
        if result:
            return True
        else:
            return False

    @staticmethod
    def has_last_for_loop(nodes) -> bool:
        for node in nodes:
            if node.is_last_for_loop:
                return True
        return False

    def reset_traversed(self) -> None:
        """
        Resets the traversed pointers to Node objects

        :return: None
        """
        for node in self.program_dag.nodes:
            node.traversed = False
