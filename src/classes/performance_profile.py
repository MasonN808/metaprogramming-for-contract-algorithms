import json
import numpy as np

from src.classes.nodes.node import Node


class PerformanceProfile:
    """
    A performance profile is attached to a node in the DAG via an id associated with the node.

    :param: file_name: the file name of the JSON file of performance profiles to be used
    :param time_interval: the interval w.r.t. time to query from in the quality mapping
    :param time_limit: the time limit for each quality mapping
    :param time_step_size: the step size for each time step
    :param quality_interval: the interval w.r.t. qualities to query from in the quality mapping
    """

    def __init__(self, program_dag, file_name, time_interval, time_limit, time_step_size, quality_interval):
        self.program_dag = program_dag
        self.dictionary = self.import_quality_mappings(file_name)
        self.time_interval = time_interval
        self.time_limit = time_limit
        self.time_step_size = time_step_size
        self.quality_interval = quality_interval

    @staticmethod
    def import_quality_mappings(file_name) -> dict:
        """
        Imports the performance profiles as dictionary via an external JSON file

        :param: file_name: the name of the file with quality mappings for each node
        :return An embedded dictionary
        """
        f = open('{}'.format(file_name), "r")
        return json.loads(f.read())

    def query_quality_list_on_interval(self, time, id, parent_qualities) -> [float]:
        """
        Queries the quality mapping at a specific time, using some interval to create a distribution over qualities

        :param parent_qualities: List of qualities of the parent nodes
        :param id: The node id
        :param time: float, The time allocation by which the contract algorithm stops
        :return: A list of qualities for node with self.id
        """
        if self.dictionary is None:
            raise ValueError("The quality mapping for this node is null")

        else:
            # ["node_{}".format(id)]: The node
            # ['qualities']: The node's quality mappings
            dictionary = self.dictionary["node_{}".format(id)]['qualities']

            # Finding node quality given the parents' qualities
            if parent_qualities:
                for parent_quality in parent_qualities:
                    parent_quality = self.round_nearest(parent_quality, step=self.quality_interval)
                    dictionary = dictionary["{:.2f}".format(parent_quality)]

            qualities = []

            # Initialize the start and end of the time interval for descritization of the prior
            # Check if time is equal to limit
            if time == self.time_limit:
                start_step = ((time - self.time_interval) // self.time_interval) * self.time_interval
                end_step = start_step + self.time_interval
            else:
                start_step = (time // self.time_interval) * self.time_interval
                end_step = start_step + self.time_interval

            # Note: interval is [start_step, end_step) or [start_step, end_step] for time at limit
            num_decimals = self.find_number_of_decimals(self.time_step_size)

            # Round to get rid of rounding error in division of time
            for t in np.arange(start_step, end_step, self.time_step_size).round(num_decimals):
                # ["{}".format(t)]: The time allocation
                qualities += dictionary["{}".format(t)]

            return qualities

    def query_average_quality(self, id, time, parent_qualities) -> float:
        """
        Queries a single, estimated quality given a time allocation and possibly has parent qualities

        :param parent_qualities: float[] (order matters), the qualities of the parents given their time allocations
        :param id: non-negative int: the id of the Node object
        :param time: TimeAllocation object, The time allocation to the node
        :return: A quality
        """
        adjusted_id = id
        if PerformanceProfile.is_conditional_node(self.program_dag.nodes[id]):
            adjusted_id = id + 1

        if self.dictionary is None:
            raise ValueError("The quality mapping for this node is null")

        # For leaf nodes
        elif not parent_qualities:
            # ["node_{}".format(id)]: The node
            # ['qualities']: The node's quality mappings
            dictionary = self.dictionary["node_{}".format(adjusted_id)]['qualities']
            estimated_time = self.round_nearest(time.time, self.time_interval)

            # Use .1f to add a trailing zero
            qualities = dictionary["{:.1f}".format(estimated_time)]

            average_quality = self.average_quality(qualities)

            return average_quality

        # For intermediate or root nodes
        else:
            dictionary = self.dictionary["node_{}".format(adjusted_id)]['qualities']
            estimated_time = self.round_nearest(time.time, self.time_interval)

            for parent_quality in parent_qualities:
                parent_quality = self.round_nearest(parent_quality, step=self.quality_interval)
                dictionary = dictionary["{:.2f}".format(parent_quality)]

            qualities = dictionary["{:.1f}".format(estimated_time)]

            average_quality = self.average_quality(qualities)

            return average_quality

    @staticmethod
    def average_quality(qualities) -> float:
        """
        Gets the average quality over a list of qualities

        :param qualities: float[]
        :return: float
        """
        average = sum(qualities) / len(qualities)
        return average

    def query_probability_contract_expression(self, queried_quality, quality_list) -> float:
        """
        The performance profile (contract expression): Queries the quality mapping at a specific time given the
        previous qualities of the contract algorithm's parents

        :param quality_list: A list of qualities from query_quality_list_on_interval()
        :param queried_quality: The conditional probability of obtaining the queried quality
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time
        allocation
        """
        # Sort in ascending order
        quality_list = sorted(quality_list)

        number_in_interval = 0

        # Initialize the start and end of the quality interval for the posterior
        start_quality = (queried_quality // self.quality_interval) * self.quality_interval
        end_quality = start_quality + self.quality_interval

        # Note: interval of [start_step, end_step)
        for quality in quality_list:
            if start_quality <= quality <= end_quality:
                number_in_interval += 1

        probability = number_in_interval / len(quality_list)

        return probability

    # def query_probability_conditional_expression(self, conditional_node, queried_quality_branches, qualities_branches) -> float:
    #     """
    #     The performance profile (conditional expression): Queries the quality mapping at a specific time given the
    #     previous qualities of the contract algorithm's parents
    #
    #     :param conditional_node: Node object, the conditional node being evaluated
    #     :param queried_quality_branches: [float], A list of qualities from query_quality_list_on_interval() for the two branches
    #     :param qualities_branches: [float], A list of the queried qualities from the branches given their time allocations
    #     :return: [0,1], the probability of getting the current_quality, given the previous qualities and time
    #     allocation
    #     """
    #     # Sort in ascending order
    #     qualities_true_branch = sorted(qualities_branches[0])
    #     qualities_false_branch = sorted(qualities_branches[1])
    #
    #     found_embedded_if = False
    #
    #     # Query the probability of the condition being true
    #     rho = self.estimate_rho()
    #
    #     # TODO: implement recursion for embedded if statements
    #     # TODO: For now assume that only two branches exist that are contract expressions
    #     for child in conditional_node.children:
    #         # Take into account branched if statements
    #         if child.expression_type == "conditional":
    #             found_embedded_if = True
    #
    #     if not found_embedded_if:
    #         performance_profile_true = self.query_probability_contract_expression(queried_quality_branches[0], qualities_true_branch)
    #         performance_profile_false = self.query_probability_contract_expression(queried_quality_branches[1], qualities_false_branch)
    #
    #         probability = rho * performance_profile_true + (1 - rho) * performance_profile_false
    #
    #     else:
    #         # TODO: Finish this later (likely some recursion)
    #         raise ValueError("Found an embedded conditional")
    #     return probability

    def query_probability_conditional_expression(self, conditional_node) -> [float, [float]]:
        """
        The performance profile (conditional expression): Queries the quality mapping at a specific time given the
        previous qualities of the contract algorithm's parents

        :param conditional_node: Node object, the conditional node being evaluated
        :param queried_quality_branches: [float], A list of qualities from query_quality_list_on_interval() for the two branches
        :param qualities_branches: [float], A list of the queried qualities from the branches given their time allocations
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time
        allocation
        """
        # Sort in ascending order
        # qualities_true_branch = sorted(qualities_branches[0])
        # qualities_false_branch = sorted(qualities_branches[1])

        found_embedded_if = False

        # A list of the root qualities from the branches
        root_qualities = []

        # Query the probability of the condition being true offline
        rho = self.estimate_rho()

        for child in conditional_node.children:
            # Take into account branched if statements
            if child.expression_type == "conditional":
                found_embedded_if = True

        if not found_embedded_if:
            # Create a list with the joint probability distribution of the conditional branch and the last quality of the branch
            true_probability_quality = self.conditional_contract_program_probability_quality(conditional_node.true_subprogram)
            false_probability_quality = self.conditional_contract_program_probability_quality(conditional_node.false_subprogram)

            performance_profile_true = true_probability_quality[0]
            performance_profile_false = false_probability_quality[0]

            true_quality = true_probability_quality[1]
            false_quality = false_probability_quality[1]

            root_qualities.extend([true_quality, false_quality])

            probability = rho * performance_profile_true + (1 - rho) * performance_profile_false

        else:
            raise ValueError("Found an embedded conditional")
        return [probability, root_qualities]

    def conditional_contract_program_probability_quality(self, contract_program):
        # The for-loop is a breadth-first search given that the time-allocations is ordered correctly
        # Assume for now that a contract_program is a conditional contract program
        probability = 1.0
        last_quality = []

        time_allocations = contract_program.time_allocations
        for (id, time) in enumerate(time_allocations):
            node = self.find_node(id)

            if node.traversed:
                pass

            else:
                node.traversed = True

                if node.expression_type != "conditional":
                    # Get the parents' qualities given their time allocations
                    parent_qualities = self.find_parent_qualities(node, time_allocations, depth=0)

                    # Outputs a list of qualities from the instances at the specified time given a quality mapping
                    qualities = self.query_quality_list_on_interval(time.time, id, parent_qualities=parent_qualities)

                    # Calculates the average quality on the list of qualities for querying
                    average_quality = self.average_quality(qualities)

                    # Keep only the average quality of the last node in the program
                    if id == len(time_allocations) - 1:
                        last_quality.append(average_quality)

                    probability *= self.query_probability_contract_expression(average_quality, qualities)

        return [probability, last_quality]

    @staticmethod
    def estimate_rho() -> float:
        """
        Returns the probability of the condition in the conditional being true
        :return: float
        """
        # Assume it's constant for now
        return 0.4

    @staticmethod
    def calculate_tau() -> float:
        """
        Returns the amount of time to obtain rho
        :return: float
        """
        # Assume it takes constant time
        return 0.1

    def find_parent_qualities(self, node, time_allocations, depth) -> [float]:
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
            # Check that none of the parents are conditional expressions
            if not self.is_conditional_node(node, "parents"):

                parent_qualities = []

                for parent in node.parents:
                    quality = self.find_parent_qualities(parent, time_allocations, depth)
                    # Reset the parent qualities for the next node
                    parent_qualities.append(quality)

                if depth == 1:
                    return parent_qualities

                else:
                    # Return a list of parent-dependent qualities (not a leaf or root)
                    quality = self.query_average_quality(node.id, time_allocations[node.id], parent_qualities)

                    return quality
            else:
                # Assumption: Node only has one parent (the conditional)
                # Skip the conditional node since no relevant mapping exists
                node_conditional = node.parents[0]

                parent_qualities = []

                for parent in node_conditional.parents:
                    quality = self.find_parent_qualities(parent, time_allocations, depth)
                    # Reset the parent qualities for the next node_conditional
                    parent_qualities.append(quality)

                if depth == 1:
                    return parent_qualities

                else:
                    # Return a list of parent-dependent qualities (not a leaf or root)
                    quality = self.query_average_quality(node.id, time_allocations[node_conditional.id],
                                                         parent_qualities)

                    return quality

        # Base Case (Leaf Nodes in a functional expression)
        else:
            # Leaf Node as a trivial functional expression
            if depth == 1 or self.is_conditional_node(node):
                return []

            else:
                quality = self.query_average_quality(node.id, time_allocations[node.id], [])

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
    def is_conditional_node(node, family_type=None) -> bool:
        """
        Checks whether the parents or children are a conditional node

        :param node: Node object
        :param family_type: The "children" or "parents"
        :return: bool
        """
        if family_type is None:
            if node.expression_type == "conditional":
                return True
            return False
        if family_type == "parents":
            for parent in node.parents:
                if parent.expression_type == "conditional":
                    return True
            return False
        elif family_type == "children":
            for child in node.parents:
                if child.expression_type == "conditional":
                    return True
            return False
        else:
            raise ValueError("Invalid family_type")

    def find_node(self, node_id) -> Node:
        """
        Finds the node in the node list given the id

        :param: node_id: The id of the node
        :return Node object
        """
        for node in self.program_dag.nodes:
            if node.id == node_id:
                return node
        raise IndexError("Node not found with given id")
