def expected_utility_genetic(self, time_allocations):
    """
    Gives the expected utility of the contract program given the performance profiles of the nodes
    (i.e., the probability distribution of each contract program's conditional performance profile) and the
    global utility

    Assumption: A time-allocation is given to each node in the contract program

    :param time_allocations: float[], required
            The time allocations for each contract algorithm
    :return: float
    """
    epsilon = .01
    if (self.budget - epsilon) <= sum(time_allocations) <= self.budget:
        probability = 1
        average_qualities = []
        # The for loop should be a breadth-first search given that the time-allocations is ordered correctly
        for (id, time) in enumerate(time_allocations):
            # TODO: will have to change this somewhat to incorporate conditional expressions
            node = self.find_node(id)
            parent_qualities = self.find_parent_qualities(node, time_allocations, depth=0)
            if self.using_genetic_algorithm:
                qualities = self.query_quality_list_on_interval(time, id, parent_qualities=parent_qualities)
            else:
                qualities = self.query_quality_list_on_interval(time.time, id, parent_qualities=parent_qualities)
            average_quality = self.average_quality(qualities)
            average_qualities.append(average_quality)
            if not self.child_of_conditional(node):
                probability *= self.query_probability_contract_expression(average_quality, qualities)
            else:
                pass
        expected_utility = probability * self.utility(average_qualities)
        return -expected_utility
    else:
        return None
