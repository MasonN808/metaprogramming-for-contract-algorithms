class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param id: non-negative int, required
        A unique index appended to the node
    """

    def __init__(self, id):
        self.id = id
        self.performance_profile = None

    def populate(self, instances):
        """
        Populates the performance profile using the average over a list of performance profiles from simulated instances
        # TODO: Get back to this to confirm validity
        :param instances: a list of embedded dictionaries stored as a JSON file
        :return: An embedded dictionary
        """
        # TODO: Finish this
        return self.performance_profile

    def query(self, time, current_quality, previous_qualities):
        """
        Queries the performance profile at a specific time given the previous qualities of the contract algorithm's
        parents
        :param time: the time allocation by which the contract algorithm stops
        :param current_quality: the potential quality of the current contract algorithm
        :param previous_qualities: the qualities outputted from the parent nodes
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time allocation
        """
        # TODO: Finish this
        return self.performance_profile[time]
