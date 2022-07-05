class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param id: non-negative int, required
        A unique index appended to the node
    """
    QUALITY_INTERVAL = 10  # The number of quality intervals to calculate the probability distribution given a
    # performance profile
    TIME_INTERVAL = 10  # The number of time intervals to calculate the probability distribution given a performance
    # profile

    def __init__(self, id):
        self.id = id
        self.performance_profile = None

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
