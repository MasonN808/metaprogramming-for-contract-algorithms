import json


class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param file_name: the file name of the JSON file of performance profiles to be used
    """
    QUALITY_INTERVAL = 10  # The number of quality intervals to calculate the probability distribution given a
    # performance profile
    TIME_INTERVAL = 10  # The number of time intervals to calculate the probability distribution given a performance
    # profile

    def __init__(self, file_name):
        self.dictionary = self.import_performance_profiles(file_name)  # a dictionary of performance profiles

    @staticmethod
    def import_performance_profiles(file_name):
        """
        Imports the performance profiles as dictionary via an external JSON file.

        :return: an embedded dictionary
        """
        # JSON file
        f = open('{}'.format(file_name), "r")
        # Reading from file
        return json.loads(f.read())

    def query_quality_list(self, time, id):
        """
        Queries the performance profile at a specific time given the previous qualities of the contract algorithm's
        parents
        :param id: The node id
        :param time: the time allocation by which the contract algorithm stops
        :return: a list of qualities for node with self.id
        """
        if self.dictionary is None:
            raise ValueError("performance profile for node is null: Import performance profiles")
        else:
            # ["{}".format(id)]: The node id
            # ['0']: The node's performance profile
            # ["{}".format(time)]: The time allocation
            return self.dictionary["{}".format(id)]['0']["{}".format(time)]

    def query_probability(self, time, current_quality, previous_qualities):
        """
        Queries the performance profile at a specific time given the previous qualities of the contract algorithm's
        parents
        :param time: the time allocation by which the contract algorithm stops
        :param current_quality: the potential quality of the current contract algorithm
        :param previous_qualities: the qualities outputted from the parent nodes
        :return: [0,1], the probability of getting the current_quality, given the previous qualities and time allocation
        """
        # TODO: Finish this

        return self.dictionary[time]
