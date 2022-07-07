import json

import numpy as np


class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param file_name: the file name of the JSON file of performance profiles to be used
    """

    def __init__(self, file_name, time_interval=10, time_limit=50, step_size=50, quality_interval=.05):
        # a dictionary of performance profiles
        self.dictionary = self.import_performance_profiles(file_name)
        # The interval for the prior of the conditional probability
        self.time_interval = time_interval
        # The interval over the qualities to create a quality distribution
        self.quality_interval = quality_interval
        # The time limit of the performance profiles
        self.time_limit = time_limit
        # The step size of the time steps in the performance profiles
        self.step_size = step_size

    @staticmethod
    def import_performance_profiles(file_name):
        """
        Imports the performance profiles as dictionary via an external JSON file.

        :return: An embedded dictionary
        """
        # JSON file
        f = open('{}'.format(file_name), "r")
        # Reading from file
        return json.loads(f.read())

    def query_quality_list(self, time, id):
        """
        Queries the quality mapping at a specific time, using some interval to create a distribution over qualities
        :param id: The node id
        :param time: The time allocation by which the contract algorithm stops
        :return: A list of qualities for node with self.id
        """
        if self.dictionary is None:
            raise ValueError("Quality mappings for node is null: Import quality mappings")
        else:
            qualities = []
            # Initialize the start and end of the time interval for the prior
            start_step = (time // self.time_interval) * self.time_interval
            end_step = start_step + self.time_interval
            # Note: interval of [start_step, end_step)
            for t in np.arange(start_step, end_step, self.step_size).round(1):
                # ["node_{}".format(id)]: The node
                # ['qualities']: The node's quality mappings
                # ["{}".format(t)]: The time allocation
                qualities += self.dictionary["node_{}".format(id)]['qualities']["{}".format(t)]
            return qualities

    def query_probability(self, time, id, queried_quality):
        """
        The performance profile: Queries the quality mapping at a specific time given the previous qualities of the
        contract algorithm's parents
        :param id: The id of the node/contract algorithm being queried
        :param time: The time allocation by which the contract algorithm stops
        :param queried_quality: The conditional probability of obtaining the queried quality
        :return: [0,1], the probability of getting the current_quality, given the previous qualities (not yet) and time
        allocation
        """
        # Sort in ascending order
        quality_list = sorted(self.query_quality_list(time, id))
        number_in_interval = 0
        # Initialize the start and end of the quality interval for the posterior
        start_quality = (queried_quality // self.quality_interval) * self.quality_interval
        end_quality = start_quality + self.quality_interval
        # Note: interval of [start_step, end_step)
        for quality in quality_list:
            if start_quality <= quality < end_quality:
                number_in_interval += 1
        probability = number_in_interval / len(quality_list)
        return probability
