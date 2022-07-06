import json

import numpy as np


class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param file_name: the file name of the JSON file of performance profiles to be used
    """

    def __init__(self, file_name, time_interval=10, time_limit=50, step_size=50):
        self.dictionary = self.import_performance_profiles(file_name)  # a dictionary of performance profiles
        self.time_interval = time_interval  # To calculate the conditional performance profile
        self.time_limit = time_limit  # The time limit of the performance profiles
        self.step_size = step_size  # The step size of the time steps in the performance profiles

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
        Queries the performance profile at a specific time, using some interval to create a distribution over qualities
        :param id: The node id
        :param time: the time allocation by which the contract algorithm stops
        :return: a list of qualities for node with self.id
        """
        if self.dictionary is None:
            raise ValueError("performance profile for node is null: Import performance profiles")
        else:

            qualities = []
            start_step = (time // self.time_interval) * self.time_interval
            end_step = start_step + self.time_interval
            # Note: interval of [start_step, end_step)
            for i in np.arange(start_step, end_step, self.step_size).round(1):
                # ["{}".format(id)]: The node id
                # ['0']: The node's performance profile
                # ["{}".format(time)]: The time allocation
                qualities += self.dictionary["{}".format(id)]['0']["{}".format(i)]  # Concatenates
            return qualities

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
