import json
import numpy as np


class PerformanceProfile:
    """
    A performance profile attached to a node in the DAG via an id associated with the node

    :param file_name: the file name of the JSON file of performance profiles to be used
    """

    def __init__(self, file_name, time_interval=10, time_limit=50, step_size=50, quality_interval=.05):
        self.dictionary = self.import_quality_mappings(file_name)
        self.time_interval = time_interval
        self.quality_interval = quality_interval
        self.time_limit = time_limit
        self.step_size = step_size

    @staticmethod
    def import_quality_mappings(file_name):
        """
        Imports the performance profiles as dictionary via an external JSON file.

        :return: An embedded dictionary
        """
        f = open('{}'.format(file_name), "r")
        return json.loads(f.read())

    def query_quality_list(self, time, id, parent_qualities):
        """
        Queries the quality mapping at a specific time, using some interval to create a distribution over qualities
        :param parent_qualities: List of qualities of the parent nodes
        :param id: The node id
        :param time: The time allocation by which the contract algorithm stops
        :return: A list of qualities for node with self.id
        """
        if self.dictionary is None:
            raise ValueError("The quality mapping for this node is null")
        else:
            # ["node_{}".format(id)]: The node
            # ['qualities']: The node's quality mappings
            dictionary = self.dictionary["node_{}".format(id)]['qualities']
            for parent_quality in parent_qualities:
                dictionary = dictionary[parent_quality]
            qualities = []
            # Initialize the start and end of the time interval for the prior
            start_step = (time // self.time_interval) * self.time_interval
            end_step = start_step + self.time_interval
            # Note: interval is [start_step, end_step)
            for t in np.arange(start_step, end_step, self.step_size).round(1):
                # ["{}".format(t)]: The time allocation
                qualities += self.dictionary["{}".format(t)]
            return qualities

    @staticmethod
    def average_quality(qualities):
        """
        Gets the average quality over a list of qualities
        :param qualities: float[]
        :return: float
        """
        average = sum(qualities) / len(qualities)
        return average

    def query_probability(self, time, id, queried_quality, parent_qualities):
        """
        The performance profile: Queries the quality mapping at a specific time given the previous qualities of the
        contract algorithm's parents
        :param parent_qualities: float[], the qualities of the parent nodes given their respective time allocations
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
