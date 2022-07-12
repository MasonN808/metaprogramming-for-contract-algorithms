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

    def query_quality_list_on_interval(self, time, id, parent_qualities):
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
            # Finding node quality given the parents' qualities
            if parent_qualities:
                for parent_quality in parent_qualities:
                    parent_quality = self.round_nearest(parent_quality, step=self.quality_interval)
                    dictionary = dictionary["{:.2f}".format(parent_quality)]
            qualities = []
            # Initialize the start and end of the time interval for descritization of the prior
            start_step = (time // self.time_interval) * self.time_interval
            end_step = start_step + self.time_interval
            # Note: interval is [start_step, end_step)
            # TODO: may need to fix the hardcoded round function
            for t in np.arange(start_step, end_step + self.step_size, self.step_size).round(1):
                # ["{}".format(t)]: The time allocation
                qualities += dictionary["{}".format(t)]
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

    def query_probability(self, queried_quality, quality_list):
        """
        The performance profile: Queries the quality mapping at a specific time given the previous qualities of the
        contract algorithm's parents

        :param quality_list: A list of qualities from query_quality_list_on_interval()
        :param queried_quality: The conditional probability of obtaining the queried quality
        :return: [0,1], the probability of getting the current_quality, given the previous qualities (not yet) and time
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
            if start_quality <= quality < end_quality:
                number_in_interval += 1
        probability = number_in_interval / len(quality_list)
        return probability

    def query_quality(self, id, time, parent_qualities):
        """
        Queries a single, estimated quality given a time allocation and possibly has parent qualities

        :param parent_qualities: float[] (order matters), the qualities of the parents given their time allocations
        :param id: non-negative int: the id of the Node object
        :param time: TimeAllocation object, The time allocation to the node
        :return: A quality
        """
        if self.dictionary is None:
            raise ValueError("The quality mapping for this node is null")
        # For leaf nodes
        elif not parent_qualities:
            # ["node_{}".format(id)]: The node
            # ['qualities']: The node's quality mappings
            dictionary = self.dictionary["node_{}".format(id)]['qualities']
            # Round the time to the respective
            estimated_time = self.round_nearest(time.time, self.time_interval)
            # Use .1f to add a trailing zero

            qualities = dictionary["{:.1f}".format(estimated_time)]
            average_quality = self.average_quality(qualities)
            return average_quality
        # For intermediate or root nodes
        else:
            dictionary = self.dictionary["node_{}".format(id)]['qualities']
            # Round the time to the respective
            estimated_time = self.round_nearest(time.time, self.time_interval)
            for parent_quality in parent_qualities:
                parent_quality = self.round_nearest(parent_quality, step=self.quality_interval)
                dictionary = dictionary["{:.2f}".format(parent_quality)]
            qualities = dictionary["{:.1f}".format(estimated_time)]
            average_quality = self.average_quality(qualities)
            return average_quality

    @staticmethod
    def round_nearest(number, step):
        """
        Finds the nearest element with respect to the step size

        :param number: A float
        :param step: A float
        :return: A float
        """
        return round(number / step) * step
