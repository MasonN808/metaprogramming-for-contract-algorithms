import math
import numpy as np


def create_dictionary(dag):
    """
    Creates a dictionary for one instance of the performance profiles of the DAG using synthetic data
    :param dag: The DAG
    :return: dictionary
    """
    dictionary = {}
    for i in dag.node_list:
        # Make an embedded dictionary for each node in the DAG
        # 0: represents the node's performance profile
        # 1: represents the list of pointers to the node's parents
        parent_ids = []
        for parent in i.parents:
            parent_ids.append(parent.id)
        dictionary_inner = {0: simulate_performance_profile(50, .1), 1: parent_ids}
        dictionary[i.id] = dictionary_inner
    return dictionary


def simulate_performance_profile(time_limit, step_size):
    """
    Simulates a performance profile of a contract algorithm using synthetic data
    :param time_limit: the time that the performance profile terminates at
    :param step_size: the step sizes of time
    :return: dictionary
    """
    dictionary = {}
    c = np.random.gamma(shape=2, scale=1)  # generate a random number from the gamma distribution
    for t in np.arange(0, time_limit, step_size):  # Using np.arange() for float step values
        dictionary[t] = 1 - math.e ** (-c * t)  # Use this function to approximate the performance profile
    return dictionary
