import json
import math
import numpy as np


class Generator:
    """
    A generator to create synthetic performance profiles

    :param instances: the number of instances to be produced
    :param dag: the dag to be used for performance profile simulation
    """

    def __init__(self, instances, dag, time_limit, step_size, uniform_low, uniform_high):
        self.instances = instances
        self.dag = dag
        self.time_limit = time_limit
        self.step_size = step_size
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high

    def simulate_performance_profile(self, random_number):
        """
        Simulates a performance profile of a contract algorithm using synthetic data

        :param random_number: a random number from a uniform distribution with some noise
        :return: dictionary
        """
        dictionary = {}
        for t in np.arange(0, self.time_limit, self.step_size).round(1):  # Using np.arange() for float step values
            # round to one decimal place
            dictionary[t] = 1 - math.e ** (
                -random_number * t)  # Use this function to approximate the performance profile
        return dictionary

    @staticmethod
    def import_performance_profiles(file_name):
        """
        Imports the performance profiles as dictionary via an external JSON file.

        :return: an embedded dictionary
        """
        f = open('{}'.format(file_name), "r")
        return json.loads(f.read())

    def create_dictionary(self, node):
        """
        Creates a dictionary for one instance of the performance profiles of the DAG using synthetic data
        :param node: A node object
        :return: dictionary
        """
        dictionary = {'instances': {}}
        # Take a random value from a uniform distribution
        c = np.random.uniform(low=self.uniform_low, high=self.uniform_high)
        for i in range(self.instances):
            # Add some noise to the random value
            c = c + abs(np.random.normal(loc=0, scale=.05))  # loc is mean; scale is st. dev.
            # Make an embedded dictionary for each instance of the node in the DAG
            dictionary_inner = self.simulate_performance_profile(c)
            dictionary['instances']['instance_{}'.format(i)] = dictionary_inner
        dictionary['parents'] = [parent.id for parent in node.parents]
        return dictionary

    def generate_nodes(self):
        """
        Generates instances using the DAG and number of instances required
        :return: a list of the file names of the instances stored in JSON files
        """
        instances = []  # file names of the instances
        # Create a finite number of unique instances and create JSON files for each
        for (i, node) in enumerate(self.dag.nodes):
            dictionary_temp = self.create_dictionary(node)
            with open('node_{}.json'.format(i), 'w') as f:
                instances.append('node_{}.json'.format(i))
                json.dump(dictionary_temp, f, indent=2)
                print("New JSON file created for node_{}".format(i))
        return instances

    def populate(self, nodes, out_file):
        """
        Populates a single file with the data from the quality mappings of the node JSON files

        :param nodes: a list of file names (strings) of the JSON performance profiles to be merged
        :param out_file: the file to be populated
        :return: An embedded dictionary
        """
        with open('{}'.format(out_file), 'w') as f:
            bundle = {}
            for (i, node) in enumerate(nodes):
                bundle["node_{}".format(i)] = {}
                bundle["node_{}".format(i)]['qualities'] = {}
                bundle["node_{}".format(i)]['parents'] = {}
                # Convert the JSON file into a dictionary
                temp_dictionary = self.import_performance_profiles(node)
                for instance in temp_dictionary['instances']:
                    # Loop through all the time steps
                    for t in temp_dictionary['instances'][instance]:
                        try:
                            bundle["node_{}".format(i)]['qualities']["{}".format(t)]
                        except KeyError:
                            bundle["node_{}".format(i)]['qualities']["{}".format(t)] = []
                        bundle["node_{}".format(i)]['qualities']["{}".format(t)].append(
                            temp_dictionary['instances'][instance][t])
                bundle["node_{}".format(i)]['parents'] = temp_dictionary['parents']
            json.dump(bundle, f, indent=2)
        print("Finished populating JSON file using nodes JSON files")
