import json
import math
import numpy as np


class Generator:
    """
    A generator to create synthetic performance profiles

    :param instances: the number of instances to be produced
    :param dag: the dag to be used for performance profile simulation
    """

    def __init__(self, instances, dag, time_limit, step_size):
        self.instances = instances
        self.dag = dag
        self.time_limit = time_limit
        self.step_size = step_size

    def simulate_performance_profile(self):
        """
        Simulates a performance profile of a contract algorithm using synthetic data
        :return: dictionary
        """
        dictionary = {}
        c = np.random.gamma(shape=2, scale=1)  # generate a random number from the gamma distribution
        for t in np.arange(0, self.time_limit, self.step_size).round(1):  # Using np.arange() for float step values
            # round to one decimal place
            dictionary[t] = 1 - math.e ** (-c * t)  # Use this function to approximate the performance profile
        return dictionary

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

    def create_dictionary(self, dag):
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
            dictionary_inner = {0: self.simulate_performance_profile(), 1: parent_ids}
            dictionary[i.id] = dictionary_inner
        return dictionary

    def generate_instances(self):
        """
        Generates instances using the DAG and number of instances
        :return: a list of the file names of the instances stored in JSON files
        """
        instances = []  # file names of the instances
        for i in range(self.instances):  # Create a finite number of unique instances and create JSON files for each
            dictionary_temp = self.create_dictionary(self.dag)
            with open('instance_{}.json'.format(i), 'w') as f:
                instances.append('instance_{}.json'.format(i))  # Add the instance name for the populate() method
                json.dump(dictionary_temp, f, indent=2)
                print("New JSON file created for instance_{}".format(i))
        return instances

    def populate(self, instances, out_file):
        """
        Populates the performance profile using the average over a list of performance profiles from simulated instances

        :param instances: a list of file names (strings) of the JSON performance profiles to be merged
        :param out_file: the file to be populated
        :return: An embedded dictionary
        """
        with open('{}'.format(out_file), 'w') as f:
            bundle = self.import_performance_profiles(instances[0])  # Use the first instance as the dictionary to be
            # populated with the rest of the instances
            for node in bundle:  # Loop through all the nodes in the dictionary
                for t in bundle[node]['0']:  # Loop through all the time steps
                    bundle[node]['0'][t] = [bundle[node]['0'][t]]  # Make the singleton a list
            instances.remove(instances[0])  # remove the first instance from the list
            for instance in instances:
                temp_dictionary = self.import_performance_profiles(instance)  # Convert the JSON file into a dictionary
                for node in temp_dictionary:  # Loop through all the nodes in the dictionary
                    for t in bundle[node]['0']:  # Loop through all the time steps
                        bundle[node]['0'][t].append(temp_dictionary[node]['0'][t])  # Append the information from the
                        # performance profile to the bundled performance profile
            json.dump(bundle, f, indent=2)
        print("Finished populating JSON file using instances")
