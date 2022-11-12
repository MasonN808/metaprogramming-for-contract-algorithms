import sys
from typing import List
import copy
import json
import math
import numpy as np

sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes import utils  # noqa
from classes.directed_acyclic_graph import DirectedAcyclicGraph  # noqa
from classes.node import Node  # noqa


class Generator:
    """
    A generator to create synthetic performance profiles

    :param instances: the number of instances to be produced
    :param generator_dag: the program_dag to be used for performance profile simulation
    """

    def __init__(self, instances, program_dag, time_limit, time_step_size, uniform_low, uniform_high, generator_dag=None, quality_interval=.05):
        self.instances = instances
        self.generator_dag = generator_dag
        self.program_dag = program_dag
        self.time_limit = time_limit
        self.time_step_size = time_step_size
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.quality_interval = quality_interval
        # Flatten all the embedded lists in manual_override
        self.manual_override_index = -1
        self.manual_override = None

    def simulate_performance_profile(self, random_number, node):
        """
        Simulates a performance profile of a contract algorithm using synthetic data

        :param node: Node object
        :param random_number: a random number from a uniform distribution with some noise
        :return: dictionary
        """
        dictionary = self.recur_build(0, node, [], {}, random_number)
        return dictionary

    def recur_build(self, depth, node, qualities, dictionary, random_number):
        """
        Used to recursively generate the synthetic quality mappings

        :param depth: The depth of the recursion
        :param node: Node object, Synthetic quality mapping for this particular node given its parents
        :param qualities: The parent qualities as inputs of the node
        :param dictionary: The dictionary being generated
        :param random_number: To produce noise in the quality mappings
        :return:
        """
        potential_parent_qualities = [format(i, '.2f') for i in np.arange(0, 1 + self.quality_interval, self.quality_interval).round(2)]

        if not node.parents:
            velocity = self.parent_dependent_transform(node, qualities, random_number)

            for t in np.arange(0, self.time_limit + self.time_step_size, self.time_step_size).round(utils.find_number_decimals(self.time_step_size)):
                # Use this function to approximate the performance profile
                dictionary[t] = 1 - math.e ** (-velocity * t)

        else:
            parents_length = len(node.parents)

            if utils.has_conditional_roots_as_parents(node):
                parents_length -= 1

            for quality in potential_parent_qualities:
                dictionary[quality] = {quality: {}}

                # Base Case
                if depth == parents_length - 1:
                    dictionary[quality] = {}

                    # To change the quality mapping with respect to the parent qualities
                    velocity = self.parent_dependent_transform(node, qualities, random_number)

                    for t in np.arange(0, self.time_limit + self.time_step_size, self.time_step_size).round(utils.find_number_decimals(self.time_step_size)):
                        # Use this function to approximate the performance profile
                        dictionary[quality][t] = 1 - math.e ** (-velocity * t)
                else:
                    qualities.append(quality)
                    self.recur_build(depth + 1, node, qualities, dictionary[quality], random_number)

        return dictionary

    def parent_dependent_transform(self, node, qualities, random_number):
        if self.manual_override and self.valid_manual_override():
            if not self.manual_override[node.id] is None:
                average_parent_quality = 1

                if qualities:
                    # Convert the list of strings to floats
                    qualities = [float(quality) for quality in qualities]

                    # Get the average parent quality (this may not be what we want)
                    average_parent_quality = sum(qualities) / len(node.parents)

                return self.manual_override[node.id] * average_parent_quality

            else:
                return random_number

        else:
            if qualities:
                # Convert the list of strings to floats
                qualities = [float(quality) for quality in qualities]

                # Get the average parent quality (this may not be what we want)
                average_parent_quality = sum(qualities) / len(node.parents)

                velocity = (10**average_parent_quality) - 1

                return velocity

            else:
                # If node has no parents (i.e., a leaf node)
                # Check to see if manual override is in place
                return random_number

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

        :param: manual_override: A float to manually adjust the quality mapping
        :param: node: A node object
        :return: dictionary
        """
        dictionary = {'instances': {}}
        # Take a random value from a uniform distribution; used for nodes without parents
        c = np.random.uniform(low=self.uniform_low, high=self.uniform_high)

        for i in range(self.instances):
            # Add some noise to the random value
            c = c + abs(np.random.normal(loc=0, scale=.05))  # loc is mean; scale is st. dev.

            # Make an embedded dictionary for each instance of the node in the DAG
            dictionary_inner = self.simulate_performance_profile(c, node)

            dictionary['instances']['instance_{}'.format(i)] = dictionary_inner

        dictionary['parents'] = [parent.id for parent in node.parents]

        return dictionary

    def generate_nodes(self) -> List[str]:
        """
        Generates instances using the DAG and number of instances required
        :param: manual_override: A list of floats to manually adjust the quality mappings
        :return: a list of the file names of the instances stored in JSON files
        """
        nodes = []  # file names of the nodes

        # Create a finite number of unique nodes and create JSON files for each
        i = 0
        for node in self.generator_dag.nodes:
            dictionary_temp = self.create_dictionary(node)

            # Compare the generator program_dag with the program program_dag to see if conditional is encountered
            # If so, go up an index since it's not present in the generator program_dag
            if Node.is_conditional_node(self.program_dag.nodes[i]) or Node.is_for_node(self.program_dag.nodes[i]):
                i += 1

            with open('node_{}.json'.format(i), 'w') as f:
                nodes.append('node_{}.json'.format(i))
                json.dump(dictionary_temp, f, indent=2)
                print("New JSON file created for node_{}".format(i))

            i += 1

        return nodes

    def valid_manual_override(self):
        if len(self.manual_override) != len(self.program_dag.nodes):
            raise ValueError("Manual override list must be same length as DAG")
        else:
            return True

    def activate_manual_override(self, performance_profile_velocities):
        # Flatten all the embedded lists in manual_override
        self.manual_override = utils.flatten_list(performance_profile_velocities)

    def populate(self, nodes, out_file):
        """
        Populates a single file with the data from the quality mappings of the node JSON files

        :param nodes: a list of file names (strings) of the JSON performance profiles to be merged
        :param out_file: the file to be populated
        :return: An embedded dictionary
        """
        with open('{}'.format(out_file), 'w') as f:
            bundle = {}
            i = 0
            j = 0

            for node in nodes:
                if Node.is_conditional_node(self.program_dag.nodes[i]) or Node.is_for_node(self.program_dag.nodes[i]):
                    i += 1

                bundle["node_{}".format(i)] = {}
                bundle["node_{}".format(i)]['qualities'] = {}
                bundle["node_{}".format(i)]['parents'] = {}

                # Convert the JSON file into a dictionary
                temp_dictionary = self.import_performance_profiles(node)

                for instance in temp_dictionary['instances']:
                    # Loop through all the time steps
                    recursion_dictionary = temp_dictionary['instances'][instance]
                    populate_dictionary = bundle["node_{}".format(i)]['qualities']

                    self.recur_traverse(0, self.generator_dag.nodes[j], [], recursion_dictionary, populate_dictionary)

                bundle["node_{}".format(i)]['parents'] = temp_dictionary['parents']
                j += 1
                i += 1

            json.dump(bundle, f, indent=2)

        print("Finished populating JSON file using nodes JSON files")

    def recur_traverse(self, depth, node, qualities, dictionary, populate_dictionary):
        """
        Used to recursively traverse the synthetic quality mappings to generate the populated JSON file of quality
        mappings

        :param populate_dictionary: Dictionary to be populated
        :param depth: The depth of the recursion
        :param node: Synthetic quality mapping for this particular node given its parents
        :param qualities: The parent qualities as inputs of the node
        :param dictionary: The dictionary being generated
        :return:
        """
        # Node without parents
        if not node.parents:
            for t in dictionary:
                try:
                    # See if a list object exists
                    if not isinstance(populate_dictionary["{}".format(t)], list):
                        populate_dictionary["{}".format(t)] = []
                except KeyError:
                    populate_dictionary["{}".format(t)] = []

                populate_dictionary["{}".format(t)].append(dictionary[t])

        # Node with parents
        else:
            parents_length = len(node.parents)

            if utils.has_conditional_roots_as_parents(node):
                parents_length -= 1

            # Loop through layer of the parent qualities
            for parent_quality in dictionary:
                try:
                    populate_dictionary[parent_quality]
                except KeyError:
                    populate_dictionary[parent_quality] = {}

                # Base Case
                if depth == parents_length - 1:
                    # To change the parent_quality mapping with respect to the parent qualities
                    for t in dictionary[parent_quality]:
                        try:
                            # See if a list object exists
                            if not isinstance(populate_dictionary[parent_quality]["{}".format(t)], list):
                                populate_dictionary[parent_quality]["{}".format(t)] = []
                        except KeyError:
                            populate_dictionary[parent_quality]["{}".format(t)] = []

                        populate_dictionary[parent_quality]["{}".format(t)].append(dictionary[parent_quality][t])
                else:
                    self.recur_traverse(depth + 1, node, qualities.append(parent_quality),
                                        dictionary[parent_quality], populate_dictionary[parent_quality])

        return populate_dictionary

    @staticmethod
    def adjust_dag_with_fors(dag) -> DirectedAcyclicGraph:
        """
        Changes the structure of the DAG by removing any for nodes and appending its parents to its children
        temporarily for generation. Note that the original structure of the DAG remains intact

        :param dag: directedAcyclicGraph Object, original version
        :return: directedAcyclicGraph Object, a trimmed version
        """
        dag = copy.deepcopy(dag)
        for node in dag.nodes:
            if node.expression_type == "for":
                # for parent in node.parents:
                #     parent.children.extend(node.children)
                #     if node in parent.children:
                #         parent.children.remove(node)
                for child in node.children:
                    child.parents.extend(node.parents)
                    child.parents.remove(node)

                for parent in node.parents:
                    parent.children.extend(node.children)
                    parent.children.remove(node)
                dag.nodes.remove(node)

            if node.is_last_for_loop:
                # Change the parent pointers of the children since it is not done manually in testing file
                for child in node.children:
                    child.parents = [node]
        return dag

    @staticmethod
    def adjust_dag_with_conditionals(dag) -> DirectedAcyclicGraph:
        """
        Changes the structure of the DAG by removing any conditional nodes and appending its parents to its children
        temporarily for generation. Note that the original structure of the DAG remains intact
        :param dag: directedAcyclicGraph Object, original version
        :return: directedAcyclicGraph Object, a trimmed version
        """
        dag = copy.deepcopy(dag)
        for node in dag.nodes:
            if node.expression_type == "conditional":
                # Append its parents to the children
                # Then remove the node from the parents and children
                # Then remove the node from the nodes list
                for child in node.children:
                    child.parents.extend(node.parents)
                    child.parents.remove(node)

                for parent in node.parents:
                    parent.children.extend(node.children)
                    parent.children.remove(node)

                dag.nodes.remove(node)

        return dag

    @staticmethod
    def adjust_dag_structure_with_for_loops(dag) -> DirectedAcyclicGraph:
        """
        This rolls out the for loop into a chain of contract programs

        :param dag: directedAcyclicGraph Object, original version
        :return: directedAcyclicGraph Object, a trimmed version
        """
        dag = copy.deepcopy(dag)
        added_index = 0
        largest_added_index = 0

        for node in dag.nodes:
            if node.expression_type == "for" and not node.traversed:
                node.traversed = True
                for_node = node
                root = None
                leaf = None
                for_node_old_children = for_node.children

                # Expand the for subtree into a chain
                # Edit the parents and children of each added node
                for i in range(for_node.num_loops):
                    # Make deep copy to avoid overlapped pointers
                    for_dag = copy.deepcopy(for_node.for_dag)

                    previous_root = root

                    # Get the root node and the leaf node of the subprogram
                    root = for_dag.nodes[0]
                    leaf = for_dag.nodes[len(for_dag.nodes) - 1]

                    # Check for first iteration
                    if i == 0:
                        # Make the parents of the for-node the parents of the first iteration
                        for_node.children = [leaf]
                        leaf.parents = [for_node]
                    # Check for last iteration
                    elif i == for_node.num_loops - 1:
                        leaf.parents = [node]
                        leaf.children = for_node_old_children
                        previous_root.children = [leaf]

                        for loop_node in for_dag.nodes:
                            loop_node.is_last_for_loop = True
                    else:
                        # Make the parent the root of the previous iteration
                        leaf.parents = [node]
                        previous_root.children = [leaf]

                    largest_added_index = len(for_dag.nodes) * (for_node.num_loops)
                    added_index = len(for_dag.nodes) * (for_node.num_loops - i) - for_node.id

                    # Reinstate the node ids when appending to the current dag
                    for node in for_dag.nodes:
                        node.id += added_index
                        node.traversed = True

                    # Add the nodes to the ndoe list
                    # Use slicing to extend a list at a specified index
                    dag.nodes[for_node.id:for_node.id] = for_dag.nodes

                    # Go to the end of the internals of the for loop and reinitialize the node pointer
                    node = root

            elif node.expression_type == "for" or not node.traversed:
                added_index = largest_added_index
                node.id += largest_added_index

        # Reset the traveresed pointers
        for node in dag.nodes:
            node.traversed = False

        # Adjust the order
        dag.order = len(dag.nodes)

        return dag

    @staticmethod
    def rollout_for_loops(dag) -> DirectedAcyclicGraph:
        """
        Rolls out the for loop given one iteration of the internal componnets of the for expression

        :param dag: directedAcyclicGraph Object, original version
        :return: directedAcyclicGraph Object, a rollout version
        """
        added_index = 0
        root = None
        leaf = None
        roll_out_dag = copy.deepcopy(dag)
        copied_dag = copy.deepcopy(dag)
        copied_dag.nodes = copied_dag.nodes[0:len(dag.nodes) - 1]

        for loop_node in roll_out_dag.nodes[0:len(roll_out_dag.nodes) - 1]:
            loop_node.is_last_for_loop = True

        # previous_root = copied_dag.nodes[0]
        previous_leaf = copied_dag.nodes[len(copied_dag.nodes) - 1]

        # Expand the for subtree into a chain
        # Edit the parents and children of each added node
        for i in range(dag.number_of_loops - 1):

            # Make deep copy to avoid overlapped pointers
            copied_dag = copy.deepcopy(copied_dag)

            if i != 0:
                previous_leaf = leaf

            # Get the root node and the leaf node of the subprogram
            root = copied_dag.nodes[0]
            leaf = copied_dag.nodes[len(copied_dag.nodes) - 1]

            previous_leaf.parents = [root]
            root.children = [previous_leaf]

            # Check for last iteration (i.e., the first loop)
            # subtracted two since it is offset from previous node indexes
            if i == copied_dag.number_of_loops - 2:
                root.first_loop = True
                previous_leaf.parents = [root]
                root.children = [previous_leaf]
            else:
                # Make the parent the root of the previous iteration
                previous_leaf.parents = [root]
                root.children = [previous_leaf]

            added_index = len(copied_dag.nodes) * (copied_dag.number_of_loops - i) - (copied_dag.number_of_loops - i) + 1

            # Reinstate the node ids when appending to the current copied_dag
            for node in copied_dag.nodes:
                node.id += added_index

            # Add the nodes to the ndoe list
            # Use slicing to extend a list at a specified index
            roll_out_dag.nodes[len(roll_out_dag.nodes) - 1:len(roll_out_dag.nodes) - 1] = copied_dag.nodes

            # Go to the end of the internals of the for loop and reinitialize the node pointer

        # Adjust the first node's parent pointer
        roll_out_dag.nodes[0].parents = [roll_out_dag.nodes[1]]
        roll_out_dag.nodes[len(roll_out_dag.nodes) - 1].children = [roll_out_dag.nodes[len(roll_out_dag.nodes) - 2]]

        # Adjust the order
        roll_out_dag.order = len(roll_out_dag.nodes)

        return roll_out_dag
