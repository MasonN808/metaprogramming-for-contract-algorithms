import sys

import numpy as np
sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.node import Node  # noqa


def find_node(node_id, dag) -> Node:
    nodes = dag.nodes
    for node in nodes:
        if node.id == node_id:
            return node
    raise IndexError("Node not found with given id -> {}".format(node_id))


@staticmethod
def find_number_decimals(number):
    return len(str(number).split(".")[1])


@staticmethod
def has_conditional_roots_as_parents(node):
    num = 0
    for parent in node.parents:
        if parent.is_conditional_root:
            num += 1
    if num == 1 or num > 2:
        raise ValueError("Invalid root pointers from conditionals")
    elif num == 2:
        return True
    else:
        return False


def flatten(arr):
    flattened_list = []
    for sublist in arr:
        if isinstance(sublist, list):
            for item in sublist:
                flattened_list.append(item)
        else:
            flattened_list.append(sublist)
    return flattened_list


def remove_nones_list(list):
    return [element for element in list if element is not None]


def find_leaves_in_dag(program_dag):
    leaves = []
    for node in program_dag.nodes:
        if not node.parents:
            leaves.append(node)
    return leaves


def dirichlet_growth_factor_generator(dag, alpha=1, lower_bound=0, upper_bound=10):
    # Apply the Dirichlet distribution to pull ranodm values for the growth factors
    # Then turn the numpy array into a list
    growth_factors = (np.random.dirichlet(np.repeat(alpha, len(dag.nodes)), size=1).squeeze() * (upper_bound - lower_bound) + lower_bound).tolist()
    return growth_factors


def uniform_growth_factor_generator(dag, lower_bound=0, upper_bound=10):
    # Apply the Dirichlet distribution to pull ranodm values for the growth factors
    # Then turn the numpy array into a list
    growth_factors = (np.random.uniform(lower_bound, upper_bound, len(dag.nodes))).tolist()
    return growth_factors


def ppv_generator(node_id, dag, c_list, constant=1):
    # node_id is the index accounting for all nodes in the contract program including fors and conditionals
    # Make all parameters C to be constant except for node_id where we do INTERVAL intervals to generate ppvs
    accumulated_ppv = []
    number_conditionals_and_fors = number_of_fors_conditionals(dag)
    number_conditionals = number_conditionals_and_fors[0]
    number_fors = number_conditionals_and_fors[1]
    conditional_indices = find_conditional_indices(dag)
    for_indices = find_for_indices(dag)

    for custom_c in c_list:
        velocities_array = np.repeat(constant, len(dag.nodes) - number_conditionals - number_fors)
        # Turn the numpy array into a list
        velocities_list = velocities_array.tolist()

        # Do some manipulations on the node_id since for nodes and conditional nodes are excluded, thus indexes must be changed
        if node_id > 7:  # Conditional node
            node_id -= 1
        if node_id > 12:  # For node
            node_id -= 1

        # Insert the desired custom c param into the desired node
        velocities_list[node_id] = custom_c

        # Do some manipulations to add the sublists and strings
        # Create the sublist for conditional
        accumlated_velocities = []
        for index in range(0, len(conditional_indices) + 1):
            if index == len(conditional_indices):
                accumlated_velocities.append("conditional")
            else:
                accumlated_velocities.append(velocities_list[conditional_indices[index]])
        # Place the sublist in the list
        velocities_list[conditional_indices[0]] = accumlated_velocities
        # Remove the duplicates in outer list
        for i in range(0, len(conditional_indices) - 1):
            velocities_list.pop(conditional_indices[0] + 1)

        # Place the sublist in the list
        accumlated_velocities = []
        for index in range(0, len(for_indices) + 1):
            if index == len(for_indices):
                accumlated_velocities.append("for")
            else:
                accumlated_velocities.append(velocities_list[for_indices[index] - len(conditional_indices) + 1])

        # Place the sublist in the list
        velocities_list[for_indices[0] - len(conditional_indices)] = accumlated_velocities
        # Remove the duplicates in outer list
        for i in range(0, len(for_indices) - 1):
            velocities_list.pop(for_indices[0] - len(conditional_indices) + 1)

        accumulated_ppv.append(velocities_list)

    return accumulated_ppv


def number_of_fors_conditionals(dag):
    # Get the number of fors and conditionals in a dag object
    conditional_count = 0
    for_count = 0
    for node in dag.nodes:
        if (node.expression_type == "conditional"):
            conditional_count += 1
        elif (node.expression_type == "for"):
            for_count += 1
    return [conditional_count, for_count]


def find_conditional_indices(dag, include_meta=False):
    '''
    param: include_meta includes the node that determines the expression type (conditional or for)
    '''
    indices = []
    for node in dag.nodes:
        if (node.in_true or node.in_false):
            indices.append(node.id)
    if include_meta:
        for node in dag.nodes:
            if node.expression_type == "conditional":
                indices.append(node.id)
    return indices


def find_true_indices(dag, include_meta=False):
    '''
    param: include_meta includes the node that determines the expression type (conditional or for)
    '''
    indices = []
    for node in dag.nodes:
        if node.in_true:
            indices.append(node.id)
    if include_meta:
        for node in dag.nodes:
            if node.expression_type == "conditional":
                indices.append(node.id)
    return indices


def find_false_indices(dag, include_meta=False):
    '''
    param: include_meta includes the node that determines the expression type (conditional or for)
    '''
    indices = []
    for node in dag.nodes:
        if node.in_false:
            indices.append(node.id)
    if include_meta:
        for node in dag.nodes:
            if node.expression_type == "conditional":
                indices.append(node.id)
    return indices


def find_for_indices(dag, include_meta=False):
    '''
    param: include_meta includes the node that determines the expression type (conditional or for)
    '''
    indices = []
    for node in dag.nodes:
        if (node.in_for):
            indices.append(node.id)
    if include_meta:
        for node in dag.nodes:
            if node.expression_type == "for":
                indices.append(node.id)
    return indices


def find_non_meta_indicies(dag):
    '''
    Finds the indicies that are not meta/placeholder nodes (e.g., conditional nodes and for nodes that are not contract algorithms)
    '''
    indices = []
    for node in dag.nodes:
        if node.expression_type == "contract":
            indices.append(node.id)
    return indices


def find_node_in_full_dag(node, full_dag):
    for full_node in full_dag.nodes:
        if node.id == full_node.id:
            return full_node
