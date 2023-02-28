import sys
from typing import List

import numpy as np
sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.node import Node  # noqa


def print_allocations(allocations):
    print([i.time for i in allocations])


def find_node(node_id, dag) -> Node:
    nodes = dag.nodes
    for node in nodes:
        if node.id == node_id:
            return node
    raise IndexError("Node not found with given id -> {}".format(node_id))


def child_of_conditional(node) -> bool:
    for parent in node.parents:
        if parent.expression_type == "conditional":
            return True
    return False


def child_of_for(node) -> bool:
    for parent in node.parents:
        if parent.expression_type == "for":
            return True
    return False


def find_children_fors(node) -> List[Node]:
    """
    Finds the children of the node that are for-loops

    :param: node: Node object
    :return list of nodes that are for-loop nodes
    """
    for_nodes = []
    for parent in node.parents:
        if parent.expression_type == "for":
            for_nodes.append(parent)
    return for_nodes


def parent_of_conditional(node) -> bool:
    for child in node.children:
        if child.expression_type == "conditional":
            return True
    return False


def find_neighbor_branch(node) -> Node:
    """
    Finds the neighbor branch of the child node of a conditional node
    Assumption: the input node is the child of a conditional node

    :param node: Node object
    :return: Node object
    """
    conditional_node = node.parents[0]
    for child in conditional_node.children:
        if child != node:
            return


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


def remove_nones_time_allocations(allocations):
    return [time_allocation for time_allocation in allocations if time_allocation.time is not None]


def remove_nones_times(allocations):
    return [time for time in allocations if time is not None]


def remove_nones_list(list):
    return [element for element in list if element is not None]


def find_leaves_in_dag(program_dag):
    leaves = []
    for node in program_dag.nodes:
        if not node.parents:
            leaves.append(node)
    return leaves


def initialize_node_pointers_current_program(contract_program):
    for node in contract_program.program_dag.nodes:
        node.current_program = contract_program


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def flatten_list(nested_list):
    # https://stackabuse.com/python-how-to-flatten-list-of-lists/
    flat_list = []
    # Iterate through the outer list
    for element in nested_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
    # return reduce(lambda a,b:a+b, nested_list)


def dirichlet_ppv(iterations, dag, alpha=1, constant=10):
    # Create Dirichlet initial ppv
    accumulated_ppv = []
    number_conditionals_and_fors = number_of_fors_conditionals(dag)
    number_conditionals = number_conditionals_and_fors[0]
    number_fors = number_conditionals_and_fors[1]
    conditional_indices = find_conditional_indices(dag)
    for_indices = find_for_indices(dag)

    for _ in range(0, iterations):
        # Remove one of the branches and the conditional node before applying the Dirichlet distribution
        velocities_array = np.random.dirichlet(np.repeat(alpha, len(dag.nodes) - number_conditionals - number_fors), size=1).squeeze() * constant
        # Turn the numpy array into a list
        velocities_list = velocities_array.tolist()

        if number_conditionals > 0:
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
            for _ in range(0, len(conditional_indices) - 1):
                velocities_list.pop(conditional_indices[0] + 1)

        if number_fors > 0:
            # Create the sublist for the For
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


def dirichlet_growth_factor_generator(dag, alpha=1, lower_bound=0, upper_bound=10):
    # Apply the Dirichlet distribution to pull ranodm values for the growth factors
    # Then turn the numpy array into a list
    growth_factors = (np.random.dirichlet(np.repeat(alpha, len(dag.nodes)), size=1).squeeze() * (upper_bound - lower_bound) + lower_bound).tolist()
    return growth_factors

def uniform_growth_factor_generator(dag, lower_bound=0, upper_bound=10):
    # Apply the Dirichlet distribution to pull ranodm values for the growth factors
    # Then turn the numpy array into a list
    growth_factors = (np.random.uniform(lower_bound,upper_bound,len(dag.nodes))).tolist()
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


def safe_arange(start, stop, step):
    # For arange without the bad floating point accumulation
    return step * np.arange(start / step, stop / step)


def find_node_in_full_dag(node, full_dag):
    for full_node in full_dag.nodes:
        if node.id == full_node.id:
            return full_node