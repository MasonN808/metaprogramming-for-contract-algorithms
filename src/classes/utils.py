import sys
from typing import List

sys.path.append("/Users/masonnakamura/Local-Git/mca/src")

from classes.node import Node  # noqa


def print_allocations(allocations) -> None:
    print([i.time for i in allocations])


def find_node(node_id, dag) -> Node:
    nodes = dag.nodes
    for node in nodes:
        if node.id == node_id:
            return node
    raise IndexError("Node not found with given id")


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
