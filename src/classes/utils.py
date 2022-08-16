from classes.nodes.node import Node
import sys

sys.path.append("/Users/masonnakamura/Local-Git/mca/src")


def print_allocations(allocations) -> None:
    """
    Prints the time allocations in a list of TimeAllocation objects

    :param allocations: TimeAllocations[]
    :return: None
    """
    print([i.time for i in allocations])


def find_node(node_id, dag) -> Node:
    """
    Finds the node in the node list given the id

    :param: node_id: The id of the node
    :return Node object
    """
    # print([i.id for i in dag.nodes])
    # print(node_id)
    nodes = dag.nodes
    for node in nodes:
        if node.id == node_id:
            return node
    raise IndexError("Node not found with given id")


def child_of_conditional(node) -> bool:
    """
    Checks whether the node is a child of a conditional

    :param: node: Node object
    :return bool
    """
    for parent in node.parents:
        if parent.expression_type == "conditional":
            return True
    return False


def parent_of_conditional(node) -> bool:
    """
    Checks whether the node is a parent of a conditional

    :param: node: Node object
    :return bool
    """
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
            return child


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
