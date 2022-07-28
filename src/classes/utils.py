
def print_allocations(allocations) -> None:
    """
    Prints the time allocations in a list of TimeAllocation objects

    :param allocations: TimeAllocations[]
    :return: None
    """
    print([i.time for i in allocations])


