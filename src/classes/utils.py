def print_allocations(allocations) -> None:
    """
    Prints the time allocations in a list of TimeAllocation objects

    :param allocations: TimeAllocations[]
    :return: None
    """
    print([i.time for i in allocations])


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
