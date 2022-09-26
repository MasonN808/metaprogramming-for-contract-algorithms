class TimeAllocation:

    def __init__(self, node_id, time, list_time_allocations=None):
        if list_time_allocations is None:
            list_time_allocations = []
        self.node_id = node_id
        self.time = time
        self.list_time_allocations = list_time_allocations
