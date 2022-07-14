class Test:
    def __init__(self, contract_program):
        self.contract_program = contract_program

    def test_initial_allocations(self, iterations, initial_is_random, verbose=False):
        """
        Tests different initial time allocations

        :param iterations: non-negative int
        :param initial_is_random:  bool
        :param verbose: bool
        :return:
        """
        expected_utilities = []
        for i in range(0, iterations):
            if initial_is_random:
                self.contract_program.allocations = self.contract_program.random_budget()
            else:
                self.contract_program.allocations = self.contract_program.uniform_budget()
            optimal_allocations = self.contract_program.naive_hill_climbing(verbose=verbose)
            optimal_time_allocations = [i.time for i in optimal_allocations]
            eu_optimal = self.contract_program.global_expected_utility(
                optimal_allocations) * self.contract_program.scale
            # Round the numbers
            if self.contract_program.decimals is not None:
                optimal_time_allocations = [round(i.time, self.contract_program.decimals)
                                            for i in self.contract_program.allocations]
                eu_optimal = round(eu_optimal, self.contract_program.decimals)
            expected_utilities.append(eu_optimal)
            if verbose:
                print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                      "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))
        return sorted(expected_utilities)

    def find_utility_and_allocations(self, allocation_type, initial_is_random, verbose=False):
        """
        Finds the expected utility and time allocations for an optimal expected utility or initial expected utility
        given the initial time allocations

        :param allocation_type:
        :param initial_is_random:
        :param verbose:
        :return:
        """
        if initial_is_random:
            # Optimal using Dirichlet Distribution
            self.contract_program.allocations = self.contract_program.random_budget()
        else:
            # Optimal using Uniform Distribution
            self.contract_program.allocations = self.contract_program.uniform_budget()
        if allocation_type == "optimal":
            # This is a list of TimeAllocation objects
            allocations = self.contract_program.naive_hill_climbing(verbose=verbose)
            optimal_time_allocations = [i.time for i in allocations]
            eu_optimal = self.contract_program.global_expected_utility(allocations) * self.contract_program.scale
            if self.contract_program.decimals is not None:
                optimal_time_allocations = [round(i.time, self.contract_program.decimals) for i in
                                            self.contract_program.allocations]
                eu_optimal = round(eu_optimal, self.contract_program.decimals)
            print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                  "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))
        elif allocation_type == "initial":
            # This is a list of TimeAllocation objects
            allocations = self.contract_program.allocations
            initial_time_allocations = [i.time for i in allocations]
            eu_initial = self.contract_program.global_expected_utility(
                self.contract_program.allocations) * self.contract_program.scale
            if self.contract_program.decimals is not None:
                initial_time_allocations = [round(i.time, self.contract_program.decimals)
                                            for i in self.contract_program.allocations]
                eu_initial = round(eu_initial, self.contract_program.decimals)
            # The initial time allocations for each contract algorithm
            print("                   Initial ==> Expected Utility: {:<5} ==> "
                  "Time Allocations: {}".format(eu_initial, initial_time_allocations))
        else:
            raise ValueError("Invalid allocation type: must be 'initial' or 'optimal'")
