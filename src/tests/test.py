import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from time import sleep
from progress.bar import ChargingBar
from timeit import default_timer as timer

# from src.classes import utils
from src.classes import utils
from src.classes.nodes.node import Node


class Test:
    def __init__(self, contract_program):
        self.contract_program = contract_program

    def test_initial_allocations(self, iterations, initial_allocation, verbose=False):
        """
        Tests different initial time allocations

        :param iterations: non-negative int
        :param initial_allocation:  string
        :param verbose: bool
        :return:
        """
        with ChargingBar('Processing:', max=iterations, suffix='%(percent)d%%') as bar:
            expected_utilities = []
            for i in range(0, iterations):
                # Generate an initial allocation
                self.initialize_allocations(initial_allocation)
                optimal_allocations = self.contract_program.naive_hill_climbing_outer(verbose=verbose)
                optimal_time_allocations = [i.time for i in optimal_allocations]
                eu_optimal = self.contract_program.global_expected_utility(
                    optimal_allocations) * self.contract_program.scale
                # Round the numbers
                if self.contract_program.decimals is not None:
                    optimal_time_allocations = [round(i.time, self.contract_program.decimals)
                                                for i in self.contract_program.allocations]
                    eu_optimal = round(eu_optimal, self.contract_program.decimals)
                expected_utilities.append(eu_optimal)
                sleep(0.2)
                bar.next()
                if verbose:
                    print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                          "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))
        return sorted(expected_utilities)

    def find_utility_and_allocations(self, initial_allocation, outer_program, verbose=False) -> None:
        """
        Finds the expected utility and time allocations for an optimal expected utility or initial expected utility
        given the initial time allocations

        :param outer_program:
        :param initial_allocation: string, the type of initial allocation given for optimization
        :param verbose: bool, prints the optimization steps
        :return: None
        """
        start = timer()

        # Generate an initial allocation pointed to self.contract_program.allocations relative to the type of allocation
        self.initialize_allocations(initial_allocation=initial_allocation, contract_program=outer_program)

        eu_initial = self.contract_program.global_expected_utility(
            self.contract_program.allocations) * self.contract_program.scale

        if self.contract_program.decimals is not None:

            initial_time_allocations_outer = []
            initial_time_allocations_inner_true = []
            initial_time_allocations_inner_false = []

            for time_allocation in self.contract_program.allocations:

                # Check that it's not None
                if time_allocation.time is None:
                    continue

                initial_time_allocations_outer.append(round(time_allocation.time, self.contract_program.decimals))

            if self.contract_program.child_programs:
                for time_allocation in self.contract_program.child_programs[0].allocations:

                    # Check that it's not None
                    if time_allocation.time is None:
                        continue

                    initial_time_allocations_inner_true.append(round(time_allocation.time, self.contract_program.decimals))

                for time_allocation in self.contract_program.child_programs[1].allocations:

                    # Check that it's not None
                    if time_allocation.time is None:
                        continue

                    initial_time_allocations_inner_false.append(round(time_allocation.time, self.contract_program.decimals))

            eu_initial = round(eu_initial, self.contract_program.decimals)

        else:
            initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                              self.contract_program.allocations]
            initial_time_allocations_inner_true = [time_allocation.time for time_allocation in
                                                   self.contract_program.child_programs[0].allocations]
            initial_time_allocations_inner_false = [time_allocation.time for time_allocation in
                                                    self.contract_program.child_programs[1].allocations]

        print(" {} \n ----------------------".format(initial_allocation))
        # The initial time allocations for each contract algorithm
        print("                   Initial ==> Expected Utility: {:<5} ==> "
              "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

        print("{:<62}Time Allocations (inner-true): {}".format("", initial_time_allocations_inner_true))

        print("{:<62}Time Allocations (inner-false): {}".format("", initial_time_allocations_inner_false))

        # Should output a list of lists of optimal time allocations
        allocations = self.contract_program.naive_hill_climbing_outer(verbose=verbose)

        optimal_time_allocations_outer = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[0]])
        optimal_time_allocations_inner_true = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[1]])
        optimal_time_allocations_inner_false = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[2]])

        eu_optimal = self.contract_program.global_expected_utility(
            allocations[0], self.contract_program.original_allocations_conditional_branches) * self.contract_program.scale

        if self.contract_program.decimals is not None:
            optimal_time_allocations_outer = [round(time, self.contract_program.decimals) for
                                              time in optimal_time_allocations_outer]

            optimal_time_allocations_inner_true = [round(time, self.contract_program.decimals) for
                                                   time in optimal_time_allocations_inner_true]

            optimal_time_allocations_inner_false = [round(time, self.contract_program.decimals) for
                                                    time in optimal_time_allocations_inner_false]

            eu_optimal = round(eu_optimal, self.contract_program.decimals)

        # End the timer
        end = timer()

        print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
              "Time Allocations (outer): {}".format(eu_optimal, optimal_time_allocations_outer))

        print("{:<62}Time Allocations (inner-true): {}".format("", optimal_time_allocations_inner_true))

        print("{:<62}Time Allocations (inner-false): {} \n".format("", optimal_time_allocations_inner_false))

        print("{:<62}Execution Time (seconds): {}".format("", end - start))

    def print_tree(self, root, marker_str="+- ", level_markers=None):
        """
        From https://simonhessner.de/python-3-recursively-print-structured-tree-including-hierarchy-markers-using-depth-first-search/
        Prints a tree structure for debugging

        :param root: The root of the DAG
        :param marker_str: How each branch is printed
        :param level_markers: Depth markers
        :return:
        """
        if level_markers is None:
            level_markers = []
        empty_str = " " * len(marker_str)
        connection_str = "|" + empty_str[:-1]
        level = len(level_markers)

        def mapper(draw):
            if draw:
                return connection_str
            else:
                return empty_str

        markers = "".join(map(mapper, level_markers[:-1]))
        markers += marker_str if level > 0 else ""
        print(f"{markers}{root.id}")
        for i, parent in enumerate(root.parents):
            is_last = i == len(root.parents) - 1
            self.print_tree(parent, marker_str, [*level_markers, not is_last])

    @staticmethod
    def genetic_algorithm(program, dag):
        """
        Genetic Algorithm implementation

        :param program: ContractProgram object
        :param dag: DirectedAcyclicGraph object
        :return: None
        """
        # switch to genetic algorithm mode
        program.using_genetic_algorithm = True

        varbound = np.array([[0, 10]] * len(dag.nodes))

        model = ga(function=program.global_expected_utility_genetic, dimension=len(
            dag.nodes), variable_type='real', variable_boundaries=varbound)

        model.run()

    def initialize_allocations(self, initial_allocation, contract_program):
        if initial_allocation == "uniform":
            contract_program.allocations = contract_program.uniform_budget()

            # Find inner contract programs
            inner_contract_programs = self.find_inner_programs(contract_program)
            if inner_contract_programs:
                for inner_contract_program in inner_contract_programs:
                    # initialize the allocations to the inner contract programs with the time allocation of the outer conditonal node
                    inner_contract_program.budget = contract_program.allocations[
                        self.find_node_id_of_conditional(contract_program)].time

                    self.initialize_allocations(initial_allocation, inner_contract_program)

        elif initial_allocation == "Dirichlet":
            contract_program.allocations = contract_program.dirichlet_budget()
        elif initial_allocation == "uniform with noise":
            contract_program.allocations = contract_program.uniform_budget_with_noise()
        else:
            raise ValueError("Invalid initial allocation type")

    @staticmethod
    def find_inner_programs(outer_program):
        inner_programs = []

        for outer_node in outer_program.program_dag.nodes:
            if Node.is_conditional_node(outer_node) and not outer_node.in_subtree:
                # Append its subprograms to the list
                inner_programs.extend([outer_node.true_subprogram, outer_node.false_subprogram])

        return inner_programs

    @staticmethod
    def find_node_id_of_conditional(outer_program):
        for outer_node in outer_program.program_dag.nodes:
            if Node.is_conditional_node(outer_node):
                return outer_node.id
        raise Exception("Didn't find a conditional Node")
