import copy
import os
import pickle
import sys
import numpy as np
from time import sleep
from progress.bar import ChargingBar
from timeit import default_timer as timer


sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.initialize_allocations import InitializeAllocations  # noqa
from classes import utils  # noqa
from classes.node import Node  # noqa


class Test:
    def __init__(self, contract_program, ppv, node_indicies_list, plot_type=None, plot_nodes=None):
        self.contract_program = contract_program
        self.ppv = ppv
        self.node_indicies_list = node_indicies_list
        self.plot_type = plot_type
        self.num_plot_methods = 0
        self.plot_nodes = plot_nodes

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
                self.initial_allocation_setup(initial_allocation)
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

    def monitor_eu_on_rhc(self, initial_allocation, outer_program, verbose=False):
        # Setup the initial time allocations
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)
        return self.contract_program.naive_hill_climbing_outer(verbose=verbose, monitoring=True)

    # For arbitrary contract programs using a variety of solution methods
    def find_utility_and_allocations(self, initial_allocation, outer_program, verbose=False):
        number_conditionals_and_fors = utils.number_of_fors_conditionals(self.contract_program.generator_dag)
        number_conditionals = number_conditionals_and_fors[0]
        number_fors = number_conditionals_and_fors[1]
        # Data for plotting
        eu = []
        # To monitor times for specific nodes
        time = [[] for i in range(0, len(self.node_indicies_list))]  # Remove 2 (TODO: THIS IS HARD CODED) for the for node and conditional node that are not anytime algos
        start = timer()

        ##############################################################################################################################
        # PROPORTIONAL ALLOCATION
        ##############################################################################################################################

        # Betas for tangent
        betas = [1, .5, .1, 0]
        # betas = [10, 5, 4, 3, 2, 1, .8, .6, .5, .1, 0]
        # Add up the number of methods where 2 represents hill climbing and uniform allocation
        self.num_plot_methods += len(betas) + 2

        # Get the eu for the proportional allocation method
        for beta in betas:
            proportional_allocations = self.contract_program.proportional_allocation_tangent(beta)

            eu_proportional = self.contract_program.global_expected_utility(proportional_allocations[0], proportional_allocations[1]) * self.contract_program.scale
            eu.append(eu_proportional)

            # Flatten the proportional allocations
            # TODO: HARDCODED
            proportional_allocations_flattened = [proportional_allocations[0], proportional_allocations[1][0], proportional_allocations[1][1], proportional_allocations[1][2]]

            cleaned_allocations_list = []

            # TODO: This is redundant
            # Remove all none time allocations and append to list
            for index, allocations in enumerate(proportional_allocations_flattened):
                # Do some transformatioins and deletion depending on allocation to get allocations for plotting
                if index == 0:
                    # Find the meta for and conditional node indices if they exist
                    # Remove the conditional and for node allocations
                    if number_conditionals > 0:
                        meta_conditional_index = utils.find_true_indices(self.contract_program.generator_dag, include_meta=True)[-1]
                        allocations.pop(meta_conditional_index)
                    if number_fors > 0:
                        meta_for_index = utils.find_for_indices(self.contract_program.generator_dag, include_meta=True)[-1]
                        allocations.pop(meta_for_index)

                # Remove nones
                cleaned_allocations = utils.remove_nones_time_allocations(allocations)

                if number_conditionals > 0:
                    if index >= 1 and index <= 2:
                        # Remove the last part of the true branch
                        cleaned_allocations.pop(len(cleaned_allocations) - 1)  # This is the tax
                if number_fors > 0:
                    if index == 3:
                        # Remove the last part of the true branch
                        cleaned_allocations.pop(len(cleaned_allocations) - 1)  # This is the 0 allocation

                cleaned_allocations_list.append(cleaned_allocations)

            # Flatten all the allocations
            flattened_allocations_list = utils.flatten_list(cleaned_allocations_list)

            # Sort the flattened list in ascending order wrt to the node id
            sorted_allocations_list = sorted(flattened_allocations_list, key=lambda time_allocation: time_allocation.node_id, reverse=True)

            for index in range(0, len(self.node_indicies_list)):
                time[index].append(sorted_allocations_list[index].time)

        if self.contract_program.decimals is not None:
            print("PPV ==> ", *self.ppv)
            print("         Proportional (inverse Tangent) ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(round(eu_proportional, self.contract_program.decimals), [round(time, self.contract_program.decimals) for time in utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[0]])]))
            if number_conditionals > 0:  # TODO: HARDCODED
                print("{:<62}Time Allocations (inner-true): {}".format("", [round(time, self.contract_program.decimals) for time in utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[1]])]))
                print("{:<62}Time Allocations (inner-false): {}".format("", [round(time, self.contract_program.decimals) for time in utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[2]])]))
            if number_fors > 0:  # TODO: HARDCODED
                print("{:<62}Time Allocations (inner-for): {}".format("", [round(time, self.contract_program.decimals) for time in utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[3]])]))
        else:
            print("         Proportional (inverse Tangent) ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_proportional, utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations[0]])))
            if number_conditionals > 0:  # TODO: HARDCODED
                print("{:<62}Time Allocations (inner-true): {}".format("", utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[1]])))
                print("{:<62}Time Allocations (inner-false): {}".format("", utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[2]])))
            if number_fors > 0:  # TODO: HARDCODED
                print("{:<62}Time Allocations (inner-for): {}".format("", utils.remove_nones_times([time_allocation.time for time_allocation in proportional_allocations_flattened[3]])))

        ##############################################################################################################################
        # UNIFORM ALLOCATION
        ##############################################################################################################################
        # TODO: GENERALIZE THIS (2/10)
        # Generate an initial allocation pointed to self.contract_program.allocations relative to the type of allocation
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)

        eu_initial = self.contract_program.global_expected_utility(self.contract_program.allocations) * self.contract_program.scale
        eu.append(eu_initial)

        # Take the outer allocations and declare any expression types with None allocations
        copy_uniform_allocations = copy.deepcopy(self.contract_program.allocations)
        # uniform_allocations_list = [copy_uniform_allocations, self.contract_program.child_programs[0].allocations[:-1],
        #                             self.contract_program.child_programs[1].allocations[:-1], self.contract_program.child_programs[2].allocations[:-1]]

        # Iteratively append the suballocations in order of expression type
        uniform_allocations_list = [copy_uniform_allocations]
        if self.contract_program.child_programs:
            for child_program in self.contract_program.child_programs:
                uniform_allocations_list.append(child_program.allocations)

        # Remove all none time allocations and append to list
        cleaned_allocations_list = []
        for index, allocations in enumerate(uniform_allocations_list):
            # Remove nones
            # Also remove the last time allocations of the different expressions
            cleaned_allocations = utils.remove_nones_time_allocations(allocations)
            if index != 0:
                # Truncate the last element for all non functional expressions
                cleaned_allocations = cleaned_allocations[:-1]
            cleaned_allocations_list.append(cleaned_allocations)

        # Flatten all the allocations
        flattened_allocations_list = utils.flatten_list(cleaned_allocations_list)
        # Sort the flattened list in ascending order
        sorted_allocations_list = sorted(flattened_allocations_list, key=lambda time_allocation: time_allocation.node_id, reverse=False)

        for index in range(0, len(self.node_indicies_list)):
            time[index].append(sorted_allocations_list[index].time)

        if self.contract_program.decimals is not None:
            initial_time_allocations_outer = []
            initial_time_allocations_inner_true = []
            initial_time_allocations_inner_false = []
            initial_time_allocations_inner_for = []
            initial_time_allocations_inner = [initial_time_allocations_inner_true, initial_time_allocations_inner_false, initial_time_allocations_inner_for]

            for time_allocation in self.contract_program.allocations:
                # Check that it's not None
                if time_allocation.time is None:
                    continue
                initial_time_allocations_outer.append(round(time_allocation.time, self.contract_program.decimals))

            if self.contract_program.child_programs:
                # TODO: DO this logic everywhere (11/7)
                for child_index in range(0, len(self.contract_program.child_programs)):
                    for time_allocation in self.contract_program.child_programs[child_index].allocations:
                        # Check that it's not None
                        if time_allocation.time is None:
                            continue
                        initial_time_allocations_inner[child_index].append(round(time_allocation.time, self.contract_program.decimals))

            eu_initial = round(eu_initial, self.contract_program.decimals)

        else:
            if outer_program.child_programs:
                initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                                  self.contract_program.allocations]
                for child_index in range(0, len(self.contract_program.child_programs)):
                    initial_time_allocations_inner[child_index] = [time_allocation.time for time_allocation in
                                                                   self.contract_program.child_programs[child_index].allocations]
            else:
                initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                                  self.contract_program.allocations]

        if outer_program.child_programs:
            # The initial time allocations for each contract algorithm
            print("         Initial (Uniform) ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

            print("{:<62}Time Allocations (inner-true): {}".format("", initial_time_allocations_inner[0]))
            print("{:<62}Time Allocations (inner-false): {}".format("", initial_time_allocations_inner[1]))
            print("{:<62}Time Allocations (inner-for): {}".format("", initial_time_allocations_inner[2]))

        else:
            print(" {} \n ----------------------".format(initial_allocation))
            # The initial time allocations for each contract algorithm
            print("                   Initial ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

        ##############################################################################################################################
        # RHC ALLOCATION
        ##############################################################################################################################

        # Should output a list of lists of optimal time allocations
        # This is the bulk of the code
        allocations = self.contract_program.naive_hill_climbing_outer(verbose=verbose)

        if outer_program.child_programs:

            print("TEST 2")
            utils.print_allocations(allocations[0])
            utils.print_allocations(allocations[1])
            utils.print_allocations(allocations[2])
            utils.print_allocations(allocations[3])

            optimal_time_allocations_outer = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[0]])
            optimal_time_allocations_inner_true = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[1]])
            optimal_time_allocations_inner_false = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[2]])
            optimal_time_allocations_inner_for = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[3]])

            eu_optimal = self.contract_program.global_expected_utility(allocations[0], self.contract_program.best_allocations_inner) * self.contract_program.scale
            eu.append(eu_optimal)

            ehc_allocations_list = [allocations[0], allocations[1], allocations[2], allocations[3]]
            # TODO: This code is redundant (refactor)
            cleaned_allocations_list = []

            # Remove all none time allocations and append to list
            for index, allocations in enumerate(ehc_allocations_list):
                # Do some transformatioins and deletion depending on allocation to get allocations for plotting
                # remove nones
                cleaned_allocations = utils.remove_nones_time_allocations(allocations)
                if index == 0:
                    # TODO: Hardcoded
                    # remove the conditional and for node allocations
                    cleaned_allocations.pop(1)  # This is the conditiional
                    cleaned_allocations.pop(2)  # THis is the for
                elif index == 1:
                    # remove the last part of the true branch
                    cleaned_allocations.pop(len(cleaned_allocations) - 1)  # This is the tax
                elif index == 2:
                    # remove the last part of the true branch
                    cleaned_allocations.pop(len(cleaned_allocations) - 1)  # This is the tax
                elif index == 3:
                    # remove the last part of the true branch
                    cleaned_allocations.pop(len(cleaned_allocations) - 1)  # This is the 0 allocation

                cleaned_allocations_list.append(cleaned_allocations)

            # Flatten all the allocations
            flattened_allocations_list = utils.flatten_list(cleaned_allocations_list)

            # Sort the flattened list in ascending order
            sorted_allocations_list = sorted(flattened_allocations_list, key=lambda time_allocation: time_allocation.node_id, reverse=False)

            for index in range(0, len(self.node_indicies_list)):
                time[index].append(sorted_allocations_list[index].time)

            if self.contract_program.decimals is not None:
                optimal_time_allocations_outer = [round(time, self.contract_program.decimals) for
                                                  time in optimal_time_allocations_outer]
                optimal_time_allocations_inner_true = [round(time, self.contract_program.decimals) for
                                                       time in optimal_time_allocations_inner_true]
                optimal_time_allocations_inner_false = [round(time, self.contract_program.decimals) for
                                                        time in optimal_time_allocations_inner_false]
                optimal_time_allocations_inner_for = [round(time, self.contract_program.decimals) for
                                                      time in optimal_time_allocations_inner_for]

                eu_optimal = round(eu_optimal, self.contract_program.decimals)

            # End the timer
            end = timer()

            print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_optimal, optimal_time_allocations_outer))

            print("{:<62}Time Allocations (inner-true): {}".format("", optimal_time_allocations_inner_true))
            print("{:<62}Time Allocations (inner-false): {}".format("", optimal_time_allocations_inner_false))
            print("{:<62}Time Allocations (inner-for): {}".format("", optimal_time_allocations_inner_for))

            print("{:<62}Execution Time (seconds): {}".format("", end - start))

        else:
            optimal_time_allocations = utils.remove_nones_times([time_allocation.time for time_allocation in allocations])

            eu_optimal = self.contract_program.global_expected_utility(allocations) * self.contract_program.scale
            eu.append(eu_optimal)

            for index in range(0, len(self.node_indicies_list)):
                time[index].append(sorted_allocations_list[index])

            if self.contract_program.decimals is not None:

                optimal_time_allocations = [round(time, self.contract_program.decimals) for
                                            time in optimal_time_allocations]

                eu_optimal = round(eu_optimal, self.contract_program.decimals)

            # End the timer
            end = timer()

            print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                  "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))

            print("{:<62}Execution Time (seconds): {}".format("", end - start))

        return [eu, time]

    def find_utility_and_allocations_for(self, initial_allocation, outer_program, verbose=False) -> None:

        start = timer()

        # Generate an initial allocation pointed to self.contract_program.allocations relative to the type of allocation
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)

        # utils.print_allocations(self.contract_program.allocations)

        eu_initial = self.contract_program.global_expected_utility(self.contract_program.allocations) * self.contract_program.scale

        if self.contract_program.decimals is not None:
            initial_time_allocations_outer = []
            initial_time_allocations_inner_for = []

            for time_allocation in self.contract_program.allocations:
                # Check that it's not None
                if time_allocation.time is None:
                    continue

                initial_time_allocations_outer.append(round(time_allocation.time, self.contract_program.decimals))

            # TODO: If a composition exists with different types of expressions, this needs to be changed for arbitrary expressions (8/18)

            if self.contract_program.child_programs:
                for time_allocation in self.contract_program.child_programs[0].allocations:
                    # Check that it's not None
                    if time_allocation.time is None:
                        continue

                    initial_time_allocations_inner_for.append(round(time_allocation.time, self.contract_program.decimals))

            eu_initial = round(eu_initial, self.contract_program.decimals)

        else:
            if outer_program.child_programs:

                initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                                  self.contract_program.allocations]
                initial_time_allocations_inner_for = [time_allocation.time for time_allocation in
                                                      self.contract_program.child_programs[0].allocations]

            else:
                initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                                  self.contract_program.allocations]

        if outer_program.child_programs:

            print(" {} \n ----------------------".format(initial_allocation))
            # The initial time allocations for each contract algorithm
            print("                   Initial ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

            print("{:<62}Time Allocations (inner-for): {}".format("", initial_time_allocations_inner_for))

        else:
            print(" {} \n ----------------------".format(initial_allocation))
            # The initial time allocations for each contract algorithm
            print("                   Initial ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

        # Should output a list of lists of optimal time allocations
        allocations = self.contract_program.naive_hill_climbing_outer(verbose=verbose)

        if outer_program.child_programs:
            optimal_time_allocations_outer = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[0]])
            optimal_time_allocations_inner_for = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[1]])

            eu_optimal = self.contract_program.global_expected_utility(allocations[0],
                                                                       self.contract_program.best_allocations_inner) * self.contract_program.scale

            if self.contract_program.decimals is not None:

                optimal_time_allocations_outer = [round(time, self.contract_program.decimals) for
                                                  time in optimal_time_allocations_outer]

                optimal_time_allocations_inner_for = [round(time, self.contract_program.decimals) for
                                                      time in optimal_time_allocations_inner_for]

                eu_optimal = round(eu_optimal, self.contract_program.decimals)

            # End the timer
            end = timer()

            print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_optimal, optimal_time_allocations_outer))

            print("{:<62}Time Allocations (inner-for): {}".format("", optimal_time_allocations_inner_for))

            print("{:<62}Execution Time (seconds): {}".format("", end - start))

        else:
            optimal_time_allocations = utils.remove_nones_times([time_allocation.time for time_allocation in allocations])

            eu_optimal = self.contract_program.global_expected_utility(allocations) * self.contract_program.scale

            if self.contract_program.decimals is not None:

                optimal_time_allocations = [round(time, self.contract_program.decimals) for
                                            time in optimal_time_allocations]

                eu_optimal = round(eu_optimal, self.contract_program.decimals)

            # End the timer
            end = timer()

            print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                  "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))

            print("{:<62}Execution Time (seconds): {}".format("", end - start))

    def find_utility_and_allocations_conditional(self, initial_allocation, outer_program, verbose=False) -> None:
        start = timer()

        # Generate an initial allocation pointed to self.contract_program.allocations relative to the type of allocation
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)

        eu_initial = self.contract_program.global_expected_utility(self.contract_program.allocations) * self.contract_program.scale

        if self.contract_program.decimals is not None:

            initial_time_allocations_outer = []
            initial_time_allocations_inner_true = []
            initial_time_allocations_inner_false = []

            for time_allocation in self.contract_program.allocations:

                # Check that it's not None
                if time_allocation.time is None:
                    continue

                initial_time_allocations_outer.append(round(time_allocation.time, self.contract_program.decimals))

            # TODO: If a composition exists with different types of expressions, this needs to be changed for arbitrary expressions (8/18)

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

            # TODO: Make this more general as well (8/18)

            if outer_program.child_programs:

                initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                                  self.contract_program.allocations]
                initial_time_allocations_inner_true = [time_allocation.time for time_allocation in
                                                       self.contract_program.child_programs[0].allocations]
                initial_time_allocations_inner_false = [time_allocation.time for time_allocation in
                                                        self.contract_program.child_programs[1].allocations]

            else:
                initial_time_allocations_outer = [time_allocation.time for time_allocation in
                                                  self.contract_program.allocations]

        if outer_program.child_programs:

            print(" {} \n ----------------------".format(initial_allocation))
            # The initial time allocations for each contract algorithm
            print("                   Initial ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

            print("{:<62}Time Allocations (inner-true): {}".format("", initial_time_allocations_inner_true))

            print("{:<62}Time Allocations (inner-false): {}".format("", initial_time_allocations_inner_false))

        else:
            print(" {} \n ----------------------".format(initial_allocation))
            # The initial time allocations for each contract algorithm
            print("                   Initial ==> Expected Utility: {:<5} ==> "
                  "Time Allocations (outer): {}".format(eu_initial, initial_time_allocations_outer))

        # Should output a list of lists of optimal time allocations
        allocations = self.contract_program.naive_hill_climbing_outer(verbose=verbose)

        if outer_program.child_programs:
            optimal_time_allocations_outer = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[0]])
            optimal_time_allocations_inner_true = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[1]])
            optimal_time_allocations_inner_false = utils.remove_nones_times([time_allocation.time for time_allocation in allocations[2]])

            eu_optimal = self.contract_program.global_expected_utility(allocations[0],
                                                                       self.contract_program.best_allocations_inner) * self.contract_program.scale

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

        else:
            optimal_time_allocations = utils.remove_nones_times([time_allocation.time for time_allocation in allocations])

            eu_optimal = self.contract_program.global_expected_utility(allocations) * self.contract_program.scale

            if self.contract_program.decimals is not None:

                optimal_time_allocations = [round(time, self.contract_program.decimals) for
                                            time in optimal_time_allocations]

                eu_optimal = round(eu_optimal, self.contract_program.decimals)

            # End the timer
            end = timer()

            print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
                  "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))

            print("{:<62}Execution Time (seconds): {}".format("", end - start))

    def find_utility_and_allocations_default(self, initial_allocation, outer_program, verbose=False) -> None:

        start = timer()

        # Generate an initial allocation pointed to self.contract_program.allocations relative to the type of allocation
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)

        utils.print_allocations(self.contract_program.allocations)

        eu_initial = self.contract_program.global_expected_utility(self.contract_program.allocations) * self.contract_program.scale

        if self.contract_program.decimals is not None:

            initial_time_allocations = []

            for time_allocation in self.contract_program.allocations:

                # Check that it's not None
                if time_allocation.time is None:
                    continue

                initial_time_allocations.append(round(time_allocation.time, self.contract_program.decimals))

            eu_initial = round(eu_initial, self.contract_program.decimals)

        else:
            initial_time_allocations = [time_allocation.time for time_allocation in self.contract_program.allocations]

        print(" {} \n ----------------------".format(initial_allocation))
        # The initial time allocations for each contract algorithm
        print("                   Initial ==> Expected Utility: {:<5} ==> "
              "Time Allocations: {}".format(eu_initial, initial_time_allocations))

        # Should output a list of lists of optimal time allocations
        allocations = self.contract_program.naive_hill_climbing_outer(verbose=verbose)

        optimal_time_allocations = utils.remove_nones_times([time_allocation.time for time_allocation in allocations])

        eu_optimal = self.contract_program.global_expected_utility(allocations) * self.contract_program.scale

        if self.contract_program.decimals is not None:

            optimal_time_allocations = [round(time, self.contract_program.decimals) for
                                        time in optimal_time_allocations]

            eu_optimal = round(eu_optimal, self.contract_program.decimals)

        # End the timer
        end = timer()

        print("Naive Hill Climbing Search ==> Expected Utility: {:<5} ==> "
              "Time Allocations: {}".format(eu_optimal, optimal_time_allocations))

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

    def save_eu_time_data(self, eu_time_list, eu_file_path, time_file_path, node_indicies):
        # Check if data files exist
        if not os.path.isfile(eu_file_path):
            with open(eu_file_path, 'wb') as file_eus:
                pickle.dump([[] for i in range(0, self.num_plot_methods)], file_eus)

        if not os.path.isfile(time_file_path):
            with open(time_file_path, 'wb') as file_times:
                pickle.dump([[[] for j in range(0, self.num_plot_methods)] for i in range(0, len(node_indicies))], file_times)

        # Open files in binary mode with wb instead of w
        file_eus = open(eu_file_path, 'rb')
        file_times = open(time_file_path, 'rb')

        # Load the saved embedded lists to append new data
        pickled_eu_list = pickle.load(file_eus)
        pickled_time_list = pickle.load(file_times)

        # Append the EUs appropriately to list in outer scope
        for method_index in range(0, self.num_plot_methods):
            pickled_eu_list[method_index].append(eu_time_list[0][method_index])
            for node in range(0, len(node_indicies)):
                pickled_time_list[node][method_index].append(eu_time_list[1][node][method_index])

        with open(eu_file_path, 'wb') as file_eus:
            pickle.dump(pickled_eu_list, file_eus)

        with open(time_file_path, 'wb') as file_times:
            pickle.dump(pickled_time_list, file_times)

    def save_eu_monitoring_data(self, sequences, eu_monitoring_file_path):
         # Check if data files exist
        if not os.path.isfile(eu_monitoring_file_path):
            with open(eu_monitoring_file_path, 'wb') as file:
                pickle.dump(sequences, file)
        else:
            # clear the data in the info file
            with open(eu_monitoring_file_path,'wb') as file:
                pass
            with open(eu_monitoring_file_path,'wb') as file:
                pickle.dump(sequences, file)

    def initial_allocation_setup(self, initial_allocation, contract_program):
        if initial_allocation == "uniform":

            contract_program.allocations = contract_program.initialize_allocations.uniform_budget()

            # Find inner contract programs
            inner_contract_programs = self.find_inner_programs(contract_program)

            if inner_contract_programs:
                for inner_contract_program in inner_contract_programs:

                    # initialize the allocations to the inner contract programs with the time allocation of the outer conditonal node
                    if inner_contract_program.subprogram_expression_type == "for":
                        inner_contract_program.change_budget(contract_program.allocations[self.find_node_id_of_for(contract_program)].time)
                        # print(inner_contract_program.program_dag.orders)
                    elif inner_contract_program.subprogram_expression_type == "conditional":
                        inner_contract_program.change_budget(contract_program.allocations[self.find_node_id_of_conditional(contract_program)].time)

                    self.initial_allocation_setup(initial_allocation, inner_contract_program)

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

            if not outer_node.in_child_contract_program:

                if Node.is_for_node(outer_node):
                    # Append its subprograms to the list
                    inner_programs.extend([outer_node.for_subprogram])

                elif Node.is_conditional_node(outer_node):
                    # Append its subprograms to the list
                    inner_programs.extend([outer_node.true_subprogram, outer_node.false_subprogram])

        return inner_programs

    @staticmethod
    def find_node_id_of_conditional(outer_program):
        for outer_node in outer_program.program_dag.nodes:
            if Node.is_conditional_node(outer_node):
                return outer_node.id
        raise Exception("Didn't find a conditional Node")

    @staticmethod
    def find_node_id_of_for(outer_program):
        for outer_node in outer_program.program_dag.nodes:
            if Node.is_for_node(outer_node):
                return outer_node.id
        raise Exception("Didn't find a for Node")
