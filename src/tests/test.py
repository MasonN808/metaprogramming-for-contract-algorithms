import copy
import os
import pickle
import sys
from time import sleep
from progress.bar import ChargingBar
from timeit import default_timer as timer


sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")
from classes import utils  # noqa
from classes.node import Node  # noqa


class Test:
    def __init__(self, contract_program, node_indicies_list, plot_type=None, plot_nodes=None):
        self.contract_program = contract_program
        self.node_indicies_list = node_indicies_list
        self.plot_type = plot_type
        self.num_plot_methods = 0
        self.plot_nodes = plot_nodes

    def monitor_eu_on_rhc(self, initial_allocation, outer_program, verbose=False):
        # Setup the initial time allocations
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)
        return self.contract_program.recursive_hill_climbing(verbose=verbose, monitoring=True)

    # For arbitrary contract programs using a variety of solution methods
    def find_utility_and_allocations(self, initial_allocation, outer_program, test_phis=[], verbose=False):
        # Data for plotting
        eu = []
        # To monitor times for specific nodes
        time = [[] for _ in range(0, len(self.contract_program.full_dag.nodes))]

        ##############################################################################################################################
        # PROPORTIONAL ALLOCATION
        ##############################################################################################################################
        # Add up the number of methods where +2 represents hill climbing and uniform allocation
        self.num_plot_methods += len(test_phis) + 2

        # Get the eu for the proportional allocation method
        for phi in test_phis:
            self.contract_program.proportional_allocation_tangent(phi)

            eu_proportional = self.contract_program.expected_utility() * self.contract_program.scale
            eu.append(eu_proportional)

            index = 0
            for node in self.contract_program.full_dag.nodes:
                if node.time is None:
                    continue
                time[index].append(node.time)
                index += 1

        ##############################################################################################################################
        # UNIFORM ALLOCATION
        ##############################################################################################################################
        # TODO: GENERALIZE THIS (2/10)
        # Generate an initial allocation pointed to self.contract_program.allocations relative to the type of allocation
        self.initial_allocation_setup(initial_allocation=initial_allocation, contract_program=outer_program)
        # print([node.time for node in self.contract_program.full_dag.nodes])
        eu_initial = self.contract_program.expected_utility() * self.contract_program.scale
        eu.append(eu_initial)

        index = 0
        for node in self.contract_program.full_dag.nodes:
            if node.time is None:
                continue
            time[index].append(node.time)
            index += 1

        ##############################################################################################################################
        # RHC ALLOCATION
        ##############################################################################################################################
        self.contract_program.recursive_hill_climbing(verbose=verbose)

        eu_optimal = self.contract_program.expected_utility() * self.contract_program.scale
        eu.append(eu_optimal)

        index = 0
        for node in self.contract_program.full_dag.nodes:
            if node.time is None:
                continue
            time[index].append(node.time)
            index += 1

        return [eu, time]

    def save_eu_time_data(self, eu_time_list, eu_file_path, time_file_path, node_indicies, clear_file=False):
        # Check if data files exist
        if not os.path.isfile(eu_file_path) or clear_file == True:
            with open(eu_file_path, 'wb') as file_eus:
                pickle.dump([[] for i in range(0, self.num_plot_methods)], file_eus)

        if not os.path.isfile(time_file_path) or clear_file == True:
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
            with open(eu_monitoring_file_path, 'wb') as file:
                pass
            with open(eu_monitoring_file_path, 'wb') as file:
                pickle.dump(sequences, file)

    def initial_allocation_setup(self, initial_allocation, contract_program, depth=0):
        if initial_allocation != "uniform":
            raise ValueError("Invalid initial allocation type")
        else:
            for node in contract_program.program_dag.nodes:
                # Give a uniform allocation to each node
                uniform_allocation = contract_program.find_uniform_allocation(contract_program.budget)
                if (node.expression_type == "conditional" or node.expression_type == "for") and depth == 1:
                    continue
                # TODO: use the previous depth as well to make arbitrary contract programs work
                elif node.expression_type == "conditional" and depth == 0:
                    node.time = uniform_allocation
                    node.true_subprogram.budget = uniform_allocation
                    node.false_subprogram.budget = uniform_allocation
                elif node.expression_type == "for" and depth == 0:
                    node.time = uniform_allocation
                    node.for_subprogram.budget = uniform_allocation
                else:
                    node.time = uniform_allocation
            if contract_program.subprogram_map:
                for _, subprogram in contract_program.subprogram_map.items():
                    # Recursively allocate a uniform allocation to each subprogram
                    self.initial_allocation_setup(initial_allocation, subprogram, depth=depth + 1)
            # Append all the allocations in the subprograms and outer program to the full dag
            if depth == 0:
                # Revert to the orginal time allocations before switch
                if self.contract_program.subprogram_map:
                    for _, subprogram in self.contract_program.subprogram_map.items():
                        self.contract_program.change_time_allocations(self.contract_program.full_dag.nodes, subprogram.program_dag.nodes)
                self.contract_program.change_time_allocations(self.contract_program.full_dag.nodes, self.contract_program.program_dag.nodes)


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
