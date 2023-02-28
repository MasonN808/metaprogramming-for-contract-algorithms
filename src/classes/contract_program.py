import sys
from typing import List
import copy
import math
from itertools import permutations

import numpy as np

sys.path.append("/Users/masonnakamura/Local-Git/metaprogramming-for-contract-algorithms/src")

from classes.time_allocation import TimeAllocation  # noqa
from classes import utils  # noqa
from classes.node import Node  # noqa
from classes.performance_profile import PerformanceProfile  # noqa

class ContractProgram:
    """
    Structures a directed-acyclic graph (DAG) as a contract program by applying a budget on a DAG of
    contract algorithms. The edges are directed from the leaves to the root.

    :param: budget : non-negative int, required
        The budget of the contract program represented as seconds
    :param: program_dag : DAG, required
        The DAG that the contract program inherits
    :param: scale : float, required
        The scale that transforms the printed expected utility for easier interpretation
    :param: decimals : int, required
        The number of decimal points that adjusts the printed expected utility and allocations for easier interpretation
    :param: quality_interval : float, required
        The interval used to help calculate the performance profiles (probabilities)
    :param: time_interval : float, required
        The interval used to help calculate the performance profiles (probabilities)
    """
    POPULOUS_FILE_NAME = "quality_mappings/populous.json"

    def __init__(self, program_id, parent_program, child_programs, program_dag, budget, scale, decimals,
                 quality_interval, time_interval, time_step_size, in_child_contract_program, full_dag,
                 expected_utility_type, possible_qualities, performance_profile_velocities=None,
                 number_of_loops=None, subprogram_expression_type=None, subprogram_map={}, sum_growth_factors = None):

        self.program_id = program_id
        self.subprogram_expression_type = subprogram_expression_type  # for or false or true
        self.program_dag = program_dag
        self.budget = budget
        self.scale = scale
        self.decimals = decimals
        self.quality_interval = quality_interval
        self.time_interval = time_interval
        self.time_step_size = time_step_size
        self.allocations = None
        self.in_child_contract_program = in_child_contract_program
        self.subprogram_map = subprogram_map

        self.sum_growth_factors = sum_growth_factors
        # Pointer to the parent program that the subprogram is an induced subgraph of
        self.parent_program = parent_program
        self.child_programs = child_programs
        self.full_dag = full_dag
        self.best_allocations_inner = None
        self.expected_utility_type = expected_utility_type
        self.possible_qualities = possible_qualities
        self.performance_profile_velocities = performance_profile_velocities
        self.number_of_loops = number_of_loops
        self.performance_profile = PerformanceProfile(program_dag=self.program_dag, full_dag=self.full_dag,
                                                      file_name=self.POPULOUS_FILE_NAME,
                                                      time_interval=self.time_interval, time_limit=budget,
                                                      quality_interval=self.quality_interval,
                                                      time_step_size=self.time_step_size,
                                                      expected_utility_type=self.expected_utility_type)

    @staticmethod
    def utility(qualities) -> float:
        """
        Gives a utility on a list of qualities

        :param qualities: Qualities[], required
                The qualities that were outputted for each contract algorithm in the DAG
        :return: float
        """
        # Flatten the list of qualities
        qualities = utils.flatten(qualities)

        return math.prod(qualities)

    def expected_utility(self) -> float:
        """
        Uses approximate methods or exact solutions to query the expected utility of the contract program given the time allocations

        :param best_allocations_inner: a list of the best allocations to compare with the current allocations
        :param expression_type: the expression type being optimized
        :param time_allocations: float[], required
                The time allocations for each contract algorithm
        :return: float
        """
        if self.expected_utility_type == "exact":
            return (self.expected_utility_exact())
        elif self.expected_utility_type == "approximate":
            return (self.expected_utility_approximate())
        else:
            raise ValueError("Improper expected utility type")

    def expected_utility_approximate(self) -> float:
        """
        Gives the estimated expected utility of the contract program given the performance profiles of the nodes
        (i.e., the probability distribution of each contract program's conditional performance profile) and the
        global utility

        :param best_allocations_inner:
        :param time_allocations: float[], required
                The time allocations for each contract algorithm
        :return: float
        """
        probability = 1.0
        # Input to calculate our utility function
        qualities = []
        for node in self.full_dag.nodes:
            # Skip the conditional and for node in a subcontract program since they have no performance profile
            if (node.expression_type == "for" and not "{}".format(node.id) in self.subprogram_map) or (node.expression_type == "conditional" and not "{}-0".format(node.id) in self.subprogram_map):
                continue
            # Calculates the EU of a conditional expression in the outermost contract program
            # Check if the node is in a subprogram using the programs hashmap of subprograms
            elif (node.expression_type == "conditional" and "{}-0".format(node.id) in self.subprogram_map):
                # Since in conditional node of the outermost contract program, evaluate the inner probability of the child conditional subprogram
                probability_and_qualities = self.performance_profile.query_probability_and_quality_from_conditional_expression(node)
                # Multiply the current probability by the performance profile of the conditional node
                probability *= probability_and_qualities[0]
                conditional_quality = probability_and_qualities[1]
                qualities.append(conditional_quality)
            # Calculates the EU of a for expression in the outermost contract program
            elif node.expression_type == "for" and "{}".format(node.id) in self.subprogram_map:
                # Since in for node of the outermost contract program, evaluate the inner probability of the child for subprogram
                probability_and_qualities = self.performance_profile.query_probability_and_quality_from_for_expression(node.for_subprogram)
                # Multiply the current probability by the performance profile of the conditional node
                probability *= probability_and_qualities[0]
                last_for_quality = probability_and_qualities[1]
                # We use the last quality of the for loop to calculate our utiltiy
                qualities.append(last_for_quality)
            else:
                # Get the parents' qualities given their time allocations
                node = utils.find_node_in_full_dag(node, full_dag=self.full_dag)
                parent_qualities = self.performance_profile.find_parent_qualities(node, depth=0)
                # Query the quality that is pulled from a Guassian distribution
                # Let the output quality be dependent on the parent qualities
                quality = self.performance_profile.query_quality(node)
                if parent_qualities:
                    quality *= min(parent_qualities)
                    # quality *= np.mean(parent_qualities)
                probability *= self.performance_profile.query_probability_contract_expression(quality, node)
                qualities.append(quality)
        # print("PROB: {}".format(probability))
        # print("utility: {}".format(self.utility(qualities)))
        # print("qualities: {}".format(qualities))
        
        expected_utility = probability * self.utility(qualities)
        return expected_utility

    def expected_utility_exact(self) -> float:
        """
        Gives the exact expected utility of the contract program given the performance profiles of the nodes
        (i.e., the probability distribution of each contract program's conditional performance profile) and the
        global utility

        Assumption: A time-allocation is given to each node in the contract program

        :param best_allocations_inner:
        :param time_allocations: float[], required
                The time allocations for each contract algorithm
        :return: float
        """
        # Use the full_dag since we want the entire program and not the hierarcy of programs (outer and inners)
        leaves = utils.find_leaves_in_dag(self.full_dag)
        # Unions the time allocation vectors in all the contract programs (does not do embedded yet)
        # if self.child_programs:
        #     for child_program in self.child_programs:
        #         for allocation in child_program.allocations:
        #             if allocation.time is not None:
        #                 time_allocations[allocation.node_id] = allocation

        # if self.parent_program:
        #     for allocation in self.parent_program.allocations:
        #         if allocation.time is not None:
        #             time_allocations[allocation.node_id] = allocation

        # utils.print_allocations(time_allocations)
        return self.find_exact_expected_utility(possible_qualities=self.possible_qualities, expected_utility=0,
                                                current_qualities=[0.0 for i in range(self.full_dag.order)], parent_qualities=[],
                                                depth=0, leaves=leaves, total_sum=0)

    # eu = 0
    # for q_1 in 1 to max_Q:
    #     ...
    #     for q_n in 1 to max_Q:
    #         pr(q_1, ..., q_n) = recursion
    #         eu += pr(q_1, ..., q_n) * utility(q_1, ..., q_n)

    # TODO: Fix this (9/22)
    def find_exact_expected_utility(self, leaves, depth, expected_utility, current_qualities, parent_qualities, possible_qualities, total_sum) -> float:
        """
        Returns the exact EU

        :param: depth: The depth of the recursive call
        :param: node: Node object, finding the parent qualities of this node
        :param: time_allocations: float[] (order matters), for the entire DAG
        :return: A list of parent qualities
        """
        # sum = copy.deepcopy(sum)
        # Recur down the DAG
        depth += 1
        # print("DEPTH: {}".format(depth))
        if leaves:
            for node in leaves:
                if node.parents and depth != 1:
                    parents = node.parents
                    for_and_conditional_nodes = []

                    for parent in parents:
                        # Check parents aren't fors or conditionals
                        # If so, get its parents instead
                        if parent.expression_type == "for" or parent.expression_type == "conditional":
                            # TODO: Make this for an arbitrary branch structure not just a P_n graph
                            parents.extend(parent.parents)
                            for_and_conditional_nodes.append(parent)
                            continue
                        # Use the qualities from the previous possible qualities in the parent nodes
                        # as parent qualities to query from performance profiles
                        parent_qualities.append(self.performance_profile.query_quality(parent))

                    # remove the for and conditional nodes from the parents list
                    for for_node_or_conditional_node in for_and_conditional_nodes:
                        parents.remove(for_node_or_conditional_node)

                # Loop through all possible qualities on the current node
                for possible_quality in possible_qualities:
                    current_qualities[node.id] = possible_quality

                    # print("TEST: {}, {}".format(node.time, node.id))
                    # print("PARENT QUALITIES: {}".format(parent_qualities))
                    # print("ID -- {}".format(node.id))

                    # Check if the node is the conditional root, then average the quality of the parents for the root
                    if node.is_conditional_root:
                        parent_qualities = [sum(parent_qualities) / len(parent_qualities)]

                    probability = self.performance_profile.query_probability_contract_expression(possible_quality, node)

                    # print("PROB: {}".format(conditional_probability))

                    # Check node.children has no for or conditional nodes
                    # If so, skip it since no relevant performance profile can be queried from stored performance profiles
                    for child in node.children:
                        if child.expression_type == "for" or child.expression_type == "conditional":
                            # TODO: Make this for an arbitrary branch structure not just a P_n graph
                            node.children.extend(child.children)
                            node.children.remove(child)

                    # Traverse up the DAG
                    new_leaves = node.children
                    # print([i.id for i in new_leaves])

                    # TODO: Subtract by 1 and the number of fors and conditionals in DAG
                    if depth == self.full_dag.order - 1:
                        # Remove nones from the list since current qualities will have model qualities for every node in the generator dag
                        utility = self.utility(utils.remove_nones_list(current_qualities))
                        # print("UTILITY: {}".format(utility))
                        total_sum += probability * utility
                    else:
                        total_sum += probability * self.find_exact_expected_utility(leaves=new_leaves, depth=depth,
                                                                                                expected_utility=expected_utility, current_qualities=current_qualities,
                                                                                                possible_qualities=possible_qualities, parent_qualities=[], total_sum=0)

            # print("FINAL SUM: {}".format(total_sum))
            # print(depth)
            return total_sum

        # If we hit the bottom of the recursion (i.e., the root)
        else:
            return total_sum

    def naive_hill_climbing_no_children_no_parents(self, decay=1.01, threshold=.01, verbose=False) -> float:
        # Initialize the amount of time to be switched
        time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)
        eu_original = self.expected_utility() * self.scale
        while time_switched > threshold:
            best_allocations_changed = False
            # Go through all permutations of the time allocations
            for permutation in permutations(self.program_dag.nodes, 2):
                node_0 = permutation[0]
                node_1 = permutation[1]
                # Avoids exchanging time with itself
                if node_0.id == node_1.id:
                    continue
                # Avoids negative time allocation
                elif node_0.time - time_switched < 0:
                    continue
                else:
                    node_0.time -= time_switched
                    node_1.time += time_switched
                    eu_adjusted = self.expected_utility() * self.scale

                    if eu_adjusted > eu_original:
                        best_allocations_changed = True
                        eu_original = eu_adjusted
                    else:
                        # Revert the time switched
                        node_0.time += time_switched
                        node_1.time -= time_switched

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations_outer = [round(node.time, self.decimals) for node in self.program_dag.nodes]
                        printed_eu_adjusted = round(eu_adjusted, self.decimals)
                        printed_eu_original = round(eu_original, self.decimals)
                        printed_time_switched = round(time_switched, self.decimals)

                    if verbose:
                        message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
                        print(message.format(printed_time_switched, printed_eu_adjusted, printed_eu_original, print_allocations_outer))

            if not best_allocations_changed:
                time_switched = time_switched / decay

        return eu_original

    # This case is specifically for a contract program with conditionals, fors, and contracts
    def recursive_hill_climbing(self, depth=0, decay=1.1, threshold=.01, verbose=False) -> List[float]:
        """
        Does hill climbing specific to an outer contract program with conditional subprograms

        :param verbose: Verbose mode
        :param threshold: float, the threshold of the temperature decay during annealing
        :param decay: float, the decay rate of the temperature during annealing
        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        # The initial true_allocations, false_allocations, and for_allocations have uniform alloations
        # Initialize the amount of time to be switched
        # if depth == 1:
            # self.initial_allocation_setup(initial_allocation="uniform", depth=depth)
        time_switched = self.find_uniform_allocation(self.budget)
        eu_original = self.expected_utility() * self.scale
        while time_switched > threshold:
            best_allocations_changed = False
            # Go through all permutations of the time allocations
            for permutation in permutations(self.program_dag.nodes, 2):
                # Makes a deep copy to avoid pointers to the same list
                node_0 = permutation[0]
                node_1 = permutation[1]
                # Avoids exchanging time with itself
                if node_0.id == node_1.id:
                    continue
                # Avoids trading time with the meta conditional node in a subprogram
                elif ((node_0.expression_type == "conditional") or (node_0.expression_type == "for") or (node_1.expression_type == "conditional") or (node_1.expression_type == "for")) and depth == 1:
                    continue
                # Avoids reducing the time allocation below the conditional time lower bound (tau)
                elif ((node_0.expression_type == "conditional") and (node_0.time - time_switched < self.performance_profile.calculate_tau())) or ((node_1.expression_type == "conditional") and (node_1.time + time_switched < self.performance_profile.calculate_tau())):
                    continue
                # Avoids negative time allocation
                elif node_0.time - time_switched < 0:
                    continue
                else:
                    self.change_time_allocations(self.full_dag.nodes, self.program_dag.nodes)
                    # print("OUTER 1: {}".format([node.time for node in self.program_dag.nodes]))
                    # print("FULL 1: {}".format([node.time for node in self.full_dag.nodes]))
                    original_program_dag_nodes = copy.deepcopy(self.full_dag.nodes)
                    node_0.time -= time_switched
                    node_1.time += time_switched
                    # print("OUTER 2: {}".format([node.time for node in self.program_dag.nodes]))
                    self.change_time_allocations(self.full_dag.nodes, self.program_dag.nodes)
                    # print("FULL 2: {}".format([node.time for node in self.full_dag.nodes]))
                    # If we are in the outermost program
                    if depth == 0:
                        # Make a copy just in case recursvie hill climbing solution is suboptimal
                        for node in [node_0, node_1]:
                            # Does hill climbing on the outer metareasoning problem that is a conditional
                            # TODO: For emebedded subprograms, make the recursion a function of depth
                            if node.expression_type == "conditional":
                                # Reallocate the budgets for the inner metareasoning problems
                                node.true_subprogram.change_budget(copy.deepcopy(node.time), depth = depth+1)
                                node.false_subprogram.change_budget(copy.deepcopy(node.time), depth = depth+1)
                                # Do recursive naive hill climbing on the branches
                                node.true_subprogram.recursive_hill_climbing(depth=depth+1)
                                node.false_subprogram.recursive_hill_climbing(depth=depth+1)
                                # print([node.time for node in node.true_subprogram.program_dag.nodes])
                                # Change the time allocations in the outer program
                                self.change_time_allocations(self.full_dag.nodes, node.true_subprogram.program_dag.nodes)
                                self.change_time_allocations(self.full_dag.nodes, node.false_subprogram.program_dag.nodes)
                                # print("PROGRAM BAD HERE: {}".format([node.time for node in node.true_subprogram.program_dag.nodes]))
                                # print([node.time for node in self.full_dag.nodes])
                            elif node.expression_type == "for":
                                # Reallocate the budgets for the inner metareasoning problems
                                node.for_subprogram.change_budget(copy.deepcopy(node.time), depth = depth+1)
                                # Do recursive naive hill climbing on the loop
                                node.for_subprogram.recursive_hill_climbing(depth=depth+1)
                                # Change the time allocations in the outer program
                                self.change_time_allocations(self.full_dag.nodes, node.for_subprogram.program_dag.nodes)

                    # Best allocations from the previous iterations are needed since the allocations of the subprograms may be adjusted from above
                    # And scale the EU for interprettable results
                    # print("FULL 3: {}".format([node.time for node in self.full_dag.nodes]))
                    eu_adjusted = self.expected_utility() * self.scale

                    if eu_adjusted > eu_original:
                        best_allocations_changed = True
                        eu_original = eu_adjusted
                    else:
                        # Revert to the orginal time allocations before switch on the full dag and the outer dag
                        self.change_time_allocations(self.full_dag.nodes, original_program_dag_nodes)
                        self.change_time_allocations(self.program_dag.nodes, original_program_dag_nodes)
                        if self.subprogram_map:
                            for _, subprogram in self.subprogram_map.items():
                                self.change_time_allocations(subprogram.program_dag.nodes, original_program_dag_nodes)

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations = []
                        for node in self.full_dag.nodes:
                            if node.time is not None:
                                print_allocations.append(round(node.time, self.decimals))
                        printed_eu_adjusted = round(eu_adjusted, self.decimals)
                        printed_eu_original = round(eu_original, self.decimals)
                        temp_time_switched = round(time_switched, self.decimals)
                        
                    if verbose:
                        message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
                        print(message.format(temp_time_switched, printed_eu_adjusted, printed_eu_original, print_allocations))

            if not best_allocations_changed:
                time_switched = time_switched / decay

        return eu_original

    def proportional_allocation_tangent(self, phi=1) -> List[float]:
        number_conditionals_and_fors = utils.number_of_fors_conditionals(self.full_dag)
        number_conditionals = number_conditionals_and_fors[0]

        # Find the indicies for the for and conditional expressions if they exist
        true_indices = utils.find_true_indices(self.full_dag)
        false_indices = utils.find_false_indices(self.full_dag)
        
        copy_budget = self.budget
        # Tax the budget given any conditionals present in the program
        taxed_budget = copy_budget - number_conditionals * self.performance_profile.calculate_tau()

        growth_factors = [node.c for node in self.full_dag.nodes]

        # Use the inverse tangent function to transform coefficients from (0, infinity) -> (0,1)
        growth_factors_transformed = []
        for c in growth_factors:
            if c is None:
                growth_factors_transformed.append(None)
            else:
                growth_factors_transformed.append(1 - (math.atan(phi * c) / (math.pi / 2)))

        # Make sure to reduce the indices since we removed strings from growth_factors
        # Make sure that the left and right branches get the same time allocation
        transformed_branch_sum = 0
        if number_conditionals > 0:  # TODO: what if there are more than one conditional
            true_sum = 0
            false_sum = 0
            for index in true_indices:
                true_sum += growth_factors_transformed[index]
            for index in false_indices:
                false_sum += growth_factors_transformed[index]

            # Check which branch has a greater sum of growth factors
            # Then normalize the branch with the greater sum of growth factors by adding the average difference to each node
            # This results in the sum of growth factors of each branch to be equal
            if (false_sum > true_sum):
                difference = false_sum - true_sum
                for index in true_indices:
                    # Add the difference to the true nodes
                    growth_factors_transformed[index] += difference / len(true_indices)
                    transformed_branch_sum += growth_factors_transformed[index]
            else:
                difference = true_sum - false_sum
                for index in false_indices:
                    # Add the difference to the false nodes
                    growth_factors_transformed[index] += difference / len(false_indices)
                    transformed_branch_sum += growth_factors_transformed[index]

        real_growth_factors = [growth_factor for growth_factor in growth_factors_transformed if growth_factor is not None]

        # Get the coefficients proportinal to the budget and ...
        # subtract a sum of growth factors in one the branches since it double counts
        budget_proportion = taxed_budget / (sum(real_growth_factors) - transformed_branch_sum)

        # Assign the time allocations
        for node, transformed_growth_factor in zip(self.full_dag.nodes, growth_factors_transformed):
            if transformed_growth_factor is None:
                continue
            node.time = transformed_growth_factor * budget_proportion

        # Assign the allocations to the subprograms
        for _, subprogram in self.subprogram_map.items():
            self.change_time_allocations(subprogram.program_dag.nodes, self.full_dag.nodes)

        # Assign the allocations to the outer program
        self.change_time_allocations(self.program_dag.nodes, self.full_dag.nodes)
        print([(node.id, node.time) for node in self.full_dag.nodes])

        print("---------------------")
        print("Growth Factors: {}".format(growth_factors))
        print("Transformed Growth Factors: {}".format(growth_factors_transformed))

        return [node.time for node in self.full_dag.nodes]

    def change_budget(self, new_budget, depth) -> None:
        """
        Changes the budget of the contract program and adjusts the objects that use the budget of the
        contract program

        :param new_budget: float
        :return: None
        """
        self.budget = new_budget
        self.initial_allocation_setup("uniform", depth = depth)

    def change_time_allocations(self, receiving_program_nodes, giving_program_nodes):
        # TODO: optimize this using a hash table (2/26)
        for receiving_node in receiving_program_nodes:
            for giving_node in giving_program_nodes:
                if receiving_node.id == giving_node.id:
                    receiving_node.time = copy.deepcopy(giving_node.time)
                    # break out of the first embedded loop
                    break

    def append_growth_factors_to_subprograms(self):
        # TODO: optimize this using a hash table (2/26)
        for _, subprogram in self.subprogram_map.items():
            for program_node in self.full_dag.nodes:
                for subprogram_node in subprogram.program_dag.nodes:
                    if program_node.id == subprogram_node.id:
                        subprogram_node.c = program_node.c
                        # print("ID: {} --> {}".format(subprogram_node.id, subprogram_node.c))

        # Append growth factors to outer program as well
        for program_node in self.full_dag.nodes:
            for outer_node in self.program_dag.nodes:
                if program_node.id == outer_node.id:
                    outer_node.c = program_node.c

    def initial_allocation_setup(self, initial_allocation, depth=0):
        if initial_allocation != "uniform":
            raise ValueError("Invalid initial allocation type")
        else:
            for node in self.program_dag.nodes:
                # Give a uniform allocation to each node
                uniform_allocation = self.find_uniform_allocation(self.budget)
                print(uniform_allocation)
                if (node.expression_type == "conditional" or node.expression_type == "for") and depth == 1:
                    # node.time = None
                    continue
                elif node.expression_type == "conditional" and depth == 0:
                    node.time = uniform_allocation
                    node.true_subprogram.budget = uniform_allocation
                    node.false_subprogram.budget = uniform_allocation
                elif node.expression_type == "for" and depth == 0:
                    node.time = uniform_allocation
                    node.for_subprogram.budget = uniform_allocation
                else:
                    node.time = uniform_allocation
            if self.subprogram_map:
                for _, subprogram in self.subprogram_map.items():
                    # Recursively allocate a uniform allocation to each subprogram
                    subprogram.initial_allocation_setup(initial_allocation, depth=depth + 1)
            # Append all the allocations in the subprograms and outer program to the full dag
            if depth == 0:
                self.change_time_allocations(self.full_dag.nodes, self.program_dag.nodes)
                if self.subprogram_map:
                    for _, subprogram in self.subprogram_map.items():
                        self.change_time_allocations(self.full_dag.nodes, subprogram.program_dag.nodes)

    def find_uniform_allocation(self, budget) -> float:
        """
        Finds the allocation that can uniformly be distributed given the budget

        :param budget: float
        :return: uniformed allocation
        """
        number_of_conditionals = self.count_conditionals()
        number_of_fors = self.count_fors()
        allocation = budget / (self.program_dag.order - number_of_conditionals - number_of_fors)
        # allocation = budget / (self.program_dag.order)
        return allocation

    def count_conditionals(self) -> int:
        """
        Counts the number of conditionals in the contract program

        :return: number of conditionals:
        """
        number_of_conditionals = 0
        if self.subprogram_expression_type == "false" or self.subprogram_expression_type == "true":
            number_of_conditionals += 1
        return number_of_conditionals

    def count_fors(self) -> int:
        """
        Counts the number of fors in the contract program

        :return: number of fors:
        """
        number_of_fors = 0
        if self.subprogram_expression_type == "for":
            number_of_fors += 1
        return number_of_fors
