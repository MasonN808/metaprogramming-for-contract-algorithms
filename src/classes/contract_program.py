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
from classes.initialize_allocations import InitializeAllocations  # noqa


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
                 number_of_loops=None, subprogram_expression_type=None, subprogram_map=None):

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

        self.initialize_allocations = InitializeAllocations(budget=self.budget, program_dag=self.program_dag,
                                                            full_dag=self.full_dag,
                                                            performance_profile=self.performance_profile,
                                                            in_child_contract_program=self.in_child_contract_program)

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

    def expected_utility(self, time_allocations=[], best_allocations_inner=None) -> float:
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
        for node in self.program_dag.nodes:
            # Skip the conditional and for node in a subcontract program since they have no performance profile
            if (node.expression_type == "conditional" or node.expression_type == "for") and node.in_child_contract_program:
                continue
            # Calculates the EU of a conditional expression in the outermost contract program
            elif (node.expression_type == "conditional" and not node.in_child_contract_program):
                # Since in conditional node of the outermost contract program, evaluate the inner probability of the child conditional subprogram
                probability_and_qualities = self.performance_profile.query_probability_and_quality_from_conditional_expression(node)
                # Multiply the current probability by the performance profile of the conditional node
                probability *= probability_and_qualities[0]
                conditional_quality = probability_and_qualities[1]
                qualities.append(conditional_quality)
            # Calculates the EU of a for expression in the outermost contract program
            elif node.expression_type == "for" and not node.in_child_contract_program:
                # Since in for node of the outermost contract program, evaluate the inner probability of the child for subprogram
                probability_and_qualities = self.performance_profile.query_probability_and_quality_from_for_expression(node.for_subprogram)
                # Multiply the current probability by the performance profile of the conditional node
                probability *= probability_and_qualities[0]
                last_for_quality = probability_and_qualities[1]
                # We use the last quality of the for loop to calculate our utiltiy
                qualities.append(last_for_quality)
            else:
                # Get the parents' qualities given their time allocations
                parent_qualities = self.performance_profile.find_parent_qualities(node, depth=0)
                # Query the quality that is pulled from a Guassian distribution
                # Let the output quality be dependent on the parent qualities
                if parent_qualities:
                    quality = self.performance_profile.query_quality(node) * np.mean(parent_qualities)
                else:
                    quality = self.performance_profile.query_quality(node)
                qualities.append(quality)
                probability *= self.performance_profile.query_probability_contract_expression(quality, node)
                # if quality < 0:
                #     print(quality)
                #     exit()
                # print("PROBABILITY IN EU: {}".format(probability))
        #         if (probability == 0):
        #             print("prob is 0")
        # print("PROB: {}".format(probability))
        # print("utility: {}".format(self.utility(qualities)))
        expected_utility = probability * self.utility(qualities)

        return expected_utility

    def expected_utility_exact(self, time_allocations) -> float:
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

        time_allocations = copy.deepcopy(time_allocations)
        # Unions the time allocation vectors in all the contract programs (does not do embedded yet)
        if self.child_programs:
            for child_program in self.child_programs:
                for allocation in child_program.allocations:
                    if allocation.time is not None:
                        time_allocations[allocation.node_id] = allocation

        # if self.parent_program:
        #     for allocation in self.parent_program.allocations:
        #         if allocation.time is not None:
        #             time_allocations[allocation.node_id] = allocation

        # utils.print_allocations(time_allocations)
        return self.find_exact_expected_utility(time_allocations=time_allocations, possible_qualities=self.possible_qualities, expected_utility=0,
                                                current_qualities=[0.0 for i in range(self.full_dag.order)], parent_qualities=[],
                                                depth=0, leaves=leaves, total_sum=0)

    # eu = 0
    # for q_1 in 1 to max_Q:
    #     ...
    #     for q_n in 1 to max_Q:
    #         pr(q_1, ..., q_n) = recursion
    #         eu += pr(q_1, ..., q_n) * utility(q_1, ..., q_n)

    # TODO: Fix this (9/22)
    def find_exact_expected_utility(self, leaves, time_allocations, depth, expected_utility, current_qualities, parent_qualities, possible_qualities, total_sum) -> float:
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

                    # print(current_qualities)

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
                    node_time = time_allocations[node.id].time

                    print("TEST: {}, {}".format(node_time, node.id))
                    print("PARENT QUALITIES: {}".format(parent_qualities))
                    print("ID -- {}".format(node.id))

                    # Check if the node is the conditional root, then average the quality of the parents for the root
                    if node.is_conditional_root:
                        parent_qualities = [sum(parent_qualities) / len(parent_qualities)]

                    sample_quality_list = self.performance_profile.query_quality_list_on_interval(
                        time=node_time, id=node.id, parent_qualities=parent_qualities)

                    conditional_probability = self.performance_profile.query_probability_contract_expression(
                        queried_quality=possible_quality, quality_list=sample_quality_list)

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
                        total_sum += conditional_probability * utility
                    else:
                        total_sum += conditional_probability * self.find_exact_expected_utility(leaves=new_leaves, time_allocations=time_allocations, depth=depth,
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

    def naive_hill_climbing_outer(self, verbose=False, monitoring=False) -> List[float]:
        # This case is specifically for a contract program with conditionals, fors, and contracts
        if self.child_programs:
            return self.naive_hill_climbing_outer_main(verbose=verbose)

        # Check if it has child programs and what type of child programs
        elif self.child_programs and self.child_programs[0].subprogram_expression_type == "conditional":
            false_allocations = copy.deepcopy(self.child_programs[1].allocations)
            true_allocations = copy.deepcopy(self.child_programs[0].allocations)

            self.best_allocations_inner = [copy.deepcopy(self.child_programs[0].allocations),
                                           copy.deepcopy(self.child_programs[1].allocations)]

            return self.naive_hill_climbing_outer_conditional(true_allocations, false_allocations, verbose=verbose)

        elif self.child_programs and self.child_programs[0].subprogram_expression_type == "for":
            for_allocations = copy.deepcopy(self.child_programs[0].allocations)
            self.best_allocations_inner = [copy.deepcopy(self.child_programs[0].allocations)]

            return self.naive_hill_climbing_outer_for(for_allocations, verbose=verbose)

        else:
            return self.naive_hill_climbing_outer_main(verbose=verbose)

    # This case is specifically for a contract program with conditionals, fors, and contracts
    def naive_hill_climbing_outer_main(self, depth=0, decay=1.1, threshold=.01, verbose=False) -> List[float]:
        """
        Does hill climbing specific to an outer contract program with conditional subprograms

        :param verbose: Verbose mode
        :param threshold: float, the threshold of the temperature decay during annealing
        :param decay: float, the decay rate of the temperature during annealing
        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        # The initial true_allocations, false_allocations, and for_allocations have uniform alloations
        # Initialize the amount of time to be switched
        time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)
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
                # Avoids negative time allocation
                elif node_0.time - time_switched < 0:
                    continue
                # Avoids reducing the time allocation below the conditional time lower bound (tau)
                elif ((node_0.expression_type == "conditional") and (node_0.time - time_switched < self.performance_profile.calculate_tau())) or ((node_1.expression_type == "conditional") and (node_1.time + time_switched < self.performance_profile.calculate_tau())):
                    continue
                else:
                    original_program_dag_nodes = copy.deepcopy(self.program_dag.nodes)
                    node_0.time -= time_switched
                    node_1.time += time_switched
                    # Make a copy just in case recursvie hill climbing solution is suboptimal
                    for node in [node_0, node_1]:
                        # Does hill climbing on the outer metareasoning problem that is a conditional
                        # TODO: For emebedded subprograms, make the recursion a function of depth
                        if node.expression_type == "conditional" and depth == 0:
                            # Reallocate the budgets for the inner metareasoning problems
                            node.true_subprogram.change_budget(copy.deepcopy(node.time))
                            node.false_subprogram.change_budget(copy.deepcopy(node.time))
                            # Do recursive naive hill climbing on the branches
                            node.true_subprogram.naive_hill_climbing_main(depth=depth + 1)
                            node.false_subprogram.naive_hill_climbing_main(depth=depth + 1)
                            # Change the time allocations in the outer program
                            self.change_time_allocations(node.true_subprogram.program_dag.nodes)
                            self.change_time_allocations(node.false_subprogram.program_dag.nodes)
                        elif node.expression_type == "for" and depth == 0:
                            # Reallocate the budgets for the inner metareasoning problems
                            node.for_subprogram.change_budget(copy.deepcopy(node.time))
                            # Do recursive naive hill climbing on the loop
                            node.for_subprogram.naive_hill_climbing_main(depth=depth + 1)
                            # Change the time allocations in the outer program
                            self.change_time_allocations(node.for_subprogram.program_dag.nodes)

                    # Best allocations from the previous iterations are needed since the allocations of the subprograms may be adjusted from above
                    # And scale the EU for interprettable results
                    eu_adjusted = self.expected_utility() * self.scale

                    if eu_adjusted > eu_original:
                        best_allocations_changed = True
                        eu_original = eu_adjusted
                    else:
                        # Revert to the orginal time allocations before switch
                        self.change_time_allocations(original_program_dag_nodes)

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations_outer = [round(node.time, self.decimals) for node in self.program_dag.nodes]
                        printed_eu_adjusted = round(eu_adjusted, self.decimals)
                        printed_eu_original = round(eu_original, self.decimals)
                        temp_time_switched = round(time_switched, self.decimals)
                    if verbose:
                        message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
                        print(message.format(temp_time_switched, printed_eu_adjusted, printed_eu_original, print_allocations_outer))

            if not best_allocations_changed:
                time_switched = time_switched / decay

        return eu_original

    def naive_hill_climbing_inner(self, decay=1.1, threshold=.01, verbose=False) -> List[float]:
        """
        Does hill climbing specific to an arbitrary inner contract program

        :param verbose: Verbose mode
        :param threshold: float, the threshold of the temperature decay during annealing
        :param decay: float, the decay rate of the temperature during annealing
        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        # Reinitialize the inner metareasoning problem with a uniform budget
        self.allocations = self.initialize_allocations.uniform_budget()
        eu_original = self.expected_utility(self.allocations) * self.scale

        # Initialize the amount of time to be switched
        time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)

        while time_switched > threshold:
            # Remove the Nones in the list before taking permutations
            refactored_allocations = utils.remove_nones_time_allocations(self.allocations)
            best_allocations_changed = False

            # Go through all permutations of the time allocations
            for permutation in permutations(refactored_allocations, 2):
                previous_eu_original = eu_original
                node_0 = utils.find_node(permutation[0].node_id, self.program_dag)
                node_1 = utils.find_node(permutation[1].node_id, self.program_dag)

                # Makes a deep copy to avoid pointers to the same list
                adjusted_allocations = copy.deepcopy(self.allocations)

                # Avoids exchanging time with itself
                if permutation[0].node_id == permutation[1].node_id:
                    continue

                # Avoids all permutations that include the conditional node in the inner metareasoning problem
                elif node_0.expression_type == "conditional" or node_1.expression_type == "conditional":
                    continue

                # Avoids all permutations that include the for node in the inner metareasoning problem
                elif node_0.expression_type == "for" or node_1.expression_type == "for":
                    continue

                # Avoids negative time allocation
                elif adjusted_allocations[permutation[0].node_id].time - time_switched <= 0:
                    continue

                else:
                    adjusted_allocations[permutation[0].node_id].time -= time_switched
                    adjusted_allocations[permutation[1].node_id].time += time_switched

                    eu_adjusted = self.expected_utility(adjusted_allocations) * self.scale

                    if eu_adjusted > eu_original:
                        self.allocations = copy.deepcopy(adjusted_allocations)
                        best_allocations_changed = True
                        eu_original = eu_adjusted

                    adjusted_allocations = utils.remove_nones_time_allocations(adjusted_allocations)

                    print_allocations_outer = [i.time for i in adjusted_allocations]

                    temp_time_switched = time_switched

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations_outer = [round(i.time, self.decimals) for i in adjusted_allocations]
                        printed_eu_adjusted = round(eu_adjusted, self.decimals)
                        printed_eu_original = round(previous_eu_original, self.decimals)
                        temp_time_switched = round(temp_time_switched, self.decimals)

                    if verbose:
                        message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
                        print(message.format(temp_time_switched, printed_eu_adjusted, printed_eu_original,
                                             print_allocations_outer))

            # if local max wasn't found
            if not best_allocations_changed:
                time_switched = time_switched / decay

        return self.allocations

        # An attempt to make a recursive function for a more genralizable method
        # node_index = 0
        # depth = 0
        # while node_index < len(ppv_ordering):
        #     if isinstance(ppv[node_index], list):
        #         inner_node_index = 0
        #         while inner_node_index < ppv[node_index].length:
        #             inner_element = ppv[node_index][inner_node_index]
        #             if inner_element == "conditional":
        #                 proportional_allocations_outer.append(TimeAllocation(self.performance_profile.calculate_tau())
        #             elif inner_element == "for":
        #             inner_node_index += 1
        #         node_index += inner_node_index

        #     else:
        #         proportional_allocations_outer.append(TimeAllocation(node_index, order*budget_proportion))
        #     node_index += 1

    def proportional_allocation_tangent(self, phi=1) -> List[float]:
        number_conditionals_and_fors = utils.number_of_fors_conditionals(self.full_dag)
        number_conditionals = number_conditionals_and_fors[0]
        number_fors = number_conditionals_and_fors[1]

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

    def naive_hill_climbing_outer_conditional(self, true_allocations, false_allocations, decay=1.1, threshold=.01, verbose=False) -> List[float]:
        """
        Does hill climbing specific to an outer contract program with conditional subprograms

        :param verbose: Verbose mode
        :param threshold: float, the threshold of the temperature decay during annealing
        :param decay: float, the decay rate of the temperature during annealing
        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        # Initialize the amount of time to be switched
        time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)

        while time_switched > threshold:
            possible_local_max = []

            # Remove the Nones in the list before taking permutations
            refactored_allocations = utils.remove_nones_time_allocations(self.allocations)

            # Go through all permutations of the time allocations
            for permutation in permutations(refactored_allocations, 2):

                node_0 = utils.find_node(permutation[0].node_id, self.program_dag)
                node_1 = utils.find_node(permutation[1].node_id, self.program_dag)

                # Makes a deep copy to avoid pointers to the same list
                adjusted_allocations = copy.deepcopy(self.allocations)

                # Avoids exchanging time with itself
                if permutation[0].node_id == permutation[1].node_id:
                    continue

                # Avoids negative time allocation
                elif adjusted_allocations[permutation[0].node_id].time - time_switched < 0:
                    continue

                else:
                    adjusted_allocations[permutation[0].node_id].time -= time_switched
                    adjusted_allocations[permutation[1].node_id].time += time_switched

                    # Does hill climbing on the outer metareasoning problem that is a conditional
                    if node_0.expression_type == "conditional":
                        # Reallocate the budgets for the inner metareasoning problems
                        node_0.true_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_0.id].time))
                        node_0.false_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_0.id].time))

                        # Do naive hill climbing on the branches
                        true_allocations = copy.deepcopy(node_0.true_subprogram.naive_hill_climbing_inner(verbose=False))
                        false_allocations = copy.deepcopy(node_0.false_subprogram.naive_hill_climbing_inner())

                    if node_1.expression_type == "conditional":
                        # Reallocate the budgets for the inner metareasoning problems
                        node_1.true_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_1.id].time))
                        node_1.false_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_1.id].time))

                        # Do naive hill climbing on the branches
                        true_allocations = copy.deepcopy(node_1.true_subprogram.naive_hill_climbing_inner())
                        false_allocations = copy.deepcopy(node_1.false_subprogram.naive_hill_climbing_inner())

                    eu_adjusted = self.expected_utility(adjusted_allocations)
                    eu_original = self.expected_utility(self.allocations, self.best_allocations_inner, "conditional")

                    if eu_adjusted > eu_original:
                        possible_local_max.append([adjusted_allocations, true_allocations, false_allocations])

                    # scale the EUs
                    eu_adjusted *= self.scale
                    eu_original *= self.scale

                    adjusted_allocations = utils.remove_nones_time_allocations(adjusted_allocations)

                    print_allocations_outer = [i.time for i in adjusted_allocations]
                    temp_time_switched = time_switched

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations_outer = [round(i.time, self.decimals) for i in adjusted_allocations]

                        eu_adjusted = round(eu_adjusted, self.decimals)
                        eu_original = round(eu_original, self.decimals)

                        # self.expected_utility(self.allocations) * self.scale
                        temp_time_switched = round(temp_time_switched, self.decimals)

                    if verbose:
                        message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
                        print(message.format(temp_time_switched, eu_adjusted, eu_original, print_allocations_outer))

                    # Reset the branches of the inner conditional
                    if self.best_allocations_inner:
                        true_allocations = self.best_allocations_inner[0]
                        false_allocations = self.best_allocations_inner[1]

            # arg max here
            if possible_local_max:
                best_allocation = max([self.expected_utility(j[0]) for j in possible_local_max])
                for j in possible_local_max:
                    if self.expected_utility(j[0]) == best_allocation:
                        # Make a deep copy to avoid pointers to the same list
                        self.allocations = copy.deepcopy(j[0])

                        self.best_allocations_inner = [copy.deepcopy(j[1]), copy.deepcopy(j[2])]

            else:
                time_switched = time_switched / decay

        return [self.allocations, self.best_allocations_inner[0],
                self.best_allocations_inner[1]]

    def naive_hill_climbing_outer_for(self, for_allocations, decay=1.1, threshold=.01, verbose=False) -> List[float]:
        """
        Does hill climbing specific to an outer contract program with for subprograms

        :param verbose: Verbose mode
        :param threshold: float, the threshold of the temperature decay during annealing
        :param decay: float, the decay rate of the temperature during annealing
        :return: A stream of optimized time allocations associated with each contract algorithm
        """
        # Initialize the amount of time to be switched
        time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)

        while time_switched > threshold:

            possible_local_max = []
            # Remove the Nones in the list before taking permutations
            refactored_allocations = utils.remove_nones_time_allocations(self.allocations)

            # Go through all permutations of the time allocations
            for permutation in permutations(refactored_allocations, 2):
                node_0 = utils.find_node(permutation[0].node_id, self.program_dag)
                node_1 = utils.find_node(permutation[1].node_id, self.program_dag)

                # Makes a deep copy to avoid pointers to the same list
                adjusted_allocations = copy.deepcopy(self.allocations)

                # Avoids exchanging time with itself
                if permutation[0].node_id == permutation[1].node_id:
                    continue

                # Avoids negative time allocation
                elif adjusted_allocations[permutation[0].node_id].time - time_switched < 0:
                    continue

                else:
                    adjusted_allocations[permutation[0].node_id].time -= time_switched
                    adjusted_allocations[permutation[1].node_id].time += time_switched

                    # Does hill climbing on the outer metareasoning problem that is a conditional
                    if node_0.expression_type == "for":
                        # Reallocate the budgets for the inner metareasoning problems
                        node_0.for_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_0.id].time))

                        # Do naive hill climbing on the branches
                        for_allocations = copy.deepcopy(node_0.for_subprogram.naive_hill_climbing_inner())

                    if node_1.expression_type == "for":
                        # Reallocate the budgets for the inner metareasoning problems
                        node_1.for_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_1.id].time))

                        # Do naive hill climbing on the branches
                        for_allocations = copy.deepcopy(node_1.for_subprogram.naive_hill_climbing_inner())

                    # eu_adjusted = self.expected_utility(adjusted_allocations, best_allocations_inner=[for_allocations], expression_type="for")
                    eu_adjusted = self.expected_utility(adjusted_allocations)
                    eu_original = self.expected_utility(self.allocations, self.best_allocations_inner, expression_type="for")

                    if eu_adjusted > eu_original:
                        possible_local_max.append([adjusted_allocations, for_allocations])

                    # scale the EUs
                    eu_adjusted *= self.scale
                    eu_original *= self.scale

                    adjusted_allocations = utils.remove_nones_time_allocations(adjusted_allocations)

                    print_allocations_outer = [i.time for i in adjusted_allocations]
                    temp_time_switched = time_switched

                    # Check for rounding
                    if self.decimals is not None:
                        print_allocations_outer = [round(i.time, self.decimals) for i in adjusted_allocations]

                        eu_adjusted = round(eu_adjusted, self.decimals)
                        eu_original = round(eu_original, self.decimals)

                        temp_time_switched = round(temp_time_switched, self.decimals)

                    if verbose:
                        message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
                        print(message.format(temp_time_switched, eu_adjusted, eu_original, print_allocations_outer))

                    # Reset the branch of the inner for
                    if self.best_allocations_inner:
                        for_allocations = self.best_allocations_inner[0]

            # arg max here
            if possible_local_max:
                best_allocation = max([self.expected_utility(j[0]) for j in possible_local_max])
                for j in possible_local_max:
                    if self.expected_utility(j[0]) == best_allocation:
                        # Make a deep copy to avoid pointers to the same list
                        self.allocations = copy.deepcopy(j[0])

                        self.best_allocations_inner = [copy.deepcopy(j[1])]
            else:
                time_switched = time_switched / decay

        return [self.allocations, self.best_allocations_inner[0]]

    def change_budget(self, new_budget) -> None:
        """
        Changes the budget of the contract program and adjusts the objects that use the budget of the
        contract program

        :param new_budget: float
        :return: None
        """
        self.budget = new_budget
        self.initialize_allocations.budget = new_budget

    def change_time_allocations(self, receiving_program_nodes, giving_program_nodes):
        # TODO: optimize this using a hash table (2/26)
        for receiving_node in receiving_program_nodes:
            for giving_node in giving_program_nodes:
                if receiving_node.id == giving_node.id:
                    receiving_node.time = giving_node.time

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
