# from src.classes.contract_program import *


# class SolutionMethods(ContractProgram):
#     def __init__(self, program_id, parent_program, child_programs, program_dag, budget, scale, decimals,
#                  quality_interval, time_interval, time_step_size, in_child_contract_program, generator_dag, allocations,
#                  performance_profile, initialize_allocations):

#         self.program_id = program_id
#         self.program_dag = program_dag
#         self.budget = budget
#         self.scale = scale
#         self.decimals = decimals
#         self.quality_interval = quality_interval
#         self.time_interval = time_interval
#         self.time_step_size = time_step_size

#         self.in_child_contract_program = in_child_contract_program

#         # Pointer to the parent program that the subprogram is an induced subgraph of
#         self.parent_program = parent_program
#         self.child_programs = child_programs

#         self.generator_dag = generator_dag

#         self.allocations = allocations

#         self.original_allocations_conditional_branches = None

#         self.performance_profile = performance_profile
#         self.initialize_allocations = initialize_allocations

#     def naive_hill_climbing_no_children_no_parents(self, decay=1.1, threshold=.01, verbose=False) -> List[float]:
#         """
#         Does outer naive hill climbing search by randomly replacing a set amount of time s between two different contract
#         algorithms. If the expected value of the root node of the contract algorithm increases, we commit to the
#         replacement; else, we divide s by a decay rate and repeat the above until s reaches some threshold by which we
#         terminate.

#         :param verbose: Verbose mode
#         :param threshold: float, the threshold of the temperature decay during annealing
#         :param decay: float, the decay rate of the temperature during annealing
#         :return: A stream of optimized time allocations associated with each contract algorithm
#         """
#         # Initialize the amount of time to be switched
#         time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)

#         while time_switched > threshold:

#             possible_local_max = []

#             # Remove the Nones in the list before taking permutations
#             refactored_allocations = utils.remove_nones_time_allocations(self.allocations)

#             # Go through all permutations of the time allocations
#             for permutation in permutations(refactored_allocations, 2):

#                 # node_0 = utils.find_node(permutation[0].node_id, self.program_dag)
#                 # node_1 = utils.find_node(permutation[1].node_id, self.program_dag)

#                 # Makes a deep copy to avoid pointers to the same list
#                 adjusted_allocations = copy.deepcopy(self.allocations)

#                 # Avoids exchanging time with itself
#                 if permutation[0].node_id == permutation[1].node_id:
#                     continue

#                 # Avoids negative time allocation
#                 elif adjusted_allocations[permutation[0].node_id].time - time_switched < 0:
#                     continue

#                 else:
#                     adjusted_allocations[permutation[0].node_id].time -= time_switched
#                     adjusted_allocations[permutation[1].node_id].time += time_switched

#                     if self.global_expected_utility(adjusted_allocations) > self.global_expected_utility(
#                             self.allocations):
#                         possible_local_max.append(adjusted_allocations)

#                     eu_adjusted = self.global_expected_utility(adjusted_allocations) * self.scale
#                     eu_original = self.global_expected_utility(self.allocations) * self.scale

#                     adjusted_allocations = utils.remove_nones_time_allocations(adjusted_allocations)

#                     print_allocations_outer = [i.time for i in adjusted_allocations]

#                     temp_time_switched = time_switched

#                     # Check for rounding
#                     if self.decimals is not None:
#                         # utils.print_allocations(adjusted_allocations)
#                         print_allocations_outer = [round(i.time, self.decimals) for i in adjusted_allocations]

#                         eu_adjusted = round(eu_adjusted, self.decimals)
#                         eu_original = round(eu_original, self.decimals)

#                         temp_time_switched = round(temp_time_switched, self.decimals)

#                     if verbose:
#                         message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
#                         print(message.format(temp_time_switched, eu_adjusted, eu_original, print_allocations_outer))

#             # arg max here
#             if possible_local_max:
#                 best_allocation = max([self.global_expected_utility(j) for j in possible_local_max])
#                 for j in possible_local_max:
#                     if self.global_expected_utility(j) == best_allocation:
#                         # Make a deep copy to avoid pointers to the same list
#                         self.allocations = copy.deepcopy(j)

#             # if local max wasn't found
#             else:
#                 time_switched = time_switched / decay

#         return self.allocations

#     def naive_hill_climbing_outer(self, decay=1.1, threshold=.01, verbose=False) -> List[float]:
#         """
#         Does outer naive hill climbing search by randomly replacing a set amount of time s between two different contract
#         algorithms. If the expected value of the root node of the contract algorithm increases, we commit to the
#         replacement; else, we divide s by a decay rate and repeat the above until s reaches some threshold by which we
#         terminate.

#         :param verbose: Verbose mode
#         :param threshold: float, the threshold of the temperature decay during annealing
#         :param decay: float, the decay rate of the temperature during annealing
#         :return: A stream of optimized time allocations associated with each contract algorithm
#         """
#         if self.child_programs:
#             true_allocations = copy.deepcopy(self.child_programs[0].allocations)
#             false_allocations = copy.deepcopy(self.child_programs[1].allocations)

#             self.original_allocations_conditional_branches = [copy.deepcopy(self.child_programs[0].allocations),
#                                                               copy.deepcopy(self.child_programs[1].allocations)]

#         else:
#             return self.naive_hill_climbing_no_children_no_parents()

#         # Initialize the amount of time to be switched
#         time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)

#         while time_switched > threshold:

#             possible_local_max = []

#             # Remove the Nones in the list before taking permutations
#             refactored_allocations = utils.remove_nones_time_allocations(self.allocations)

#             # Go through all permutations of the time allocations
#             for permutation in permutations(refactored_allocations, 2):

#                 node_0 = utils.find_node(permutation[0].node_id, self.program_dag)
#                 node_1 = utils.find_node(permutation[1].node_id, self.program_dag)

#                 # Makes a deep copy to avoid pointers to the same list
#                 adjusted_allocations = copy.deepcopy(self.allocations)

#                 # Avoids exchanging time with itself
#                 if permutation[0].node_id == permutation[1].node_id:
#                     continue

#                 # Avoids negative time allocation
#                 elif adjusted_allocations[permutation[0].node_id].time - time_switched < 0:
#                     continue

#                 else:
#                     adjusted_allocations[permutation[0].node_id].time -= time_switched
#                     adjusted_allocations[permutation[1].node_id].time += time_switched

#                     # Does hill climbing on the outer metareasoning problem that is a conditional
#                     if node_0.expression_type == "conditional":
#                         # Reallocate the budgets for the inner metareasoning problems
#                         node_0.true_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_0.id].time))
#                         node_0.false_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_0.id].time))

#                         # Do naive hill climbing on the branches
#                         true_allocations = copy.deepcopy(
#                             node_0.true_subprogram.naive_hill_climbing_inner(verbose=False))
#                         false_allocations = copy.deepcopy(node_0.false_subprogram.naive_hill_climbing_inner())

#                     if node_1.expression_type == "conditional":
#                         # Reallocate the budgets for the inner metareasoning problems
#                         node_1.true_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_1.id].time))
#                         node_1.false_subprogram.change_budget(copy.deepcopy(adjusted_allocations[node_1.id].time))

#                         # Do naive hill climbing on the branches
#                         true_allocations = copy.deepcopy(node_1.true_subprogram.naive_hill_climbing_inner())
#                         false_allocations = copy.deepcopy(node_1.false_subprogram.naive_hill_climbing_inner())

#                     # TODO: make a pointer from an element of the list of time allocations to a pointer to the left and right time allocations for conditional time allocations in the outer program
#                     if self.global_expected_utility(adjusted_allocations) > self.global_expected_utility(
#                             self.allocations, self.original_allocations_conditional_branches):
#                         possible_local_max.append([adjusted_allocations, true_allocations, false_allocations])

#                     eu_adjusted = self.global_expected_utility(adjusted_allocations) * self.scale
#                     eu_original = self.global_expected_utility(self.allocations,
#                                                                self.original_allocations_conditional_branches) * self.scale

#                     adjusted_allocations = utils.remove_nones_time_allocations(adjusted_allocations)

#                     print_allocations_outer = [i.time for i in adjusted_allocations]
#                     temp_time_switched = time_switched

#                     # Check for rounding
#                     if self.decimals is not None:
#                         print_allocations_outer = [round(i.time, self.decimals) for i in adjusted_allocations]

#                         eu_adjusted = round(eu_adjusted, self.decimals)
#                         eu_original = round(eu_original, self.decimals)

#                         # self.global_expected_utility(self.allocations) * self.scale
#                         temp_time_switched = round(temp_time_switched, self.decimals)

#                     if verbose:
#                         message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
#                         print(message.format(temp_time_switched, eu_adjusted, eu_original, print_allocations_outer))

#                     # Reset the branches of the inner conditional
#                     if self.original_allocations_conditional_branches:
#                         true_allocations = self.original_allocations_conditional_branches[0]
#                         false_allocations = self.original_allocations_conditional_branches[1]

#             # arg max here
#             if possible_local_max:

#                 best_allocation = max(
#                     [self.global_expected_utility(j[0], [j[1], j[2]]) for j in possible_local_max])

#                 for j in possible_local_max:
#                     if self.global_expected_utility(j[0], [j[1], j[2]]) == best_allocation:
#                         # Make a deep copy to avoid pointers to the same list
#                         self.allocations = copy.deepcopy(j[0])
#                         ContractProgram.change_allocations(self.allocations)
#                         self.original_allocations_conditional_branches = [
#                             copy.deepcopy(j[1]),
#                             copy.deepcopy(j[2])
#                         ]

#             else:
#                 time_switched = time_switched / decay

#         return [self.allocations, self.original_allocations_conditional_branches[0],
#                 self.original_allocations_conditional_branches[1]]

#     def naive_hill_climbing_inner(self, decay=1.1, threshold=.01, verbose=False) -> List[float]:
#         """
#         Does inner naive hill climbing search on one of the branches of a conditional by randomly replacing a set
#         amount of time s between two different contract algorithms. If the expected value of the root node of the
#         contract algorithm increases, we commit to the replacement; else, we divide s by a decay rate and repeat the
#         above until s reaches some threshold by which we terminate.

#         :param verbose: Verbose mode
#         :param threshold: float, the threshold of the temperature decay during annealing
#         :param decay: float, the decay rate of the temperature during annealing
#         :return: A stream of optimized time allocations associated with each contract algorithm
#         """
#         # Check that the net budget doesn't go negative if taxed with tau for inner
#         tau = self.performance_profile.calculate_tau()
#         taxed_budget = self.budget - tau

#         # Check if the budget is 0 for the inner metareasoning
#         if taxed_budget > 0:

#             # Reinitialize the inner metareasoning problem with a uniform budget
#             self.allocations = self.initialize_allocations.uniform_budget()

#             # Initialize the amount of time to be switched
#             time_switched = self.initialize_allocations.find_uniform_allocation(self.budget)

#             while time_switched > threshold:

#                 possible_local_max = []

#                 # Remove the Nones in the list before taking permutations
#                 refactored_allocations = utils.remove_nones_time_allocations(self.allocations)

#                 # Go through all permutations of the time allocations
#                 for permutation in permutations(refactored_allocations, 2):

#                     node_0 = utils.find_node(permutation[0].node_id, self.program_dag)
#                     node_1 = utils.find_node(permutation[1].node_id, self.program_dag)

#                     # Makes a deep copy to avoid pointers to the same list
#                     adjusted_allocations = copy.deepcopy(self.allocations)

#                     # Avoids exchanging time with itself
#                     if permutation[0].node_id == permutation[1].node_id:
#                         continue

#                     # Avoids all permutations that include the conditional node in the inner metareasoning problem
#                     elif node_0.expression_type == "conditional" or node_1.expression_type == "conditional":
#                         continue

#                     # Avoids negative time allocation
#                     elif adjusted_allocations[permutation[0].node_id].time - time_switched < 0:
#                         continue

#                     else:
#                         adjusted_allocations[permutation[0].node_id].time -= time_switched
#                         adjusted_allocations[permutation[1].node_id].time += time_switched

#                         # TODO: make a pointer from an element of the list of time allocations to a pointer to the left and right time allocations for conditional time allocations in the outer program
#                         if self.global_expected_utility(adjusted_allocations) > self.global_expected_utility(
#                                 self.allocations):
#                             possible_local_max.append(adjusted_allocations)

#                         eu_adjusted = self.global_expected_utility(adjusted_allocations) * self.scale
#                         eu_original = self.global_expected_utility(self.allocations) * self.scale

#                         adjusted_allocations = utils.remove_nones_time_allocations(adjusted_allocations)

#                         print_allocations_outer = [i.time for i in adjusted_allocations]

#                         temp_time_switched = time_switched

#                         # Check for rounding
#                         if self.decimals is not None:
#                             print_allocations_outer = [round(i.time, self.decimals) for i in adjusted_allocations]

#                             eu_adjusted = round(eu_adjusted, self.decimals)
#                             eu_original = round(eu_original, self.decimals)

#                             # self.global_expected_utility(self.allocations) * self.scale
#                             temp_time_switched = round(temp_time_switched, self.decimals)

#                         if verbose:
#                             message = "Amount of time switched: {:<12} ==> EU(adjusted): {:<12} EU(original): {:<12} ==> Allocations: {}"
#                             print(message.format(temp_time_switched, eu_adjusted, eu_original,
#                                                  print_allocations_outer))

#                 # arg max here
#                 if possible_local_max:
#                     best_allocation = max([self.global_expected_utility(j) for j in possible_local_max])
#                     for j in possible_local_max:
#                         if self.global_expected_utility(j) == best_allocation:
#                             # Make a deep copy to avoid pointers to the same list
#                             self.allocations = copy.deepcopy(j)

#                 # if local max wasn't found
#                 else:
#                     time_switched = time_switched / decay

#             return self.allocations

#         else:

#             for time_allocation in self.allocations:
#                 if time_allocation.time is not None and not Node.is_conditional_node(
#                         utils.find_node(time_allocation.node_id, self.program_dag)):
#                     time_allocation.time = 0

#                 return self.allocations
