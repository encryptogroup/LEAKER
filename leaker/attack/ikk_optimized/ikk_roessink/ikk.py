# Imports random library which is used to simulate randomness in selecting the new state of a Simulated Annealing algorithm
import random

# Imports warnings library which we use to catch a warning as an error
import warnings

# Imports math library used for specific mathematical operations (log, sqrt)
import math

# Imports numpy library which is used to calculate an exponent to compare the Euclidean distance between
#   the current state and the new state in the simulated annealing algorithm
import numpy as np

# Class IKK provides the user with all the tools necessary to execute the IKK attack
# Note: This implementation is an (optimized) implementation of the algorithm(s) described by Islam et al. (IKK) in
#   'Access Pattern disclosure on Searchable Encryption: Ramification, Attack and Mitigation'
# server_knowledge_index and background_knowledge_index (inverted indices) were added to implement the proposed 'deterministic' IKK attack,
#   these variables are not used in the original IKK attack
class IKK:

    # Method find_occurrence_dicts creates two dictionaries (one from the server knowledge index and one from the background knowledge index)
    #   for easy searching specific variables while executing the 'deterministic' IKK attack. As we generate these dictionaries once these
    #   calculations do not have to be done in the ANNEAL algorithm thereby ensuring no calculation to get the total occurrence of
    #   a query/keyword is executed more than once.
    def find_occurrence_dicts(self, server_knowledge_index, background_knowledge_index):
        delta = 0.05  # This might need to be a global parameter in future work
        num_server_knowledge_docs = len(server_knowledge_index.columns.tolist())
        num_background_knowledge_docs = len(background_knowledge_index.columns.tolist())
        numerator = num_server_knowledge_docs * math.log(2.0 / delta)
        epsilon = round(
            math.sqrt(numerator / 2.0)
        )  # Calculation taken by the implementation of Cash et al.'s paper 'Leakage-Abuse Attacks Against Searchable Encryption'

        # This part of the code generates a keyword occurrence dictionary containing:
        # - Background knowledge count - The amount of files in the background the keyword occurs in at least once
        # - Expected server knowledge count - Based on the (relative) difference in the amount of documents known to the server and
        #     the amount of documents in the actual (encrypted) server knowledge
        # - Lower_bound/Upper_bound - Calculates an interval around the expected server knowledge count with a confidence level of 95%
        keyword_occurrence_dict = {}
        for keyword, row in background_knowledge_index.iterrows():
            background_knowledge_count = sum([1 if x else 0 for x in list(row)])
            expected_server_knowledge_count = round(
                (background_knowledge_count / num_background_knowledge_docs)
                * num_server_knowledge_docs
            )
            lower_bound = max(expected_server_knowledge_count - epsilon, 0)
            upper_bound = min(
                expected_server_knowledge_count + epsilon, num_server_knowledge_docs
            )
            row_dict = {}
            row_dict["server_count"] = background_knowledge_count
            row_dict["expected_count"] = expected_server_knowledge_count
            row_dict["lower_bound"] = lower_bound
            row_dict["upper_bound"] = upper_bound
            keyword_occurrence_dict[keyword] = row_dict

        # This part of the code generates:
        # - An index (query_in_range_dict) of queries as keys and corresponding lists of keywords that are in (occurrence) range of each query as values
        # - An index (keyword_in_range_dict) of keywords as keys and corresponding lists of queries that are in (occurrence) range of each keyword as values
        query_in_range_dict = {}
        keyword_in_range_dict = {}
        for query, row in server_knowledge_index.iterrows():
            query_count = sum([1 if x else 0 for x in list(row)])
            list_of_keywords_within_range = [
                keyword
                for keyword, values in keyword_occurrence_dict.items()
                if values["lower_bound"] <= query_count
                and query_count <= values["upper_bound"]
            ]

            # The deterministic IKK attack will assign None to a query if no keywords are within its relative occurrence rate
            # This query will not be regarded anywhere in the system
            # The final mapping will include this query and the value None as no keyword could be assigned to it
            if list_of_keywords_within_range != []:
                query_in_range_dict[query] = list_of_keywords_within_range
                for keyword in list_of_keywords_within_range:
                    if keyword in keyword_in_range_dict.keys():
                        keyword_in_range_dict[keyword] = keyword_in_range_dict[
                            keyword
                        ] + [query]
                    else:
                        keyword_in_range_dict[keyword] = [query]
            else:
                query_in_range_dict[query] = None

        # Asserts all queries and all keywords are present in their corresponding lists
        # Asserts a mapping for every query will be returned (can be None though)
        assert len(query_in_range_dict) == len(server_knowledge_index.index)
        return query_in_range_dict, keyword_in_range_dict

    # Method deterministic_init_state is used to set the init state for the deterministic IKK algorithm
    #   (which takes the relative occurrence of keywords/queries into regard)
    # First all queries which have no keywords in range are assigned the None value
    # Then, all other queries are sorted on the amount of keywords they have in range in ascending order
    # If a query is encountered which has keywords in range, but they have already been assigned to other queries it tries to, with a depth of 1
    # See whether those queries can be re-assigned another keyword
    def deterministic_init_state(self, query_in_range_dict, value_list):

        # Return variable
        init_state = {}

        # Assign 'None' to all queries in that have no keywords in range in the initial mapping
        # During the program run queries that are mapped to None are disregarded
        # The loop below adds all queries with 1 or more keywords in range (in terms of occurrence count) in the temp variable query_occurrences
        query_occurrences = {}
        for query, keywords_in_range in query_in_range_dict.items():
            if keywords_in_range is None:
                init_state[query] = None
            elif keywords_in_range is not None:
                query_occurrences[query] = keywords_in_range

        # Sorts query_occurrences in increasing lengths of keywords in range (queries with a lower amount of keywords in range are mapped first)
        sorted_query_occurrences = sorted(
            query_occurrences.items(), key=lambda x: len(x[1])
        )

        # For all queries which were not assigned value None the algorithm assigns a random keyword in range
        for item in sorted_query_occurrences:
            query, keywords_in_range = item[0], item[1]

            # All keywords in range of the query which have not been assigned yet.
            possible_keywords = [
                keyword for keyword in keywords_in_range if keyword in value_list
            ]

            # If possible_keywords is empty the algorithm tries (with a depth of one) to assign an already assigned keyword to the query and to assign
            # another keyword to the other query
            if len(possible_keywords) == 0:
                assigned_possible_keywords = [
                    keyword
                    for keyword in keywords_in_range
                    if keyword in list(init_state.values())
                ]

                for i, option in enumerate(assigned_possible_keywords):
                    # The query which is currently mapped to the candidate keyword in the init_state variable
                    mapped_query = [
                        key for key, value in init_state.items() if value == option
                    ][0]
                    reassign_possible_keywords = [
                        keyword
                        for keyword in query_in_range_dict[mapped_query]
                        if keyword in value_list
                    ]
                    if len(reassign_possible_keywords) == 0:
                        # If we tried all other possible keywords the query cannot be assigned a keyword and is disregarded in the rest of the protocol run
                        if i == len(assigned_possible_keywords) - 1:
                            init_state[query] = None
                    elif len(reassign_possible_keywords) != 0:
                        new_assignment = random.choice(reassign_possible_keywords)
                        init_state[mapped_query] = new_assignment
                        init_state[query] = option
                        value_list.remove(new_assignment)

            # possible_keywords is not empty the algorithm randomly assigns one of the keywords to the query and removes it from value_list
            else:
                keyword = random.choice(possible_keywords)
                init_state[query] = keyword
                value_list.remove(keyword)

        assert len(init_state) == len(query_in_range_dict)

        # Returns initial state of the deterministic ikk setting
        return init_state

    # Method fill_init_state fills an initial state by assigning a unique 1-to-1 random mapping of for every query to a keyword
    # In the normal setting the keyword assigned to a query is chosen random
    # In the 'deterministic' IKK setting the keyword assigned to a query is within a certain occurrence rate of said query (relatively)
    def fill_init_state(
        self,
        init_state,
        value_list,
        variable_list,
        server_knowledge_index=None,
        background_knowledge_index=None,
    ):

        # Assigns a unique 1-to-1 random mapping of a query and a keyword in the simulation (num_keywords >= num_queries)
        if (
            server_knowledge_index is not None
            and background_knowledge_index is not None
        ):
            query_in_range_dict, keyword_in_range_dict = self.find_occurrence_dicts(
                server_knowledge_index, background_knowledge_index
            )
            init_state = self.deterministic_init_state(query_in_range_dict, value_list)
            return init_state, query_in_range_dict, keyword_in_range_dict
        else:
            for i, var in enumerate(variable_list):
                value = random.choice(value_list)
                init_state[var] = value
                value_list.remove(value)
            return init_state

    # Method optimizer randomly selects an initial state for the Simulated Annealing algorithm
    # If the deterministic_ikk variable is set the initial state is not chosen randomly, but based on the relative occurrence count of a single keyword
    def optimizer(
        self,
        server_knowledge_cooccurrence_matrix,
        background_knowledge_coocurrence_matrix,
        server_knowledge_index,
        background_knowledge_index,
        init_temperature,
        cooling_rate,
        reject_threshold,
        deterministic_ikk=False,
    ):

        # Known assignments is always empty in these simulations as we believe this to be most useful setting of the system
        # Currently this implementation does not give the user a method to fill the known_assignments variable
        known_assignments = {}

        # First the initial state includes already known_assignments
        init_state = known_assignments  # Currently always an empty dictionary ({})

        # Domain list contains all keywords in the background knowledge (index of background_knowledge_cooccurrence_matrix)
        domain_list = background_knowledge_coocurrence_matrix.index.tolist()
        # Not used right now, removes known assignments from domain_list
        domain_list = [
            x for x in domain_list if x not in list(known_assignments.values())
        ]

        # Ensures the domain list is not changed as values are removed from the variable value_list
        # This ensures no two queries are mapped to the same keyword in the initial state
        value_list = domain_list.copy()

        # Variable list contains all 'observed' queries from a Searchable Encryption scheme (index of server_knowledge_cooccurrence_matrix)
        variable_list = server_knowledge_cooccurrence_matrix.index.tolist()

        # Not used right now, removes known assignments from variable_list
        variable_list = [
            x for x in variable_list if x not in list(known_assignments.keys())
        ]

        if deterministic_ikk:
            (
                init_state,
                query_in_range_dict,
                keyword_in_range_dict,
            ) = self.fill_init_state(
                init_state,
                value_list,
                variable_list,
                server_knowledge_index=server_knowledge_index,
                background_knowledge_index=background_knowledge_index,
            )
            return self.ANNEAL(
                init_state,
                domain_list,
                server_knowledge_cooccurrence_matrix,
                background_knowledge_coocurrence_matrix,
                init_temperature,
                cooling_rate,
                reject_threshold,
                query_in_range_dict=query_in_range_dict,
                keyword_in_range_dict=keyword_in_range_dict,
            )
        else:
            init_state = self.fill_init_state(init_state, value_list, variable_list)
            return self.ANNEAL(
                init_state,
                domain_list,
                server_knowledge_cooccurrence_matrix,
                background_knowledge_coocurrence_matrix,
                init_temperature,
                cooling_rate,
                reject_threshold,
            )

    # Method remove_None_queries is used to remove None assigned queries in case of the deterministic IKK attack
    # These None assigned queries are regarded not assignable in our designed protocol
    def remove_None_queries(self, current_state):
        temp_current_state = current_state.copy()
        none_assigned_queries = []
        for query, keyword in current_state.items():
            if keyword is None:
                none_assigned_queries.append(query)
                temp_current_state.pop(query)
        return none_assigned_queries, temp_current_state

    # Method find_squared_Euclidean_distance was added as it is more efficient than the methodology as proposed by Islam et al. in their paper
    # To calculate whether a new state should be accepted, Islam et al. compare the total squared Euclidean distance of new mapping (new state) to
    #   the total squared Euclidean distance of the old state
    # To calculate these mappings Islam et al. calculate the squared Euclidean distance between the server_knowledge_cooccurrence_matrix and a subset of
    #   the background_knowledge_cooccurrence_matrix every iteration of
    # the while loop at a total complexity of O(n*2) where n denotes the amount of queries in the system.
    # As only 1 or 2 mappings are changed in the (simplified) version of the IKK attack we figured it would make more sense to compute
    #   the current cost once and then changing only the costs of the changed mappings
    # at a cost of at most O(2n)/O(n). This dramatically changes the runtime of the attack in this Python script.
    # Method find_squared_Euclidean_distance therefore calculates the squared Euclidean distance between two matrices once
    #   (with only a subset of the background_knowledge_cooccurrence_matrix considered)
    def find_squared_Euclidean_distance(
        self,
        current_state,
        server_knowledge_cooccurrence_matrix,
        background_knowledge_coocurrence_matrix,
    ):

        # The rows and columns of the server_knowledge_cooccurrence_matrix/the queries
        rows = list(server_knowledge_cooccurrence_matrix.index.tolist())
        columns = rows

        current_cost = 0
        for i, row in enumerate(rows):
            for j, column in enumerate(columns):
                k = current_state.get(row)
                l = current_state.get(column)

                # Queries which have been assigned None in the deterministic IKK setting
                # This should never occur in the original IKK setting
                if k is not None and l is not None:
                    current_cost += (
                        server_knowledge_cooccurrence_matrix.loc[row, column]
                        - background_knowledge_coocurrence_matrix.loc[k, l]
                    ) ** 2

        return current_cost

    # Method choose_new_state is used to choose a new candidate state for the Simulated Annealing algorithm. This can be done in two ways:
    # - IKK method:
    #   1) Randomly select a query, keyword (q_1:k_1 for simplicity sake) mapping (in the current state) to change
    #   2) Choose another keyword (k_2 for example) from the domain list (all keywords in the system) and assign this keyword to
    #     the query chosen in step 1 (q_1)
    #   3) The mapping (q_1:k_1) changes to (q_1:k_2) in the next state
    #   4) Check whether k_2 is already assigned to another query (q_2)
    #       If that is not the case:
    #           5) Return the next state
    #       If that is the case:
    #           5) Change mapping (q_2:k_2) to (q_2:k_1) in the next state
    #           6) Return the next state
    # - deterministic IKK method:
    #   In this method the total occurrences of queries/keywords are taken into regard (i.e. how many documents each query/keywords occurs in
    #     at least once, relative to the total amount of documents in a dataset)
    #   The selection of a new query/keyword is not completely random, but only taken from a subset of the queries/keywords that are within the right range
    #     (based on the Hoeffding Inequality as used by Cash et al. in their paper)
    def choose_new_state(
        self,
        current_state,
        reversed_current_state,
        domain_list,
        query_in_range_dict=None,
        keyword_in_range_dict=None,
    ):
        next_state = current_state.copy()
        reversed_next_state = reversed_current_state.copy()

        random_query, random_keyword = random.choice(list(next_state.items()))

        in_range_domain_list = domain_list

        if query_in_range_dict is not None:
            in_range_domain_list = query_in_range_dict[random_query]

        # Asserts the new_random_keyword variable is never the same as the random_keyword variable
        in_range_domain_list = [
            keyword for keyword in in_range_domain_list if keyword != random_keyword
        ]

        # In the unlikely case that there are no values in range for this query (without assigning the current mapped keyword)
        # The system returns the current state as the next state. As the Euclidean distance is the same, this means the 'next state' is rejected
        if len(in_range_domain_list) != 0:
            new_random_keyword = random.choice(in_range_domain_list)
        else:
            return current_state, reversed_current_state, random_query, None

        # Only in the deterministic IKK setting
        if query_in_range_dict is not None:
            count = 0
            # The while loop is used to find a 'new_random_keyword' to assign to the query and simulate a new state, by:
            #   1) Picking a keyword that was not mapped to a query already (reversed_current_state.get(new_random_keyword) is None
            #   2) Finding a keyword that was mapped to another query, but is in the right range
            while (
                reversed_current_state.get(new_random_keyword) is not None
                and not reversed_current_state.get(new_random_keyword)
                in keyword_in_range_dict[random_keyword]
            ):
                count += 1
                new_random_keyword = random.choice(in_range_domain_list)

                # If we tried twice as much possibilities as there are possible matches for a keyword we return the current state without
                #   changing a state.
                # As the next state and the current state have the same Euclidean distance the next state is rejected
                if count == 2 * len(keyword_in_range_dict[random_keyword]):
                    return current_state, reversed_current_state, random_query, None

        next_state[random_query] = new_random_keyword
        reversed_next_state[new_random_keyword] = random_query

        new_random_query = reversed_current_state.get(new_random_keyword)

        if new_random_query is None:
            reversed_next_state.pop(random_keyword)
            return next_state, reversed_next_state, random_query, None
        elif new_random_query is not None:
            next_state[new_random_query] = random_keyword
            reversed_next_state[random_keyword] = new_random_query
            return next_state, reversed_next_state, random_query, new_random_query

    # Method calculate cost change calculates the changes in (squared) Euclidean distance due to (a) changed mapping(s) between
    #   current state and next state
    # The complexity is O(2n) at most (if two mappings were changed) and O(n) if only one mapping was changed as opposed to the O(n^2) algorithm as
    #   described by Islam et al.
    def calculate_cost_change(
        self,
        current_state,
        next_state,
        server_knowledge_cooccurrence_matrix,
        background_knowledge_coocurrence_matrix,
        query_1,
        query_2,
    ):
        cost_change_old = 0
        cost_change_new = 0

        row = query_1
        k = current_state.get(row)
        k_prime = next_state.get(row)
        mappings_changed = {row: {"k": k, "k_prime": k_prime}}
        row_2 = query_2
        if row_2 is not None:
            k_2 = current_state.get(row_2)
            k_prime_2 = next_state.get(row_2)
            mappings_changed[row_2] = {"k": k_2, "k_prime": k_prime_2}

        for i, column in enumerate(list(current_state.keys())):
            l = current_state.get(column)
            l_prime = next_state.get(column)
            for row_identifier, corresponding_values in mappings_changed.items():
                server_knowledge_cell_value = server_knowledge_cooccurrence_matrix.loc[
                    row_identifier, column
                ]
                cost_change_old += (
                    server_knowledge_cell_value
                    - background_knowledge_coocurrence_matrix.loc[
                        corresponding_values["k"], l
                    ]
                ) ** 2
                cost_change_new += (
                    server_knowledge_cell_value
                    - background_knowledge_coocurrence_matrix.loc[
                        corresponding_values["k_prime"], l_prime
                    ]
                ) ** 2

        cost_change_old = (
            cost_change_old * 2
        )  # Column values and row values as the matrices are symmetric
        cost_change_new = cost_change_new * 2

        for row_identifier, corresponding_values in mappings_changed.items():
            for row_identifier_2, corresponding_values_2 in mappings_changed.items():
                cost_change_old -= (
                    server_knowledge_cooccurrence_matrix.loc[
                        row_identifier, row_identifier_2
                    ]
                    - background_knowledge_coocurrence_matrix.loc[
                        corresponding_values["k"], corresponding_values_2["k"]
                    ]
                ) ** 2
                cost_change_new -= (
                    server_knowledge_cooccurrence_matrix.loc[
                        row_identifier, row_identifier_2
                    ]
                    - background_knowledge_coocurrence_matrix.loc[
                        corresponding_values["k_prime"],
                        corresponding_values_2["k_prime"],
                    ]
                ) ** 2
        return cost_change_old, cost_change_new

    # Method accept_new_state determines whether a new state should be accepted or not
    # A next state is accepted if:
    # - The value of E is lower than 0, meaning that the (squared) Euclidean distance of the next state is lower than current state and therefore better
    # - With a (very) small probability exp(-E / current_temperature) is accepted (meaning that if we randomly pick a value between 0 and 1 and
    #   it is lower than the exp the next state is accepted
    def accept_new_state(self, E, current_temperature):
        accept_new_state = False
        if E < 0:
            accept_new_state = True
        else:
            # RuntimeWarnings are defined as errors instead of warnings (so we can catch them)
            warnings.filterwarnings("error")
            e = random.random()

            # Unfortunately it is sometimes the case that -E / current_temperature is lower than Python can express causing a RuntimeWarning
            # To catch this adequately (as this means the exponent nears -inf and is thus lower than e) we catch this using a try, except block
            try:
                if e < np.exp(-E / current_temperature):
                    accept_new_state = True
            except RuntimeWarning as e:
                pass
        return accept_new_state

    # Method merge_queries merges the queries that were assigned None and all other queries in current_state
    def merge_queries(self, none_assigned_queries, current_state):
        result = current_state.copy()
        for query in none_assigned_queries:
            result[query] = None
        return result

    # Method ANNEAL is the implementation of the Simulated Annealing algorithm proposed by Islam et al. (IKK)
    def ANNEAL(
        self,
        init_state,
        domain_list,
        server_knowledge_cooccurrence_matrix,
        background_knowledge_coocurrence_matrix,
        init_temperature,
        cooling_rate,
        reject_threshold,
        query_in_range_dict=None,
        keyword_in_range_dict=None,
    ):
        # The current_state variable of the ANNEAL method contains a mapping for every query in the system to a unique keyword
        # Every loop in the while loop changed this state by 1 or 2 mappings if the total Euclidean distance in terms of cooccurrence counts is
        #   better than the state before
        current_state = init_state.copy()

        # Only in the case of the deterministic IKK setting, splits queries assigned a None value and queries assigned a not None value
        none_assigned_queries = []
        if query_in_range_dict is not None and keyword_in_range_dict is not None:
            none_assigned_queries, current_state = self.remove_None_queries(
                current_state
            )

        # The reversed_current_state variable was added by us for efficiency sake (as we can easily find values and change them)
        # Adding a reversed_current_state variable and changing 1 or 2 values is far more efficient than
        #   generating the reversed current state every iteration in the while loop
        reversed_current_state = dict((v, k) for k, v in current_state.items())

        # The current_temperature variable contains the current temperature of the (Simulated) Annealing algorithm (it is initialized to
        #   the init_temperature value chosen by the user while starting the simulation
        current_temperature = init_temperature

        # sequential_rejects is a variable used by the Simulated Annealing algorithm to ensure the system will stop sooner if it does not find
        #   a better state (mapping between queries and keywords) for a long time
        # In most of our simulations the Simulated Annealing algorithm stops if the while loop does not find a better state 50000 times
        sequential_rejects = 0

        # The following variables are used for logging interesting information regarding the Simulated Annealing algorithm, in short:
        # - total_count logs the amount of loops the Simulated Annealing algorithm takes before returning, either because the current temperature approaches
        #   zero or the maximum rejection rate is met
        # - accepted_count logs the amount of loops where a new state was accepted
        total_count = 0
        accepted_count = 0

        # The following variables are used for logging interesting information regarding the Simulated Annealing algorithm, in short:
        # - E_min is the lowest (squared) Euclidean distance between the two matrices the system encounters
        #   (initialized at the highest possible (squared) Euclidean distance between two matrices (dimensions * 1))
        # - E_max is the highest (squared) Euclidean distance between the two matrices the system encounters
        #   (initialized at the lowest possible (squared Euclidean distance between two matrics (0))
        # - E_total_loops is the total (squared) Euclidean distance between the two matrices in every loop combined
        # - E_accepted_loops is the total (squared) Euclidean distance between the two matrices combined for every loops where a new state was accepted
        E_min = len(server_knowledge_cooccurrence_matrix.index.tolist()) ** 2
        E_max = 0
        E_total_loops = 0
        E_accepted_loops = 0

        current_cost = self.find_squared_Euclidean_distance(
            current_state,
            server_knowledge_cooccurrence_matrix,
            background_knowledge_coocurrence_matrix,
        )

        # The while loop below should stop if the current temperature reaches 0 (or if another condition is met)
        # The current temperature however is calculated by taking a percentage of the initial temperature every iteration of the while loops and
        #   both the initial temperature and this 'cool down rate' are expressed in floats
        # If have the number which is one number above 0.0, which is expressable in floats in Python,
        #   and you take for example 0.9 of it this number is closer to that number than to 0.0 and is thus rounded to the first number
        # Therefore the system stops at the approx_zero variable which is closest to 0.0 as we can get
        approx_zero = 2.465e-321

        # The system runs until the current temperature reaches approximately zero (fixed amount of loops, depending on the input parameters
        #   init_temperature and cool_down_rate) or a new state has not bee accepted for {{ reject_rate }} iterations of the while loop
        while (
            current_temperature > approx_zero and sequential_rejects < reject_threshold
        ):

            # Chooses next candidate state by changing one or two mappings as opposed to the current state
            # changed_query_2 can be None
            (
                next_state,
                reversed_next_state,
                changed_query_1,
                changed_query_2,
            ) = self.choose_new_state(
                current_state,
                reversed_current_state,
                domain_list,
                query_in_range_dict,
                keyword_in_range_dict,
            )

            # Calculates cost change of changing a mapping between the current state of the system and the (candidate) next state
            cost_change_old, cost_change_new = self.calculate_cost_change(
                current_state,
                next_state,
                server_knowledge_cooccurrence_matrix,
                background_knowledge_coocurrence_matrix,
                changed_query_1,
                changed_query_2,
            )
            next_cost = current_cost - cost_change_old + cost_change_new
            E = next_cost - current_cost

            # Checks if the next state should be accepted as it is better than the current state
            accept_new_state = self.accept_new_state(E, current_temperature)

            # If the next state is accepted as it has a lower (squared) Euclidean distance than:
            # - The next state is set as the current state for the next iteration of the while loop, the reversed_current_state variable is
            #   set accordingly as well
            # - The amount of sequential_rejects (current amount of consecutive iterations of the while loop where no new state has been accepted) is set to 0
            # - The current cost is set to the (squared) Euclidean distance of the mappings in the next state
            # - The logging variables are updated accordingly
            if accept_new_state:
                current_state = next_state.copy()
                reversed_current_state = reversed_next_state.copy()

                sequential_rejects = 0

                # Because of rounding errors the next cost might become a bit lower than 0 (and stay that way as the optimal solution is met)
                #   (only in the case where the server knowledge is an exact subset of the background knowledge). This if loop ensures this stays that way
                if next_cost < 0:
                    assert (
                        cost_change_new == 0.0
                    )  # Meaning we found the exact right mapping
                    next_cost = 0.0

                current_cost = next_cost

                # The following part of the code adds interesting information of this iteration to their corresponding variables
                accepted_count += 1
                E_accepted_loops += E

            # If the next (candidate) state is not accepted the sequential_rejects variable is incremented by one
            else:
                sequential_rejects += 1

            # The following part of the code adds interesting information of this iteration to their corresponding variables
            total_count += 1

            E_total_loops += E
            if E < E_min:
                E_min = E
            if E > E_max:
                E_max = E

            # The current_temperature is decreased using the cooling_rate
            current_temperature = cooling_rate * current_temperature

        # Merges queries that were split in None assigned queries and other queries at the end of the deterministic IKK protocol run
        if (
            query_in_range_dict is not None
            and keyword_in_range_dict is not None
            and len(none_assigned_queries) > 0
        ):
            current_state = self.merge_queries(none_assigned_queries, current_state)

        guess_of_E = {
            "max": E_max,
            "min": E_min,
            "avg_all": E_total_loops / total_count,
            "avg_accept": E_accepted_loops / accepted_count,
        }
        return (
            current_state,
            current_temperature,
            sequential_rejects,
            guess_of_E,
            total_count,
            accepted_count,
        )
