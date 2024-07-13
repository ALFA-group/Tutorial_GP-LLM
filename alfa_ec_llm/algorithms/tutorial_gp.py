import collections
import copy
import logging
import random
import time
from typing import Any, Dict, List, Tuple

from alfa_ec_llm.algorithms.evolutionary_algorithm import (
    EvolutionaryAlgorithm, FitnessFunction, Individual, Population)
from alfa_ec_llm.utils.utils import get_fitness_function, write_run_output

""" Implementation of Genetic Programming(GP), the purpose of this
code is to describe how the algorithm works. The intended use is for
teaching. The design is supposed to be simple.
"""

MAX_CNT = 100


class Tree:
    @staticmethod
    def append_node(node: List[Any], symbol: str) -> List[Any]:
        """
        Return the appended node. Append a symbol to the node.
        """

        # Create a list with the symbol and append it to the node
        new_node = [symbol]
        node.append(new_node)

        return new_node

    @staticmethod
    def grow(
        node: List[Any], depth: int, max_depth: int, full: bool, symbols: Dict[str, Any]
    ) -> None:
        """
        Recursively grow a node to max depth in a pre-order, i.e. depth-first
        left-to-right traversal.
        """
        # grow is called recursively in the loop. The loop iterates arity number
        # of times. The arity is given by the node symbol
        node_symbol = node[0]
        for _ in range(symbols["arities"][node_symbol]):
            # Get a random symbol
            symbol = Tree.get_random_symbol(depth, max_depth, symbols, full)
            # Create a child node and append it to the tree
            new_node = Tree.append_node(node, symbol)
            # if statement
            # Call grow with the child node as the current node
            Tree.grow(new_node, depth + 1, max_depth, full, symbols)

            assert len(node) == (_ + 2), len(node)

        assert depth <= max_depth, f"{depth} {max_depth}"

    @staticmethod
    def get_children(node: List[Any]) -> List[Any]:
        """
        Return the children of the node. The children are all the elements of the
        except the first
        """
        # Take a slice of the list except the head
        return node[1:]

    @staticmethod
    def get_number_of_nodes(root: List[Any], cnt: int) -> int:
        """
        Return the number of nodes in the tree. A recursive depth-first
        left-to-right search is done
        """

        # Increase the count
        cnt += 1
        # Iterate over the children
        for child in Tree.get_children(root):
            # Recursively count the child nodes
            cnt = Tree.get_number_of_nodes(child, cnt)

        return cnt

    @staticmethod
    def get_node_at_index(root: List[Any], idx: int) -> List[Any]:
        """
        Return the node in the tree at a given index. The index is
        according to a depth-first left-to-right ordering.
        """

        # Stack of unvisited nodes
        unvisited_nodes = [root]
        # Initial node is the same as the root
        node = root
        # Set the current index
        cnt = 0
        # Iterate over the tree until the index is reached
        while cnt <= idx and unvisited_nodes:
            # Take an unvisited node from the stack
            node = unvisited_nodes.pop()
            # Add the children of the node to the stack
            # Get the children
            children = Tree.get_children(node)
            # Reverse the children before appending them to the stack
            children.reverse()
            # Add children to the stack
            unvisited_nodes.extend(children)

            # Increase the current index
            cnt += 1

        return node

    @staticmethod
    def get_max_tree_depth(root: List[Any], depth: int, max_tree_depth: int) -> int:
        """
        Return the max depth of the tree. Recursively traverse the tree
        """

        # Update the max depth if the current depth is greater
        if max_tree_depth < depth:
            max_tree_depth = depth

        # Traverse the children of the root node
        for child in Tree.get_children(root):
            # Increase the depth
            depth += 1
            # Recursively get the depth of the child node
            max_tree_depth = Tree.get_max_tree_depth(child, depth, max_tree_depth)
            # Decrease the depth
            depth -= 1

        assert depth <= max_tree_depth

        return max_tree_depth

    @staticmethod
    def get_depth_at_index(
        node: List[Any], idx: int, node_idx: int, depth: int, idx_depth: int = 0
    ) -> Tuple[int, int]:
        """
        Return the depth of a node based on the index. The index is based on
        depth-first left-to-right traversal.

        TODO implement breakout
        """
        # Assign the current depth when the current index matches the given index
        if node_idx == idx:
            idx_depth = depth

        idx += 1
        # Iterate over the children
        for child in Tree.get_children(node):
            # Increase the depth
            depth += 1
            # Recursively check the child depth and node index
            idx_depth, idx = Tree.get_depth_at_index(
                child, idx, node_idx, depth, idx_depth
            )
            # Decrease the depth
            depth -= 1

        return idx_depth, idx

    @staticmethod
    def replace_subtree(new_subtree: List[Any], old_subtree: List[Any]) -> None:
        """
        Replace a subtree.
        """

        # Delete the nodes of the old subtree
        del old_subtree[:]
        for node in new_subtree:
            # Insert the nodes in the new subtree
            old_subtree.append(copy.deepcopy(node))

    @staticmethod
    def get_random_symbol(
        depth: int, max_depth: int, symbols: Dict[str, str], full: bool = False
    ) -> str:
        """
        Return a randomly chosen symbol. The depth determines if a terminal
        must be chosen. If `full` is specified a function will be chosen
        until the max depth. The symbol is picked with a uniform probability.
        """

        # Pick a terminal if max depth has been reached
        if depth >= (max_depth - 1):
            # Pick a random terminal
            symbol = random.choice(symbols["terminals"])
        else:
            # Can it be a terminal before the max depth is reached
            # then there is 50% chance that it is a terminal
            if not full and bool(random.getrandbits(1)):
                # Pick a random terminal
                symbol = random.choice(symbols["terminals"])
            else:
                # Pick a random function
                symbol = random.choice(symbols["functions"])

        # Return the picked symbol
        return symbol


class TutorialGP(EvolutionaryAlgorithm):
    def evaluate_individual(
        self,
        individual: Individual,
        fitness_function: FitnessFunction,
        cache: Dict[str, float],
    ):
        assert len(individual.genome) > 0
        individual.fitness = fitness_function(individual.genome, cache)

    def initialize_population(self, param: Dict[str, Any]) -> List[Individual]:
        """
        Ramped half-half initialization. The individuals in the
        population are initialized using the grow or the full method for
        each depth value (ramped) up to max_depth.
        """

        individuals = []
        for i in range(param["population_size"]):
            individual = Individual(None)
            # Pick full or grow method
            full = bool(random.getrandbits(1))
            # Ramp the depth
            max_depth = (i % param["max_depth"]) + 1
            # Create root node
            symbol = Tree.get_random_symbol(0, max_depth, param["symbols"])
            tree = [symbol]
            # Grow the tree if the root is a function symbol
            if max_depth > 0 and symbol in param["symbols"]["functions"]:
                Tree.grow(tree, 1, max_depth, full, param["symbols"])
                assert Tree.get_max_tree_depth(tree, 0, 0) < (max_depth + 1)

            individual.genome = tree
            individual.phenotype = tree
            # Append the individual to the population
            individuals.append(individual)

        return individuals

    def evaluate_fitness(
        self,
        individuals: List[Individual],
        fitness_function: FitnessFunction,
        param: Dict[str, Any],
    ):
        """
        Evaluation each individual of the population.
        """

        # Iterate over all the individual solutions
        for ind in individuals:
            # Execute the fitness function
            self.evaluate_individual(ind, fitness_function, param["cache"])

    def search_loop(
        self, population: Population, param: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return the best individual from the evolutionary search
        loop. Starting from the initial population.
        """

        param["cache"] = collections.OrderedDict()
        param["xo_attempts"] = 0
        param["xo_max_cnt"] = 0
        param["mu_attempts"] = 0
        param["mu_identical"] = 0
        start_time = time.time()
        stats: Dict[str, List[Any]] = collections.defaultdict(list)

        ######################
        # Evaluate fitness
        ######################
        self.evaluate_fitness(
            population.individuals, population.fitness_function, param
        )
        # Print the stats of the population
        self.print_stats(0, population.individuals, stats, start_time)
        # Set best solution
        population.individuals = self.sort_population(population.individuals)
        best_ever = population.individuals[0]

        ######################
        # Generation loop
        ######################
        generation = 1
        while generation < param["generations"]:
            new_individuals = []
            ##################
            # Selection
            ##################
            parents = self.tournament_selection(
                population.individuals,
                param["population_size"],
                param["tournament_size"],
            )
            assert all([len(_.genome) for _ in population.individuals])
            ##################
            # Variation. Generate new individual solutions
            ##################

            # Crossover
            while len(new_individuals) < param["population_size"]:
                # Select parents
                _parents = random.sample(parents, 2)
                # Generate children by crossing over the parents
                children = self.subtree_crossover(_parents[0], _parents[1], param)
                # Append the children to the new population
                for child in children:
                    new_individuals.append(child)

            # Select population size individuals. Handles uneven population
            # sizes, since crossover returns 2 offspring
            new_individuals = new_individuals[: param["population_size"]]
            assert all([len(_.genome) for _ in new_individuals])

            # Vary the population by mutation
            for i in range(len(new_individuals)):
                new_individuals[i] = self.subtree_mutation(new_individuals[i], param)

            assert all([len(_.genome) for _ in new_individuals])
            ##################
            # Evaluate fitness
            ##################
            self.evaluate_fitness(new_individuals, population.fitness_function, param)

            ##################
            # Replacement. Replace individual solutions in the population
            ##################
            population.individuals = self.generational_replacement(
                new_individuals,
                population.individuals,
                population_size=param["population_size"],
                elite_size=param["elite_size"],
            )

            # Set best solution
            self.sort_population(population.individuals)
            best_ever = population.individuals[0]
            assert all([len(_.genome) for _ in new_individuals])

            # Print the stats of the population
            self.print_stats(generation, population.individuals, stats, start_time)

            # Increase the generation counter
            generation += 1

        write_run_output(generation, stats, param)

        return best_ever

    def subtree_mutation(
        self, individual: Individual, param: Dict[str, Any]
    ) -> Individual:
        """Subtree mutation. Pick a node and grow it."""

        # Copy the individual for mutation
        # Check if mutation should be applied
        if random.random() < param["mutation_probability"]:
            cnt = 0
            while cnt < MAX_CNT:
                new_individual = copy.deepcopy(Individual(individual.genome))
                # Pick node
                end_node_idx = Tree.get_number_of_nodes(new_individual.genome, 0) - 1
                node_idx = random.randint(0, end_node_idx)
                node = Tree.get_node_at_index(new_individual.genome, node_idx)

                # Clear children
                node_depth = Tree.get_depth_at_index(
                    new_individual.genome, 0, node_idx, 0
                )[0]
                assert node_depth <= param["max_depth"]

                new_symbol = Tree.get_random_symbol(
                    node_depth, param["max_depth"] - node_depth, param["symbols"]
                )
                new_subtree = [new_symbol]
                # Grow tree
                Tree.grow(
                    new_subtree, node_depth, param["max_depth"], False, param["symbols"]
                )
                Tree.replace_subtree(new_subtree, node)
                cnt += 1
                param["mu_attempts"] += 1
                if str(new_individual.genome) != str(individual.genome):
                    break

            if str(new_individual.genome) == str(individual.genome):
                param["mu_identical"] += 1

        else:
            new_individual = Individual(individual.genome)

        new_individual.phenotype = new_individual.genome

        return new_individual

    def subtree_crossover(
        self, parent_1: Individual, parent_2: Individual, param: Dict[str, Any]
    ) -> Tuple[Individual, Individual]:
        """
        Returns two individuals. The individuals are created by
        selecting two random nodes from the parents and swapping the
        subtrees.
        """
        # Copy the parents to make offsprings
        offsprings = (
            copy.deepcopy(Individual(parent_1.genome)),
            copy.deepcopy(Individual(parent_2.genome)),
        )
        # Check if offspring will be crossed over
        if random.random() < param["crossover_probability"]:
            cnt = 0
            while cnt < MAX_CNT:
                xo_nodes = []
                node_depths = []

                # Pick a crossover point
                end_node_idx = Tree.get_number_of_nodes(offsprings[0].genome, 0) - 1
                node_idx = random.randint(0, end_node_idx)
                # Find the subtree at the crossover point
                xo_nodes.append(Tree.get_node_at_index(offsprings[0].genome, node_idx))
                xo_point_depth = Tree.get_max_tree_depth(xo_nodes[-1], 0, 0)
                offspring_depth = Tree.get_max_tree_depth(offsprings[0].genome, 0, 0)
                node_depths.append((xo_point_depth, offspring_depth))
                # print(end_node_idx, node_idx, xo_point_depth, offspring_depth)

                # Pick a crossover point
                end_node_idx = Tree.get_number_of_nodes(offsprings[1].genome, 0) - 1
                node_idx = random.randint(0, end_node_idx)
                # Find the subtree at the crossover point
                xo_nodes.append(Tree.get_node_at_index(offsprings[1].genome, node_idx))
                xo_point_depth = Tree.get_max_tree_depth(xo_nodes[-1], 0, 0)
                offspring_depth = Tree.get_max_tree_depth(offsprings[1].genome, 0, 0)
                node_depths.append((xo_point_depth, offspring_depth))
                # print(end_node_idx, node_idx, xo_point_depth, offspring_depth)
                param["xo_attempts"] += 1

                # Make sure that the offspring is deep enough
                child_1_depth_ = node_depths[0][1] + node_depths[1][0]
                child_2_depth_ = node_depths[1][1] + node_depths[0][0]
                # print(child_1_depth_, child_2_depth_)
                cnt += 1
                if (
                    child_1_depth_ < param["max_depth"]
                    and child_2_depth_ < param["max_depth"]
                ):
                    break

            if (
                child_1_depth_ > param["max_depth"]
                or child_2_depth_ > param["max_depth"]
            ):
                param["xo_max_cnt"] += 1
                return offsprings

            # Swap the nodes
            tmp_offspring_1_node = copy.deepcopy(xo_nodes[1])
            # Copy the children from the subtree of the first offspring
            # to the chosen node of the second offspring
            Tree.replace_subtree(xo_nodes[0], xo_nodes[1])
            # Copy the children from the subtree of the second offspring
            # to the chosen node of the first offspring
            Tree.replace_subtree(tmp_offspring_1_node, xo_nodes[0])

            for offspring in offsprings:
                assert (
                    Tree.get_max_tree_depth(offspring.genome, 0, 0)
                    <= param["max_depth"]
                )
                offspring.phenotype = offspring.genome
                offspring.phenotype = offspring.genome

        # Return the offsprings
        return offsprings

    def run(self, param: Dict[str, Any]) -> Individual:
        """
        Return the best solution. Create an initial
        population. Perform an evolutionary search.
        """
        seed = param["seed"]
        # Set random seed if not 0 is passed in as the seed
        if seed != 0:
            random.seed(seed)

        logging.info(f"Setting random seed: {param['seed']} {random.random():.5f}")

        fitness_function = get_fitness_function(param["fitness_function"])
        param = self.setup(param, fitness_function)
        # Create population
        individuals = self.initialize_population(param)
        population = Population(fitness_function, individuals)
        # Start evolutionary search
        best_ever = self.search_loop(population, param)

        return best_ever

    def out_of_sample_test(
        self,
        individual: Individual,
        fitness_cases: List[List[float]],
        targets: List[float],
    ) -> None:
        """
        Out-of-sample test on an individual solution.
        """
        self.evaluate_individual(individual.genome, fitness_cases, targets)
        print("Best solution on test data: individual")

    def setup(
        self, args: Dict[str, Any], fitness_function: FitnessFunction
    ) -> Dict[str, Any]:
        """Wrapper for set up."""
        # Set arguments
        # Get the namespace dictionary
        args["symbols"] = fitness_function.symbols

        # Print GP settings
        print(f"GP settings:\n{args}")

        return args