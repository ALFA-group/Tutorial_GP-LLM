import collections
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from alfa_ec_llm.algorithms.evolutionary_algorithm import (
    EvolutionaryAlgorithm, FitnessFunction, Individual, Population)
from alfa_ec_llm.utils.openai_interface import OpenAIInterface
from alfa_ec_llm.utils.utils import (get_fitness_function,
                                     write_run_output_gp_plus_llm)

""" Implementation of Genetic Programming(GP) plus Large Language
Model, the purpose of this code is to describe how the algorithm
works. The intended use is for teaching. The design is supposed to be
simple.
"""


class TutorialLLMGPMuXo(EvolutionaryAlgorithm):

    def mutation(
        self,
        individual: Individual,
        fitness_function: FitnessFunction,
        llm_interface: OpenAIInterface,
        generation_history: List[Tuple[str, str]],
        mutation_probability: float,
        samples: Optional[List[Any]] = None,
    ) -> List[Individual]:
        """
        Return a mutated individual
        """
        new_individual = Individual(individual.genome)
        new_individual.phenotype = individual.phenotype
        if random.random() < mutation_probability:
            prompt = fitness_function.form_prompt_rephrase_mutation(
                individual.phenotype, samples
            )
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "mutation"
            generation_history.append(response)
            phenotype = fitness_function.check_response_rephrase_mutation(
                response["content"], individual.phenotype
            )
            new_individual.phenotype = phenotype

        return new_individual

    def crossover(
        self,
        parents: List[Individual],
        fitness_function: FitnessFunction,
        llm_interface: OpenAIInterface,
        generation_history: List[Tuple[str, str]],
        crossover_probability: float,
        samples: Optional[List[Any]] = None,
    ) -> List[Individual]:
        """
        Return a crossed over individuals
        """
        children = []
        for individual in parents:
            new_individual = Individual(individual.genome)
            new_individual.phenotype = individual.phenotype
            children.append(new_individual)

        if random.random() < crossover_probability:
            prompt = fitness_function.form_prompt_crossover(
                [_.phenotype for _ in parents], samples
            )
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "crossover"
            generation_history.append(response)
            try:
                phenotypes = fitness_function.check_response_crossover(
                    response["content"], parents
                )
            except AssertionError as e:
                phenotypes = [_.phenotype for _ in parents]
                logging.error(
                    f"{e} from formatting response for crossover for {response['content']} given {phenotypes}"
                )

            for child, phenotype in zip(children, phenotypes):
                child.phenotype = phenotype

        return children

    def initialize_population(
        self,
        fitness_function: FitnessFunction,
        param: Dict[str, Any],
        llm_interface: OpenAIInterface,
        generation_history: List[Tuple[str, str]],
    ) -> List[Individual]:
        """
        LLM generates random individuals based on zero-shot (no additional information to the prompt) prompt.
        """

        individuals = []
        for i in range(param["population_size"]):
            individual = Individual(None)
            prompt = fitness_function.form_prompt_individual_generation()
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "initialize_population"
            generation_history.append(response)
            individual.phenotype = (
                fitness_function.check_response_individual_generation(
                    response["content"]
                )
            )
            # Append the individual to the population
            individuals.append(individual)

        return individuals

    def run(self, param: Dict[str, Any]) -> Individual:
        """
        Return the best solution. Create an initial
        population. Perform an evolutionary search.
        """
        if "seed" not in param.keys():
            param["seed"] = int(time.time())

        random.seed(param["seed"])
        logging.info(f"Setting random seed: {param['seed']} {random.random():.5f}")
        fitness_function = get_fitness_function(param["fitness_function"])

        llm_interface = OpenAIInterface()
        param["llm_interface"] = llm_interface
        generation_history = []
        param["generation_history"] = generation_history
        # Create population
        individuals = self.initialize_population(
            fitness_function, param, llm_interface, generation_history
        )
        population = Population(fitness_function, individuals)
        # Start evolutionary search
        best_ever = self.search_loop(population, param)

        return best_ever

    def search_loop(
        self, population: Population, param: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return the best individual from the evolutionary search
        loop. Starting from the initial population.
        """

        param["cache"] = collections.OrderedDict()
        start_time = time.time()
        stats: Dict[str, List[Any]] = collections.defaultdict(list)
        llm_interface = param["llm_interface"]
        generation_history = param["generation_history"]
        fitness_function = population.fitness_function

        ######################
        # Evaluate fitness
        ######################
        self.evaluate_fitness(
            population.individuals,
            fitness_function,
            param,
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

            ##################
            # Variation. Generate new individual solutions
            ##################

            # Crossover
            while len(new_individuals) < param["population_size"]:
                # Select parents
                _parents = random.sample(parents, 2)
                # Generate children by crossing over the parents
                children = self.crossover(
                    _parents,
                    fitness_function,
                    llm_interface,
                    generation_history,
                    param["crossover_probability"],
                    param["cache"],
                )
                # Append the children to the new population
                for child in children:
                    new_individuals.append(child)

            # Select population size individuals. Handles uneven population
            # sizes, since crossover returns 2 offspring
            new_individuals = new_individuals[: param["population_size"]]

            # Vary the population by mutation
            for i in range(len(new_individuals)):
                new_individuals[i] = self.mutation(
                    new_individuals[i],
                    fitness_function,
                    llm_interface,
                    generation_history,
                    param["mutation_probability"],
                    param["cache"],
                )

            ##################
            # Evaluate fitness
            ##################
            self.evaluate_fitness(new_individuals, fitness_function, param)

            ##################
            # Replacement. Replace individual solutions in the population
            ##################
            population.individuals = self.generational_replacement(
                new_individuals,
                population.individuals,
                elite_size=param["elite_size"],
                population_size=param["population_size"],
            )

            # Set best solution
            population.individuals = self.sort_population(
                population.individuals,
            )
            best_ever = population.individuals[0]

            # Print the stats of the population
            self.print_stats(generation, population.individuals, stats, start_time)

            # Increase the generation counter
            generation += 1

        write_run_output_gp_plus_llm(generation, stats, param, generation_history)
        return best_ever
