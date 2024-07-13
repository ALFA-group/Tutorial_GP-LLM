import copy
import logging
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from alfa_ec_llm.utils.utils import FitnessFunction, get_ave_and_std

__author__ = "Erik Hemberg"

"""Evolutionray Algorithm

"""


class Individual:
    """An EA individual

    Attributes:
        codon_size: Max integer value for an inputs element
        max_length: Length of inputs
        DEFAULT_PHENOTYPE:

    """

    max_length: int = 1000
    DEFAULT_PHENOTYPE = ""

    def __init__(self, genome: Optional[List[int]]) -> None:
        """

        :param genome: Input representation
        :type genome: list of int or None
        """
        assert Individual.max_length > 0, f"max_length {Individual.max_length}"

        if genome is None:
            self.genome: List[int] = [
                random.randint(0, 100000) for _ in range(Individual.max_length)
            ]
        else:
            self.genome = genome

        self.fitness: float = EvolutionaryAlgorithm.DEFAULT_FITNESS
        self.phenotype: str = Individual.DEFAULT_PHENOTYPE
        self.used_input: int = 0

        # TODO create a map function for genome to phenotype?

    def get_fitness(self) -> float:
        """
        Return individual fitness
        """
        return self.fitness

    def __str__(self) -> str:
        return f"Ind: {self.phenotype}; {self.get_fitness()}"


class Population:
    """A population container

    Attributes:
        fitness_function:
        individuals:
    """

    def __init__(
        self,
        fitness_function: Any,
        individuals: List[Individual],
    ) -> None:
        """Container for a population.

        :param fitness_function:
        :type fitness_function: function
        :param individuals:
        :type individuals: list of Individual
        """
        self.fitness_function = fitness_function
        self.individuals = individuals

    def __str__(self) -> str:
        individuals = "\n".join(map(str, self.individuals))
        _str = f"{str(self.fitness_function)}\n{individuals}"

        return _str


class EvolutionaryAlgorithm:
    DEFAULT_FITNESS: float = -float("inf")

    def __init__(self):
        pass

    def print_stats(
        self,
        generation: int,
        individuals: List[Individual],
        stats: Dict[str, List[Any]],
        start_time: float,
    ) -> None:
        """
        Print the statistics for the generation and population.
        """

        # Make sure individuals are sorted
        individuals = self.sort_population(individuals)
        # Get the fitness values
        fitness_values: Sequence[float] = [i.get_fitness() for i in individuals]
        # Get the max length
        length_values: Sequence[float] = [float(len(i.phenotype)) for i in individuals]
        # Get average and standard deviation of fitness
        ave_fit, std_fit = get_ave_and_std(fitness_values)
        # Get average and standard deviation of max length
        ave_length, std_length = get_ave_and_std(length_values)
        # Print the statistics
        info = "Gen:{} t:{:.3f} fit_ave:{:.2f}+-{:.3f} length_ave:{:.2f}+-{:.3f} {}".format(
            generation,
            time.time() - start_time,
            ave_fit,
            std_fit,
            ave_length,
            std_length,
            individuals[0],
        )

        logging.info(info)

        stats["fitness_values"].append(fitness_values)
        stats["length_values"].append(length_values)
        stats["solution_values"].append([_.phenotype for _ in individuals])

    def tournament_selection(
        self, population: List[Individual], population_size: int, tournament_size: int
    ) -> List[Individual]:
        """
        Return individuals from a population by drawing
        `tournament_size` competitors randomly and selecting the best
        of the competitors. `population_size` number of tournaments are
        held.
        """
        assert tournament_size > 0
        assert tournament_size <= len(
            population
        ), f"{tournament_size} > {len(population)}"

        # Iterate until there are enough tournament winners selected
        winners: List[Individual] = []
        while len(winners) < population_size:
            # Randomly select tournament size individual solutions
            # from the population.
            competitors = random.sample(population, tournament_size)
            # Rank the selected solutions
            competitors = self.sort_population(competitors)
            # Append the best solution to the winners
            winners.append(competitors[0])

        assert len(winners) == population_size

        return winners

    def sort_population(self, individuals: List[Individual]) -> List[Individual]:
        """
        Return a list sorted on the fitness value of the individuals in
        the population. Descending order.
        """

        # Sort the individual elements on the fitness
        individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)

        return individuals

    def generational_replacement(
        self,
        new_population: List[Individual],
        old_population: List[Individual],
        elite_size: int,
        population_size: int,
    ) -> List[Individual]:
        """
        Return a new population. The `elite_size` best old_population
        are appended to the new population.

        # TODO the number of calls to sort_population can be reduced
        """
        assert len(old_population) == len(new_population) == population_size
        assert 0 <= elite_size < population_size

        # Sort the population
        old_population = self.sort_population(old_population)
        # Append a copy of the elite_size of the old population to
        # the new population.
        for ind in old_population[:elite_size]:
            # TODO is this deep copy redundant
            new_population.append(copy.deepcopy(ind))

        # Sort the new population
        new_population = self.sort_population(new_population)

        # Set the new population size
        new_population = new_population[:population_size]
        assert len(new_population) == population_size

        return new_population

    def validate_settings(self, param: Dict[str, Any]) -> None:
        # Print settings
        logging.info(f"Settings:{param}")

        assert param["population_size"] > 1
        assert param["generations"] > 0
        assert param["seed"] > -1
        assert param["tournament_size"] <= param["population_size"]
        assert param["elite_size"] < param["population_size"]
        assert 0.0 <= param["crossover_probability"] <= 1.0
        assert 0.0 <= param["mutation_probability"] <= 1.0

    def evaluate_fitness(
        self,
        individuals: List[Individual],
        fitness_function: FitnessFunction,
        param: Dict[str, Any],
    ) -> List[Individual]:
        """Perform the fitness evaluation for each individual of the population."""
        cache = param["cache"]
        # Iterate over all the individual solutions
        for ind in individuals:
            # Execute the fitness function
            ind.fitness = fitness_function(ind.phenotype, cache)

        return individuals

    def search_loop(self, population: Population, param: Dict[str, Any]) -> Dict[str, Any]:
        # TODO make search loop
        raise NotImplementedError("Not yet implemented")
