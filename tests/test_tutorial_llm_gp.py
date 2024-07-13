import logging
import os
import unittest

import yaml

from alfa_ec_llm.algorithms.evolutionary_algorithm import Individual
from alfa_ec_llm.algorithms.tutorial_llm_gp import TutorialLLMGPMuXo
from alfa_ec_llm.problem_environments.symbolic_regression import \
    SymbolicRegressionGPPlusSomeLLMConstrainedFewShot
from alfa_ec_llm.utils.openai_interface import OpenAIInterface

log_file = os.path.basename(__file__).replace(".py", ".log")
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)


class TestTutorialLLMGP(unittest.TestCase):
    def setUp(self):
        self.param_path = "tests/configurations/tutorial_llm_gp_sr.yml"
        with open(self.param_path, "r", encoding="utf-8") as configuration_file:
            self.param = yaml.safe_load(configuration_file)

    def test_evaluate_fitness_hold_out(self):
        individuals = [Individual(None), Individual(None), Individual(None)]
        individuals[0].phenotype = "x0**2 + x1**2"
        individuals[1].phenotype = "x0 * x0 + x1 * x1"
        individuals[2].phenotype = "5 + 2 * 3"
        fitness_function = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot(
            self.param["fitness_function"]
        )
        self.param["cache"] = {}
        fitness_function.fitness_cases = fitness_function.hold_out["fitness_cases"]
        fitness_function.targets = fitness_function.hold_out["targets"]
        algorithm = TutorialLLMGPMuXo()

        population = algorithm.evaluate_fitness(
            individuals, fitness_function, self.param
        )
        self.assertIsNotNone(population)
        print(list(map(str, population)))

    def test_evaluate_fitness(self):
        individuals = [Individual(None), Individual(None), Individual(None)]
        individuals[0].phenotype = "x0**2 + x1**2"
        individuals[1].phenotype = "x0 * x0 + x1 * x1"
        individuals[2].phenotype = "5 + 2 * 3"
        fitness_function = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot(
            self.param["fitness_function"]
        )
        self.param["cache"] = {}
        algorithm = TutorialLLMGPMuXo()

        population = algorithm.evaluate_fitness(
            individuals, fitness_function, self.param
        )
        self.assertIsNotNone(population)
        print(list(map(str, population)))

    def test_mutation(self):
        fitness_function = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot(
            self.param["fitness_function"]
        )
        individual = Individual(None)
        individual.phenotype = "lambda x: x and x"
        individual.fitness = 2.0
        llm_interface = OpenAIInterface()
        generation_history = []
        algorithm = TutorialLLMGPMuXo()

        population = algorithm.mutation(
            individual, fitness_function, llm_interface, generation_history, 1.0
        )
        self.assertIsNotNone(population)
        print(population)
        print(generation_history)

    def test_crossover(self):
        fitness_function = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot(
            self.param["fitness_function"]
        )
        individuals = [Individual(None), Individual(None)]
        individuals[0].phenotype = "lambda x, y: x or y"
        individuals[0].fitness = 2.0
        individuals[1].phenotype = "lambda x, y: y if x > y else x"
        individuals[1].fitness = 1.0
        llm_interface = OpenAIInterface()
        generation_history = []
        algorithm = TutorialLLMGPMuXo()

        population = algorithm.crossover(
            individuals,
            fitness_function,
            llm_interface,
            generation_history,
            self.param["crossover_probability"],
        )
        self.assertIsNotNone(population)
        print(list(map(str, population)))

        individuals = [Individual(None), Individual(None)]
        individuals[0].phenotype = "lambda x, y: x or y"
        individuals[0].fitness = 2.0
        individuals[1].phenotype = "lambda x, y: y and not x"
        individuals[1].fitness = 1.0
        population = algorithm.crossover(
            individuals,
            fitness_function,
            llm_interface,
            generation_history,
            self.param["crossover_probability"],
        )
        self.assertIsNotNone(population)
        print(list(map(str, population)))
        print(generation_history)

    def test_initialize_population(self):
        fitness_function = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot(
            self.param["fitness_function"]
        )
        llm_interface = OpenAIInterface()
        generation_history = []
        algorithm = TutorialLLMGPMuXo()
        population = algorithm.initialize_population(
            fitness_function, self.param, llm_interface, generation_history
        )
        self.assertIsNotNone(population)
        print(list(map(str, population)))
        print(generation_history)
