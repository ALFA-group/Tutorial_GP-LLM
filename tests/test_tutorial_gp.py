import unittest

from alfa_ec_llm.algorithms.evolutionary_algorithm import Individual
from alfa_ec_llm.algorithms.tutorial_gp import TutorialGP


class TestTutorialGP(unittest.TestCase):
    def test_subtree_crossover(self):
        parent1 = Individual(genome=["+", ["x0"], ["x0"]])
        parent2 = Individual(genome=["*", ["1.0"], ["1.0"]])
        param = {
            "crossover_probability": 1.0,
            "max_depth": 1,
            "xo_attempts": 0,
            "xo_max_cnt": 0,
        }
        algorithm = TutorialGP()
        children = algorithm.subtree_crossover(parent1, parent2, param)
        self.assertIsNotNone(children)
        for child in children:
            print(child.genome)

        parent1 = Individual(genome=["x0"])
        parent2 = Individual(genome=["+", ["1.0"], ["1.0"]])
        param = {
            "crossover_probability": 1.0,
            "max_depth": 1,
            "xo_attempts": 0,
            "xo_max_cnt": 0,
        }
        children = algorithm.subtree_crossover(parent1, parent2, param)
        self.assertIsNotNone(children)
        for child in children:
            print(child.genome)
