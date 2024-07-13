import random
import unittest

import numpy as np
import pandas as pd
import yaml

from alfa_ec_llm.problem_environments.symbolic_regression import \
    SymbolicRegressionGP


class TestSymbolicRegressionGP(unittest.TestCase):
    def test_call(self):
        solution = ["+", ["x0"], ["x0"]]
        param = {
            "fitness_cases_file": "tests/data/fitness_cases.csv",
            "test_train_split": 0.7,
            "arities": {
                "+": 2,
                "*": 2,
                # /: 2
                "-": 2,
            },
            "constants": {"values": "0, 1"},
        }
        sr = SymbolicRegressionGP(param)
        fitness = sr(solution, {})
        self.assertIsNotNone(fitness)
        print(solution, fitness)

        solutions = [
            ["+", ["+", ["1.0"], ["1.0"]], ["1.0"]],
            ["-", ["1.0"], ["+", ["1.0"], ["1.0"]]],
            ["+", ["*", ["x1"], ["1.0"]], ["x1"]],
            ["+", ["*", ["x0"], ["x0"]], ["*", ["x1"], ["x1"]]],
        ]
        state = random.getstate()
        for solution in solutions:
            fitness = sr(solution, {})
            self.assertIsNotNone(fitness)
            assert state == random.getstate()
            print(solution, fitness)
