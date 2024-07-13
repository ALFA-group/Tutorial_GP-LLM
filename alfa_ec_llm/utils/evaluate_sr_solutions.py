import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import yaml

from alfa_ec_llm.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from alfa_ec_llm.problem_environments.symbolic_regression import (
    SymbolicRegressionGP, SymbolicRegressionGPPlusSomeLLMConstrainedFewShot)
from alfa_ec_llm.utils.utils import get_fitness_function

# Evaluate Symbolic Regression Solutions. Use for e.g. hold-out-evaluation of solutions


def parse_args(args: List[str]) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run Symbolic Regression solutions on hold-out data"
    )
    parser.add_argument(
        "--configuration_file",
        type=str,
        required=True,
        help="YAML configuration file. E.g. tests/configurations/experiments/experiment_gp_sr.yml",
    )
    parser.add_argument(
        "--solutions_file",
        type=str,
        required=True,
        help="JSON file with solutions. E.g. tmp/heuristic_gp_gp_sr/0_heuristic_gp_gp_sr/alfa_ec_llm_solution_values.json",
    )
    parser.add_argument(
        "--fitness_file",
        type=str,
        required=True,
        help="JSON file with fitness. E.g. tmp/heuristic_gp_gp_sr/0_heuristic_gp_gp_sr/alfa_ec_llm_fitness_values.json",
    )
    _args = parser.parse_args(args)
    return _args


def main(args: List[str]) -> pd.DataFrame:
    params = parse_args(args)
    logging.info(f"BEGIN {params}")
    with open(params.configuration_file, "r", encoding="utf-8") as configuration_file:
        configs = yaml.safe_load(configuration_file)

    if configs.get("seed", False):
        configs["seed"] = int(time.time())

    random.seed(configs["seed"])
    logging.info(f"Setting random seed: {configs['seed']} {random.random():.5f}")

    # TODO guarantee the split
    fitness_function = get_fitness_function(configs["fitness_function"])
    if not isinstance(fitness_function, SymbolicRegressionGP):
        configs["fitness_function"]["n_shots"] = 0
        fitness_function = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot(
            configs["fitness_function"]
        )

    data_path = "tests/data/hold_out.csv"
    hold_out_data = np.loadtxt(data_path, comments="#", delimiter=",", dtype=np.float32)
    # TODO hack
    fitness_function.fitness_cases = hold_out_data[
        :, list(range(hold_out_data.shape[1]))[:-1]
    ]
    fitness_function.targets = hold_out_data[:, -1]
    out_path = os.path.split(params.solutions_file)[0]
    with open(params.solutions_file, "r", encoding="utf-8") as fd:
        solutions = json.load(fd)["solution_values"]

    with open(params.fitness_file, "r", encoding="utf-8") as fd:
        train_fitness = json.load(fd)["fitness_values"]

    cache = {}
    results = []
    for i, iteration in enumerate(solutions):
        for j, solution in enumerate(iteration):
            if not isinstance(fitness_function, SymbolicRegressionGP):
                fitness = fitness_function(solution, cache)
            else:
                fitness = evaluate_gp_tree(fitness_function, solution, cache)

            result = {
                "generation": i,
                "rank": j,
                "solution": solution,
                "test_fitness": fitness,
                "train_fitness": train_fitness[i][j],
            }
            results.append(result)

    df = pd.DataFrame(results)
    out_name = f"holdout_{os.path.basename(params.fitness_file)}l"
    out_file = os.path.join(out_path, out_name)
    df.to_json(out_file, orient="records", lines=True)
    logging.info(f"Analysed results in: {out_file}")
    return df


def evaluate_gp_tree(
    fitness_function: Callable, solution: str, cache: Dict[str, float]
) -> float:
    gp_solution = []
    fitness_function.to_node_list(solution, gp_solution)
    if len(gp_solution) > 0:
        gp_solution = gp_solution[0]
        fitness = fitness_function(gp_solution, cache)
    else:
        fitness = EvolutionaryAlgorithm.DEFAULT_FITNESS
        logging.warning(f"{solution}, {gp_solution}, {fitness}")

    return fitness


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    main(sys.argv[1:])
