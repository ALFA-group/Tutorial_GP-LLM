import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import yaml

from alfa_ec_llm.algorithms.tutorial_gp import TutorialGP
from alfa_ec_llm.algorithms.tutorial_llm_gp import TutorialLLMGPMuXo
from alfa_ec_llm.utils.evaluate_sr_solutions import \
    main as evaluate_sr_solutions_main

__author__ = "Erik Hemberg"


"""
Main function for alfa_ec_llm. Parses YML config file and calls algorithm.
"""


def parse_arguments(param: List[str]) -> Tuple[Dict[str, Any], argparse.Namespace]:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Tutorial_LLM-GP")
    parser.add_argument(
        "-f",
        "--configuration_file",
        type=str,
        required=True,
        help="YAML configuration file. E.g. " "configurations/tutorial_llm_gp_sr.yml",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Path to directory for output files. E.g. " "tutorial_llm_gp_output",
    )
    parser.add_argument(
        "--algorithm",
        choices=[
            "tutorial_gp",
            "tutorial_llm_gp_mu_xo",
        ],
        help="Algorithms",
    )

    _args = parser.parse_args(param)

    # Read configuration file
    with open(_args.configuration_file, "r", encoding="utf-8") as configuration_file:
        settings: Dict[str, Any] = yaml.safe_load(configuration_file)

    # Set CLI arguments in settings
    settings["output_dir"] = _args.output_dir
    settings["algorithm"] = _args.algorithm

    return settings, _args


def main(params: List[str]) -> Tuple[Dict[str, Any], argparse.Namespace]:
    """
    Run donkey_ge.
    """
    # Parse CLI arguments
    args, _cli_args = parse_arguments(params)
    logging.info(f"ARGS: {args}")
    # Run algorithm search
    if args["algorithm"] == "tutorial_gp":
        algorithm = TutorialGP()
    elif args["algorithm"] == "tutorial_llm_gp_mu_xo":
        algorithm = TutorialLLMGPMuXo()
    else:
        raise ValueError(f'Unknown algorithm: {args["algorithm"]}')

    logging.info(f"Use {algorithm.__class__.__name__}")
    algorithm.run(args)
    return args, _cli_args


def analyse(params: List[str]) -> None:
    # Analyse
    logging.info("Analyse results")
    output_directory = params["output_dir"]
    e_args = [
        "--configuration_file",
        params["configuration_file"],
        "--solutions_file",
        os.path.join(output_directory, "alfa_ec_llm_solution_values.json"),
        "--fitness_file",
        os.path.join(output_directory, "alfa_ec_llm_fitness_values.json"),
    ]
    df = evaluate_sr_solutions_main(e_args)
    n_generations = df["generation"].max()
    population_size = df["rank"].max() + 1
    k = min(5, population_size)
    best_solution_and_fitness = df[df["rank"] == 0][
        ["solution", "test_fitness", "train_fitness"]
    ].tail(1)
    logging.info(
        f"Results top {k} generation {n_generations}:\n{df[df['generation'] == n_generations].head(k)}"
    )
    logging.info(
        f"Best rank 0 generation {n_generations}:\n{best_solution_and_fitness}"
    )


if __name__ == "__main__":
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    _, cli_args = main(sys.argv[1:])
    analyse(vars(cli_args))
