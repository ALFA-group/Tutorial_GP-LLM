import collections
import csv
import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import sympy

from alfa_ec_llm.algorithms.evolutionary_algorithm import (
    EvolutionaryAlgorithm, Individual)


class SymbolicRegressionGP:
    def __init__(self, param: Dict[str, Any]):
        logging.info(param)
        self.fitness_cases_file = param["fitness_cases_file"]
        self.test_train_split = param["test_train_split"]
        self.execution_environment = SymbolicRegressionGPInterpreter(param)
        (
            self.test,
            self.train,
            self.hold_out,
        ) = self.execution_environment.get_test_and_train_data(
            self.fitness_cases_file, self.test_train_split
        )

        self.fitness_cases = self.train["fitness_cases"]
        self.targets = self.train["targets"]
        param = self.execution_environment.parse_gp_configs(param)
        self.arities = self.execution_environment.get_arities(param)
        self.symbols = self.execution_environment.get_symbols(self.arities)
        self.outdir = 'tmp'
        os.makedirs(self.outdir, exist_ok=True)

    def to_node_list(self, strings: List[str], node: List[Any]) -> None:
        if len(strings) < 1:
            return

        node_symbol = strings[0]
        new_node = [node_symbol]
        node.append(new_node)
        if len(strings) == 1:
            return

        for child in strings[1:]:
            self.to_node_list(child, new_node)

    def __call__(self, solution: Dict[str, Any], cache: Dict[str, float]) -> float:
        """
        Evaluate fitness based on fitness cases and target values. Fitness
        cases are a set of exemplars (input and output points) by
        comparing the error between the output of an individual(symbolic
        expression) and the target values.
        Evaluates and sets the fitness in an individual. Fitness is the
        negative mean square error(MSE).
        """
        # The string representation of the tree is the cache key
        key = str(solution)
        if key in cache.keys():
            fitness = cache[key]
        else:
            data = []
            # Initial fitness value
            fitness = 0.0
            # Calculate the error between the output of the individual solution and
            # the target for each input
            assert len(self.fitness_cases) == len(self.targets)
            for case, target in zip(self.fitness_cases, self.targets):
                # Get output from evaluation function
                output = self.execution_environment.evaluate(solution, case)
                # Get the squared error
                error = output - target
                fitness += error * error
                data.append(
                    {
                        "case": case,
                        "target": target,
                        "output": output,
                        "error": error,
                        "fitness": fitness,
                        "solution": str(solution),
                    }
                )

            assert fitness >= 0
            # Get the mean fitness and assign it to the individual
            fitness = -fitness / float(len(self.targets))
            assert fitness <= 0
            cache[key] = fitness
            df = pd.DataFrame(data)
            out_path = os.path.join(self.outdir, "exemplar_calc.jsonl")
            df.to_json(out_path, orient="records", lines=True)

        return fitness


class SymbolicRegressionGPInterpreter:
    def __init__(self, params: Dict[str, Any]):
        self.HOLD_OUT_RATIO = 0.8
        self.HOLD_OUT_SEED = 112

    def split_data(
        self, exemplars: List[Any], targets: List[Any], data_split: float
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        split_idx = int(math.floor(len(exemplars) * data_split))
        # Randomize
        idx = list(range(0, len(exemplars)))
        random.shuffle(idx)
        training_cases = []
        training_targets = []
        test_cases = []
        test_targets = []
        for i in idx[:split_idx]:
            training_cases.append(exemplars[i])
            training_targets.append(targets[i])

        for i in idx[split_idx:]:
            test_cases.append(exemplars[i])
            test_targets.append(targets[i])

        assert all(map(len, (test_cases, test_targets, training_cases, training_targets)))  # type: ignore

        return training_cases, training_targets, test_cases, test_targets

    def get_hold_out(
        self, exemplars: List[Any], targets: List[Any]
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        rnd_state = random.getstate()
        random.seed(self.HOLD_OUT_SEED)
        (
            training_cases,
            training_targets,
            hold_out_cases,
            hold_out_targets,
        ) = self.split_data(exemplars, targets, self.HOLD_OUT_RATIO)
        random.setstate(rnd_state)
        assert random.getstate() == rnd_state
        out_path = "tests/data/hold_out.json"
        with open(out_path, "w") as fd:
            json.dump(
                {
                    "training_cases": training_cases,
                    "training_targets": training_targets,
                    "hold_out_cases": hold_out_cases,
                    "hold_out_targets": hold_out_targets,
                },
                fd,
            )

        return training_cases, training_targets, hold_out_cases, hold_out_targets

    def get_test_and_train_data(
        self, fitness_cases_file: str, test_train_split: float
    ) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
        """Return test and train data. Random selection or exemplars(ros)
        from file containing data.
        """

        exemplars, targets = self.parse_exemplars(fitness_cases_file)
        exemplars, targets, hold_out_cases, hold_out_targets = self.get_hold_out(
            exemplars, targets
        )
        training_cases, training_targets, test_cases, test_targets = self.split_data(
            exemplars, targets, test_train_split
        )
        return (
            {"fitness_cases": test_cases, "targets": test_targets},
            {"fitness_cases": training_cases, "targets": training_targets},
            {"fitness_cases": hold_out_cases, "targets": hold_out_targets},
        )

    def evaluate(self, node: List[Any], case: List[float]) -> float:
        """
        Evaluate a node recursively. The node's symbol string is evaluated.
        """
        symbol = node[0]
        symbol = symbol.strip()

        # Identify the node symbol
        if symbol == "+":
            # Add the values of the node's children
            return self.evaluate(node[1], case) + self.evaluate(node[2], case)

        elif symbol == "-":
            # Subtract the values of the node's children
            return self.evaluate(node[1], case) - self.evaluate(node[2], case)

        elif symbol == "*":
            # Multiply the values of the node's children
            return self.evaluate(node[1], case) * self.evaluate(node[2], case)

        elif symbol == "/":
            # Divide the value's of the nodes children. Too low values of the
            # denominator returns the numerator
            numerator = self.evaluate(node[1], case)
            denominator = self.evaluate(node[2], case)
            if abs(denominator) < 0.00001:
                denominator = 1

            return numerator / denominator

        elif symbol.startswith("x"):
            # Get the variable value
            return case[int(symbol[1:])]
        else:
            # The symbol is a constant
            return float(symbol)

    def parse_exemplars(self, file_name: str) -> Tuple[List[List[float]], List[float]]:
        """
        Parse a CSV file. Parse the fitness case and split the data into
        Test and train data. In the fitness case file each row is an exemplar
        and each dimension is in a column. The last column is the target value of
        the exemplar.
        """

        # Open file
        with open(file_name, "r") as in_file:
            # Create a CSV file reader
            reader = csv.reader(in_file, skipinitialspace=True, delimiter=",")

            # Read the header
            headers = reader.__next__()

            # Store fitness cases and their target values
            fitness_cases = []
            targets = []
            for row in reader:
                # Parse the columns to floats and append to fitness cases
                fitness_cases.append(list(map(float, row[:-1])))
                # The last column is the target
                targets.append(float(row[-1]))

            logging.info(
                f"Reading: {file_name} headers: {headers} exemplars: {len(targets)}"
            )

        return fitness_cases, targets

    def get_arities(self, param: Dict[str, Any], outputs: int = 1) -> Dict[str, int]:
        """Assign values to arities dictionary. Variables are taken from
        fitness case file header. Constants are read from config file.
        """

        assert outputs > 0

        arities = param["arities"]
        with open(param["fitness_cases_file"], "r") as csvFile:
            reader = csv.reader(csvFile, delimiter=",")
            # Read the header in order to define the variable arities as 0.
            headers = reader.__next__()

        # Remove comment symbol for the first element, i.e. #x0
        headers[0] = headers[0][1:]
        # Input variables are all
        variables = headers[:-outputs]
        for variable in variables:
            arities[variable.strip()] = 0

        # Add constant values
        constants = param["constants"]
        if constants:
            for constant in constants:
                arities[str(constant)] = 0

        assert len(arities) > 0
        return arities

    def get_symbols(self, arities: Dict[str, int]) -> Dict[str, Any]:
        """Return a symbol dictionary. Helper method to keep the code clean.

        The nodes in a GP tree consists of different symbols. The symbols
        are either functions (internal nodes with arity > 0) or terminals
        (leaf nodes with arity = 0) The symbols are represented as a
        dictionary with the keys:
          - *arities* -- A dictionary where a key
            is a symbol and the value is the arity
          - *terminals* -- A list of
            strings(symbols) with arity 0
          - *functions* -- A list of
            strings(symbols) with arity > 0

        """
        assert len(arities) > 0

        # List of terminal symbols
        terminals = []
        # List of functions
        functions = []
        # Append symbols to terminals or functions by looping over the arities items
        for key, value in arities.items():
            # A symbol with arity 0 is a terminal
            if value == 0:
                # Append the symbols to the terminals list
                terminals.append(key)
            else:
                # Append the symbols to the functions list
                functions.append(key)

        assert len(terminals) > 0

        return {"arities": arities, "terminals": terminals, "functions": functions}

    def parse_gp_configs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Parse configuration file."""
        # Parse function(non-terminal) symbols and arity
        for k, v in args["arities"].items():
            args["arities"][k] = int(v)
        # Parse constants
        args["constants"] = [
            float(_.strip()) for _ in args["constants"]["values"].split(",")
        ]
        return args


class SymbolicRegressionGPPlusSomeLLMConstrainedFewShot:
    def __init__(self, param: Dict[str, Any]):
        self.fitness_cases_file = param["fitness_cases_file"]
        self.test_train_split = param["test_train_split"]
        self.execution_environment = SymbolicRegressionGPInterpreter(param)
        (
            self.test,
            self.train,
            self.hold_out,
        ) = self.execution_environment.get_test_and_train_data(
            self.fitness_cases_file, self.test_train_split
        )
        self.n_exemplars = param["n_exemplars"]
        self.exemplar_idx = random.sample(
            range(len(self.train["fitness_cases"])), self.n_exemplars
        )
        self.fitness_cases = []
        self.targets = []
        for i in self.exemplar_idx:
            self.fitness_cases.append(self.train["fitness_cases"][i])
            self.targets.append(self.train["targets"][i])

        self.DEFAULT_PHENOTYPE = "0"

        self.arities = self.execution_environment.get_arities(param)
        self.symbols = self.execution_environment.get_symbols(self.arities)
        self.constraints = list(self.symbols["arities"].keys())
        self.few_shot_cache = collections.defaultdict(list)
        self.n_shots = param["n_shots"]

        self.variables = [_ for _ in self.symbols["terminals"] if _.startswith("x")]
        self.replacements = []
        # TODO how many fitness cases to use
        for exemplar in self.fitness_cases:
            replacement = [
                (variable_name, value)
                for variable_name, value in zip(self.variables, exemplar)
            ]
            self.replacements.append(replacement)

        self.INDIVIDUAL_GENERATION_PROMPT = """
Generate a mathematical expression. Use the listed symbols {constraints}.

Provide no additional text in response. Format output in JSON as {{"expression": "<expression>"}}
""".format(
            constraints=self.constraints
        )
        self.REPHRASE_MUTATION_PROMPT = """
{n_samples} examples of mathematical expressions are:
{samples}

Rephrase the mathematical expression {expression} into a new mathematical expression. Use the listed symbols {constraints}.

Provide no additional text in response. Format output in JSON as {{"new_expression": "<new expression>"}}
"""
        self.CROSSOVER_PROMPT = """
{n_samples} examples of mathematical expressions are:
{samples}

Recombine the mathematical expressions {expression} and create {n_children} new expressions from the terms. Use only the existing expressions when creating the new expressions.

Provide no additional text in response. Format output in JSON as {{"expressions": ["<expression>"]}}    
"""

    def get_context(
        self, samples: Optional[List[Any]] = None, n_shots: int = 0
    ) -> Tuple[str, int]:
        if samples is not None:
            n_samples = min(len(samples), n_shots)
            sample_input = random.sample(list(samples.keys()), n_samples)
            # TODO sort (well that is for a specific mutation ablation)
        else:
            sample_input = ""
            n_samples = 0

        return sample_input, n_samples

    def form_prompt_individual_generation(
        self, prompt_inputs: Optional[Dict[str, str]] = None
    ) -> str:
        prompt = self.INDIVIDUAL_GENERATION_PROMPT
        return prompt

    def check_response_individual_generation(self, response: str) -> str:
        try:
            phenotype = json.loads(response)["expression"]
        except TypeError as e:
            phenotype = self.DEFAULT_PHENOTYPE
            logging.error(
                f"{e} when formatting response for individual generation for {response}"
            )

        return phenotype

    
    @staticmethod
    def get_tokens(solution_str: str, operators: Set[str]) -> Set[str]:
        for operator in operators:
            solution_str = solution_str.replace(operator, f" {operator} ")

        tokens = set(map(lambda x: x.strip(), solution_str.split()))
        return tokens

    def check_response_crossover(self, response: str, parents: List[Individual]) -> str:
        try:
            phenotype = json.loads(response)["expressions"]
        except TypeError as e:
            logging.error(f"{e} when formatting response for crossover for {response}")
            phenotype = [_.phenotype for _ in parents]
        except json.decoder.JSONDecodeError as e:
            logging.error(f"{e} when formatting response for crossover for {response}")
            phenotype = [_.phenotype for _ in parents]
        except KeyError as e:
            logging.error(f"{e} when formatting response for crossover for {response}")
            phenotype = [_.phenotype for _ in parents]

        assert len(phenotype) == len(parents), f"{len(phenotype)} != {len(parents)}"
        tokens = set()
        child_tokens = set()
        for parent, child in zip(parents, phenotype):
            parent_str = parent.phenotype.replace("(", "").replace(")", "")
            parent_tokens = (
                SymbolicRegressionGPPlusSomeLLMConstrainedFewShot.get_tokens(
                    parent_str, self.symbols["functions"]
                )
            )

            tokens = tokens.union(parent_tokens)
            child_str = child.replace("(", "").replace(")", "")
            child_tokens = SymbolicRegressionGPPlusSomeLLMConstrainedFewShot.get_tokens(
                child_str, self.symbols["functions"]
            )

        diff = tokens.difference(child_tokens)

        assert (
            len(diff) == 0
        ), f"{len(diff)} tokens difference. {diff}. From {child_tokens} and {tokens}"
        return phenotype

    def check_response_rephrase_mutation(self, response: str, expression: str) -> str:
        try:
            phenotype = json.loads(response)["new_expression"]
        except json.decoder.JSONDecodeError as e:
            phenotype = expression
            logging.error(
                f"{e} when formatting response for rephrase mutation for {response}"
            )
        except KeyError as e:
            phenotype = expression
            logging.error(
                f"{e} when formatting response for rephrase mutation for {response}"
            )
        except TypeError as e:
            phenotype = expression
            logging.error(
                f"{e} when formatting response for rephrase mutation for {response}"
            )

        return phenotype

    def form_prompt_rephrase_mutation(
        self, expression: str, samples: Optional[List[Any]] = None
    ) -> str:
        context, n_samples = self.get_context(samples, self.n_shots)
        prompt = self.REPHRASE_MUTATION_PROMPT.format(
            expression=expression,
            constraints=self.constraints,
            samples=context,
            n_samples=n_samples,
        )
        return prompt

    def form_prompt_crossover(
        self, expressions: str, samples: Optional[List[Any]] = None
    ) -> str:
        context, n_samples = self.get_context(samples, self.n_shots)
        expression = " and ".join(expressions)
        prompt = self.CROSSOVER_PROMPT.format(
            expression=expression,
            constraints=self.constraints,
            samples=context,
            n_samples=n_samples,
            n_children=len(expressions),
        )
        return prompt

    def __call__(self, solution: str, cache: Dict[str, float]) -> float:
        """Create a list of nodes to evaluate.

        Uses sympy for evaluation
        """
        key = str(solution)
        if key in cache.keys():
            fitness = cache[key]
        else:
            # TODO use lambdify for sympy and numpy...
            try:
                solution = sympy.sympify(solution, evaluate=True)
            except sympy.core.SympifyError as e:
                logging.error(f"{e} for sympifying {solution}")
                solution = sympy.sympify(self.DEFAULT_PHENOTYPE, evaluate=True)
            except TypeError as e:
                logging.error(f"{e} for sympifying {solution}")
                solution = sympy.sympify(self.DEFAULT_PHENOTYPE, evaluate=True)

            outputs = []
            errors = []

            for replacement, target in zip(self.replacements, self.targets):
                try:
                    output = solution.subs(replacement)
                except AttributeError as e:
                    logging.error(
                        f"{e} for error calculation of {solution} for {replacement}"
                    )
                    output = ""
                outputs.append(output)
                try:
                    error = (output - target) ** 2
                    errors.append(error)
                except TypeError as e:
                    logging.error(
                        f"{e} for error calculation of for {solution} for {output} {replacement}"
                    )
                    fitness = EvolutionaryAlgorithm.DEFAULT_FITNESS
                    cache[key] = fitness
                    return fitness

            try:
                fitness = float(-np.mean(errors))
            except TypeError as e:
                logging.error(
                    f"{e} for fitness calculation of for {solution} for {errors}"
                )
                fitness = EvolutionaryAlgorithm.DEFAULT_FITNESS
            cache[key] = fitness

        return fitness
