import collections
import importlib
import json
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class FitnessFunction:
    """
    Fitness function abstract class
    """

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        raise NotImplementedError("Define in subclass")


def import_function(fitness_function_str: str) -> str:
    module, method = fitness_function_str.rsplit(".", 1)
    fitness_function = importlib.import_module(module)
    method = getattr(fitness_function, method)
    return method


def get_fitness_function(param: Dict[str, str]) -> FitnessFunction:
    """Returns fitness function object.

    Used to construct fitness functions from the configuration parameters

    :param param: Fitness function parameters
    :type param: dict
    :return: Fitness function
    :rtype: Object
    """

    name = param["name"]
    fitness_function_method = import_function(name)
    fitness_function = fitness_function_method(param)

    return fitness_function


def get_ave_and_std(values: Sequence[float]) -> Tuple[float, float]:
    """
    Return average and standard deviation.
    """
    _ave: float = float(np.mean(values))
    _std: float = float(np.std(values))
    return _ave, _std


def get_out_file_name(out_file_name: str, param: Dict[str, Any]) -> str:
    if "output_dir" in param:
        output_dir = param["output_dir"]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        out_file_name = os.path.join(output_dir, out_file_name)
    return out_file_name


def print_cache_stats(generation: int, param: Dict[str, Any]) -> None:
    _hist: Dict[str, int] = collections.defaultdict(int)
    for v in param["cache"].values():
        _hist[str(v)] += 1

    info = f"Cache entries:{len(param['cache'])} Total Fitness Evaluations:{generation * param['population_size'] ** 2} Fitness Values:{len(_hist)}"
    logging.info(info)
    out_file_name = f"{get_out_file_name('alfa_ec_llm', param)}_cache.jsonl"
    with open(out_file_name, "w") as fd:
        for key, value in param["cache"].items():
            json.dump({key: value}, fd)
            fd.write("\n")


def write_run_output(
    generation: int, stats: Dict[str, List[Any]], param: Dict[str, Any]
) -> None:
    """Write run stats to files."""
    assert len(param["cache"]) > 0
    print_cache_stats(generation, param)
    out_file_name = get_out_file_name("alfa_ec_llm", param)
    _out_file_name = f"{out_file_name}_settings.json"
    with open(_out_file_name, "w", encoding="utf-8") as out_file:
        _settings: Dict[str, Any] = {}
        for k, v in param.items():
            if k not in ("cache", "openai_interface", "llm_interface"):
                _settings[k] = v

        try:
            json.dump(_settings, out_file, indent=1)
        except TypeError as e:
            logging.error(f"{e} when dumping {_settings} to {out_file}")

    for k, v in stats.items():
        _out_file_name = f"{out_file_name}_{k}.json"
        with open(_out_file_name, "w", encoding="utf-8") as out_file:
            json.dump({k: v}, out_file, indent=1)


def write_run_output_gp_plus_llm(
    generation: int,
    stats: Dict[str, List[Any]],
    param: Dict[str, Any],
    generation_history: List[Tuple[str, str]],
) -> None:
    write_run_output(generation, stats, param)
    out_file_name = get_out_file_name("alfa_ec_llm", param)
    _out_file_name = f"{out_file_name}_llm_generation_history.jsonl"
    with open(_out_file_name, "w", encoding="utf-8") as fd:
        for line in generation_history:
            try:
                fd.write(f"{json.dumps(line)}\n")
            except TypeError as e:
                logging.error(f"{e} for {line} in {_out_file_name}")
