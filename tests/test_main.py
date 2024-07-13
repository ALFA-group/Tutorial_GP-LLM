import logging
import os
import unittest
from typing import List

import main

log_file = os.path.basename(__file__).replace(".py", ".log")
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(funcName)s: %(module)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

ARGS: List[str] = ["-o", "tmp"]
os.makedirs(ARGS[1], exist_ok=True)


class TestMain(unittest.TestCase):
    def test_one_sr_gp(self) -> None:
        _args = ARGS + [
            "-f",
            "tests/configurations/tutorial_gp_sr.yml",
            "--algorithm=tutorial_gp",
        ]
        main.main(_args)

    def test_one_sr_tutorial_llm_gp_mu_xo(self) -> None:
        _args = ARGS + [
            "-f",
            "tests/configurations/tutorial_llm_gp_sr.yml",
            "--algorithm=tutorial_llm_gp_mu_xo",
        ]
        main.main(_args)


if __name__ == "__main__":
    unittest.main()
