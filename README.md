# Tutorial-LLM_GP

Simple implementation of Evolutionary Computation with LLMs. It is an extension of [https://github.com/flexgp/pony_gp](https://github.com/flexgp/pony_gp) with addition of LLM based operators. Uses `python3`. 

## Install

Install requirements
```
# Create virtual environment. Here named `venv_tutorial_llm_gp`
python3 -m venv ~/.venvs/venv_tutorial_llm_gp
# Activate virtual environment. Here `venv_tutorial_llm_gp`
source ~/.venvs/venv_tutorial_llm_gp/bin/activate
# Install requirements
pip install -r ./requirements.txt
```

Install as a package e.g. `pip install -e .`

## OpenAI API

Set the environment variable for `OPENAI_API_KEY` to your API key. E.g.
```
export OPENAI_API_KEY = "Some long and safe hash"
```
## Run

Paths are relative the repository root.

### Symbolic Regression

You can run [tutorial.ipynb](tutorial.ipynb)

#### Run GP

 ```
 python main.py -f tests/configurations/tutorial_gp_sr.yml -o results_gp --algorithm=tutorial_gp
 ```

#### Run LLM-GP

 ```
 python main.py -f tests/configurations/tutorial_llm_gp_sr.yml -o results_llm_gp --algorithm=tutorial_llm_gp_mu_xo
 ```

### `tutorial_llm_gp` output

`tutorial_llm_gp` prints some information to `stdout` regarding `settings` and
search progress for each iteration, see `tutorial_llm_gp.py:print_stats`. 

The output files have each generation as a list element, and each individual separated by a `,`. They are written to:
```
tutorial_llm_gp_*_fitness_values.json
tutorial_llm_gp_*_length_values.json
tutorial_llm_gp_*_size_values.json
tutorial_llm_gp_*_solution_values.json
```

### Usage

```
usage: main.py [-h] -f CONFIGURATION_FILE [-o OUTPUT_DIR] [--algorithm {tutorial_gp,tutorial_llm_gp_mu_xo}]

Run Tutorial_LLM-GP

options:
  -h, --help            show this help message and exit
  -f CONFIGURATION_FILE, --configuration_file CONFIGURATION_FILE
                        YAML configuration file. E.g. configurations/tutorial_llm_gp_sr.yml
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to directory for output files. E.g. tutorial_llm_gp_output
  --algorithm {tutorial_gp,tutorial_llm_gp_mu_xo}
                        Algorithms
```

### Settings

Configurations are in `.yml` format, see examples in folder [configurations](tests/configurations).

## Test

Tests are in `tests` folder. E.g run with
```
python -m unittest tests.test_main
```