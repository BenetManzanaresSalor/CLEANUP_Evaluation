<h1 align="center">Text Anonymization Evaluator (TAE)</h1>
<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License on Windows 11"/>
  <img src="https://img.shields.io/badge/Windows%2011-Working-ok" alt="Working on Windows 11"/>
  <img src="https://img.shields.io/badge/Linux_based-Compatible_but_not_tested-lightgrey" alt="Compatible but not tested on Linux-based systems"/>
</p>

This repository contains the code and experimental data for the **Text Anonymization Evaluator** (TAE), an evaluation tool for text anonymization including multiple state-of-the-art utility and privacy metrics.

Experimental data was extracted from the [text-anonymization-benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) repository, corresponding to the publication [Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., & Batet, M., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/). The exact data files utilized are located in the [data](data) folder.




## Table of Contents
* [Project structure](#project-structure)
* [Install](#install)
  * [From source](#from-source)
  * [From PyPi](#from-pypi)
* [Usage](#usage)
  * [From CLI](#from-cli)
  * [From code](#from-code)
* [Configuration](#configuration)
* [Examples](#examples)




# Project structure
```
Text Anonymization Evaluator (TAE)
│   README.md                               # This README
│   pyproject.toml                          # Package project defintion file
│   environment.yml                         # Dependencies file for Conda
│   requirements.txt                        # Dependencies file for Pip
│   LICENSE.txt                             # License file
│   config.json                             # Example configuration file
└───taeval                                  # Package source code folder
│   |   __init__.py                         # Script for package initialization
│   |   __main__.py                         # Script to be executed as CLI
│   │   tae.py                              # Script including the TAE class, containing the main code of the package
│   |   tri.py                              # Script including the TRI class for re-identification risk assessment
│   |   utils.py                            # Script including the general common-usage classes
└───data                                    # Folder for data files
    └───tab                                 # Folder for TAB dataset
        └───corpora                         # Folder for dataset's corpus files
        |   |...
        └───anonymizations                  # Folder for anonymizations to evaluate
        |   |...
        └───bks                             # Folder for background knowledges for re-identification risk assessment
            |...
```




# Install
Our implementation uses [Python 3.9.19](https://www.python.org/downloads/release/python-3919/) as programming language. For dependencies management, we employed [Conda](https://docs.conda.io/en/latest/) 24.1.2, with all used packages and resources listed in the [environment.yml]([environment.yml) file. However, we also considered **Pip**, including an equivalent [pyproject.toml](pyproject.toml) file and uploading the package to [PyPi](https://pypi.org/) under the name `taeval`. Below we detail how to install the package [from source](#from-source) and [from PyPi](#from-pypi).

## From source
If you want to use TAE from CLI (see [Usage section](#usage) for details), we recommend to install it from source following the next steps:
1. Download or clone this repository:
    ```console
    git clone https://github.com/NorskRegnesentral/text-anonymization-evaluator
    cd text-anonymization-evaluator
    ```
2. Install dependencies:
    * Option A: Using Conda 
        * Install [Conda](https://docs.conda.io/en/latest/) if you haven't already.
        * Create a new Conda environment using the [environment.yml](environment.yml) file (channels included for ensuring that specific versions can be installed):
            ```console
            conda create --name ENVIRONMENT_NAME --file environment.yml -c conda-forge -c spacy -c pytorch -c nvidia -c huggingface -c numpy -c pandas
            ```
        * Activate the environment:
            ```console
            conda activate ENVIRONMENT_NAME
            ```
    * Option B: Using Pip
        ```console
        pip install -e . # This uses the pyproject.toml file
        ```

## From PyPi
If you want to use TAE from code (see [Usage section](#usage) for details), we recommend installing it from PyPi via Pip with the following command:
```console
pip install taeval
```




# Usage examples
TAE was designed to be run [from CLI](#from-cli), but it can also be executed [from code](#from-code). In the following, we instruct into how to execute it using both approaches, assuming that the steps from the [Install section](#install) have been already completed.

## From CLI
Running from CLI requires to pass as argument the path to a JSON configuration file. This file contains a dictionary detailing the corpus, anonymization output file path and metrics to use (check the [Configuration section](#configuration) for details).

For instance, for using the [config.json](config.json) example configuration file, run the following command:
```console
python -m tae config.json
```
This assumes that the current working directory contains the [tae](tae) package folder. Finding this containing folder can be non-trivial if you installed the package [from PyPi](#from-pypi). That is why we recommend to install it [from source](#from-source) for CLI usage.


## From code
Running from code requires to create an instance of the TAE class (defined in the [tae.py](tae/tae.py) script) passing the desired configuration as arguments. This configuration is constituted by the corpus, anonymization output file path and metrics to use (check the [Configuration section](#configuration) for details). That is exactly the same as for the JSON configuration file mentioned in the [from CLI section](#from-cli), but defined by code.

The following script exemplifies 
```python
from tae import TAE

# Load anonymizations
anonymizations = {} #TODO
anonymizations_config = config[ANONYMIZATIONS_CONFIG_KEY]
for anon_name, anon_file_path in anonymizations_config.items():
    masked_docs = MaskedCorpus(anon_file_path)
    anonymizations[anon_name] = masked_docs

# Define metrics dictionary
metrics = {
    "Precision":{}, # Uses default configuration
    "TPI": {}, #TODO: Add default value    
}

# Define file path for the results CSV file (containing directory will be created automatically)
results_file_path = "outputs/results.csv"

# Create TAE instance from the corpus file
tae = TAE.from_file_path(corpus_file_path) #TODO

# Run evaluation
results = tae.evaluate(anonymizations, metrics, results_file_path)

# NOTE: The TAE instance can be reused for evaluating the corpus using other anonymizations, metrics and results file path
```
This assumes that you have TAE ready to import. That is trivial if you have install it [from PyPi](#from-pypi), but requires you to have the [tae](tae) package folder within your project workspace if you have install it [from source](#from-source). That is why we recommend to install it [from PyPi](#from-pypi) for usage from code.


# Configuration
TODO




# Examples
TODO (already done in Usage examples?)