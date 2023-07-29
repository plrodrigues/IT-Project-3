# IT-Project-3

This particular project has been implemented as part of the Information Theory module at [FEUP (Faculdade de Engenharia da Universidade do Porto)](https://sigarra.up.pt/feup/en/web_page.inicial). 

Its primary focus involves exploring fundamental methodologies that facilitate access to causal relationships within data.
The [Weather dataset](https://www.kaggle.com/datasets/swatikhedekar/python-project-on-weather-dataset), which is readily available on [Kaggle](https://www.kaggle.com/), is utilized for the purpose of testing the implementation of these methods.
The resulting insights derived from data analysis are subject for an internal report.

## Code for the internal report

In order to compile an internal report, a variety of notebooks were built, which can be accessed at the following directory: [/notebooks/](/notebooks/).

## Python environment

How to replicate the experiments:

Create environment inside project root directory:

```sh
python -m venv .venv_it_proj3
```

Activate it, to have the poetry environment created inside it, to avoid being created in the `.cache`:

```sh

source .venv_it_proj3/bin/activate
```

Install all poetry dependencies defined for this project:

```sh
poetry install
```

Activate them in the `.venv_it_proj3` environment:

```sh
poetry shell
```

## Run tests and checks

All the tests are inplemented under [/tests/](/tests/).

To run them, you can simply run the following command:

```sh
source .bashrc
```

The command above will:

- format the Python code with Python Black
- sort the imports using isort
- run the implemented tests


