all: black isort run_unit_tests

black:
	black src/
	black tests/

isort: 
	isort src/
	isort tests/

run_unit_tests:
	python tests/test_entropy.py
	python tests/test_mutual_information.py

.PHONY: black isort run_unit_tests