data-electricity:
	mkdir -p data
	cd data && wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
	poetry run python data/electricity_dataset.py

data-pattern:
	mkdir -p data
	poetry run python data/patterns_dataset.py

datasets: data-electricity data-pattern

format:
	poetry run isort .
	poetry run black .

install:
	poetry install