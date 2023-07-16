datasets:
	echo "Download datasets from https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy"
	unzip all_six_datasets.zip -d data
	poetry run python data/all_dataset.py

format:
	poetry run isort .
	poetry run black .

install:
	poetry install