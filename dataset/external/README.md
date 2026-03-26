## Explanation

A folder for each external dataset

### Akkademia

All the files ending in `.en` or `.tr` belong to the Akkademia Github Repository.

The corresponding `.csv` files are in the format used by the Kaggle challenge. `id`,`transliteration` and `translation`

The `all.csv` contains all the train,valid and test csv stacked together.

> [!Note] The code for this lives at `data_processing/external_prep.py`

### Scraped

Datasets obtained from scraping the website provided in the Kaggle challenge
