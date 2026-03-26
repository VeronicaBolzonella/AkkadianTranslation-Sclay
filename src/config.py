from enum import Enum
from pathlib import Path

# Get the project root dynamically
# This finds the directory where config.py lives, then goes up one level
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Types of training modes
class TrainingMode(str, Enum):
    SUPERVISED = "supervised"
    SELF_SUPERVISED = "self_supervised"


class DatasetType(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


# Names for dataframe columns
class Columns(str, Enum):
    NOISY_GAPS = "noisy_gaps_transliteration"
    NOISY_TOKENS = "noisy_tokens_transliteration"
    TRANSLITERATION = "transliteration"
    TRANSLATION = "translation"
    ID_TRAINING = "oare_id"
    ID_TEST = "id"


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset"

# These are the datasets that will be used as input
DATASET_INPUTS = DATASET_PATH / "training_input/"

EXTERNAL_DATASET_INPUTS = DATASET_INPUTS / "external/"
INTERNAL_DATASET_INPUTS = DATASET_INPUTS / "internal/"

## Alignment guide path
ALIGNMENT_PATH = DATASET_PATH / "alignment" / "Sentences_Oare_FirstWord_LinNum.csv"

## All the paths to extra dictionaries used for data augmentation or pretraining
DICTIONARY_DATASET_INPUTS = DATASET_PATH / "dictionaries/"

MORPHEMES_PATH = DICTIONARY_DATASET_INPUTS / "akkadian_morphemes.csv"
LEXICON_PATH = DICTIONARY_DATASET_INPUTS / "OA_Lexicon_eBL.csv"
DICTIONARY_PATH = DICTIONARY_DATASET_INPUTS / "eBL_Dictionary.csv"
ONOMASTICON_PATH = DICTIONARY_DATASET_INPUTS / "onomasticon-cleaned.csv"
