# Akkadian to English

Skip to section you are looking for!

- [Setting up](#setting-up)
- [Overview](#overview)
- [Pipeline](#code-pipeline)
- [Preprocessing](#preprocessing:-normalization-of-transliteration)
- [Data Augmentation](#data-augmentation)
- Tokenization
- [Loss functions](#loss-functions)
- [Base Models](#base-models)
- [Finetuning](#finetuning)
- [Dataset](#dataset)

## Setting up

### Packages

We are using `uv` as a package manager, you may need to install it. To install/sync the projects dependencies you can do:

```sh
uv sync
```

To add a library/package you can do it with

```sh
uv add PACKAGE-NAME
```

And you are done

## Overview

Decision graph for what approach to take base on [this paper](https://dl.acm.org/doi/10.1145/3567592)

![image](images/survey-2023-low-resource-languages-ml.png)

### Code Pipeline

```%%{init: { 'theme': 'default', 'sequence': {'messageMargin': 50, 'mirrorActors': false}}%%
sequenceDiagram

    participant U as Kaggle Notebook
    participant P as src/parser
    participant T as src/training
    participant M as src/models
    participant D as src/data_processing

    Note over U: Start model training

    activate T
    U->>T: User inputs

    activate P
    T-->>P: User inputs
    P-->>T: Config Dictionary
    deactivate P

    activate M
    T-->>M: Create model<br/>w/ mBART(config)
    T->>M: Initialize Model
    M-->>T: Return instance
    deactivate M

    Note over T: Apply data augmentation<br/>and preprocessing

    activate D
    T->>D: Create datasets<br/>via configs
    D-->>T: Return train.csv
    deactivate D

    Note over T: Start Training Loop
    T->>T: Iterate Epochs &<br/>Log to WandB
    T->> U: Saved model
    deactivate T
```

See training/mbart.py for an example

## Loss functions

Kaggle will score the inference's results base on `BLEU` and `chrF++`

## Preprocessing: Normalization of transliteration

This includes things to consider to normalize transliterations (e.g. artifacts useful to specialists but not for translation)

- **Hyphenated syllables** (a-na e-lá-ma → ana eláma): should we convert to non hyphernated or is this of any use?
  We need to make sure these get processed as a single token

- **Cases** How to handle them:
  - **All caps**: means logograms, similar to 漢 (e.g. the word kind is written in one character and transliterated to LUGAL) → we should **not** lowercase everything or we loose the distinction of logograms.

  - **Capitalized words:** These are proper nouns (this is said in the description in Koggle, but I have not found examples of this in the data).

- **Colons in words** (e.g. GÍN.TA): Indicates connection between compound logograms (similar to 漢字 ). For example, GIN is shekels, TA is per → Shekels per.
  We should treat these as single tokens (so maybe strip the dot).

- **Flags:** Full list below.

  > Shall we strip them or treat as special tokens? However, in koggle info ! means certainty, and in guide ! means remarkable, so which is it? I also can't find any of these in the dataset?? I decided to strip them, come to me if you wanna fight this decition (Vero).
  1.  Exclamation marks when a scholar is certain about a difficult reading of a sign !, or for remarkable(eg unusual) in the guide

  2.  Question mark (?) when a scholar is uncertain about a difficult reading of a sign

  3.  Forward slash (/) for when the signs belonging to a line are found below the line

  4.  Colon for the Old Assyrian word divider sign :

- **Breaks, errors and damage:**
  - `\[something\]` means the word was restored, I think we can just strip them if we trust the translators.

  - `\[x\]` means there is a gap of a single character→ replace with special token _\<gap\>_. x without brackets means one sign was unreadable, so I think we can treat x also as _\<gap\>_.

  - `…` or `\[... ...\]` means big gap → replace with special token `_\<big_gap\>_`.

  - Other things from the dataset are not clear: e.g. `\[... ... ...-Aš\]` is used to indicate this is a name but it is incomplete (in the transliteration there was an unreadable character at the beginning defined as x)

  - `\<something\>` means scribal omitted a word and it's being inserted. I think we can just strip them if we trust the translators.

  - `\<\<something\>\>` means the scribal wrote an erroneous signs and it's being corrected. I think we can just strip them if we trust the translators.

    > For all of these I do not understand why there are discrepancies between transliteration and translation: how can there be `\[...\]` in the translation but not in the transliteration?
    > We need to make sure the tokenizer does not split \< \> and the words inside.

- **Superscripts, subscripts:** These indicate special characters (e.g. different accents). We should see what is best for our model (e.g. convert to ascii (u₂ to u2) or something else).

- **Determinatives** give context to words (full list below). I think we should **not** remove because they help with context, but we must be careful not to treat these as words, maybe as special characters.

  > **NOTE:** I do not find any curly brackets in the dataset, but I do find some (d), (ki), info on koggle could be wrong? Koggle says round brackets are for comments for breaks and erasures.
  1.  {d} = dingir ‘god, deity’ — d preceding non-human divine actors

  2.  {mul} = ‘stars’ — MUL preceding astronomical bodies and constellations

  3.  {ki} = ‘earth’ — KI following a geographical place name or location

  4.  {lu₂} = LÚ preceding people and professions

  5.  {e₂} = {É} preceding buildings and institutions, such as temples and palaces

  6.  {uru} = (URU) preceding names of settlements, such as villages, towns and cities

  7.  {kur} = (KUR) preceding lands and territories as well as mountains

  8.  {mi} = munus (f) preceding feminine personal names

  9.  {m} = (1 or m) preceding masculine personal names

  10. {geš} / {ĝeš) = (GIŠ) preceding trees and things made of wood

  11. {tug₂} = (TÚG) preceding textiles and other woven objects

  12. {dub} = (DUB) preceding clay tablets, and by extension, documents and legal records

  13. {id₂} = (ÍD) (a ligature of A and ENGUR, transliterated: A.ENGUR) preceding names of canals or rivers or when written on its own referring to the divine river
  14. {mušen} = (MUŠEN) preceding birds

  15. {na₄} = (na4) preceding stone

  16. {kuš} = (kuš) preceding (animal) skin, fleece, hides

  17. {u₂} = (Ú) preceding plants

- **Coherence of characters** decided as majority counts seen below (from train set):
  Written numbers (one-nine): 420
  _Digit numbers (1-9): 4461_
  _Decimal fractions (.5, .25, etc.): 1740_
  Symbol fractions (½, ¼, etc.): 82
  _Triple dots (...): 2996_
  Ellipsis character (…): 0

## Data Augmentation

1- Pass sentence and sentence+noise (token or gaps), minimize error -> denoising
2- Replace names with others

## Base Models

A simple model ByT5 training/finetuning notebook can be found at https://www.kaggle.com/code/takamichitoda/dpc-starter-train
An even simpler inferring notebook (low score, ~25) can be found at https://www.kaggle.com/code/takamichitoda/dpc-starter-infer
Much to improve one these:

- Input data barely preprocessed/aligned
- Not many hyperparameters adapted
- Etc.
  Just useful to get an idea of how training a byT5 model works.

## Models Ensembling

![alt text](images/ensembling.png)
We can use an ensemble of T5 (e.g. https://huggingface.co/Thalesian/akk-111m uses google-t5-small, or byT5 https://huggingface.co/notninja/byt5-base-akkadian) as it is widely tested and shows good performance, and mBART, which leverages semi supervised learning (useful for resource constrained context).
We could have learnable weights to get a final answer form the different predictions.

According to Goodfellow, Benjo and Courville (and some comments in lecture), competitions are usualy won by ensembles (of over 12, but also 5-10 shows significant improvement). This requires extensive memory and compute, but we can use Dropout training to have effective ensembles without actually training all mdoels individually.

Train them separately, might ensemble same models with different preprocessing.

## Named Entity Recognition

# Maybe using this pass to make sure proper names match. Idea: replace all names in transliteration to English names (e.g. Jhon, Mark), check match in English, substitute back with name in original sentence.

## Finetuning

LoRA HuggingFace recomendations for finetuning in language tasks. [Link](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)

> - trainable tokens, train only the specified tokens, optionally store only the updated values
> - training an adapter on the embedding matrix, optionally store only the updated values
> - full-finetuning of the embedding layer

### Parameters

- 32 Precision is said to be way better than 16.[here](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/664981)

## Kaggle

### Running notebooks

In kaggle you can run notebooks in two ways.

- Interactive mode
- Submission mode

**Interactive mode** It is similar to the way we use Jupyter notebooks typically. Interactive mode is useful for when you are developing your notebook. These sessions usually timeout and outputs are not persistent.

**Submission mode**: Once you have a nice notebook that you would like to run for a long time, or permantly save its outputs. You can use submission mode, you do this by click _Save and Submit_. Then Kaggle will execute all the cells in the notebook and save the outputs that have been written in the output directory. This notebook can run for more than 10 hours and you dod not need to have your browser open or anything like that.

### Define Secrets

This is important for defining the Wandb API key and the GitHub personal access token.

Do the following steps

1.  Open your Kaggle Notebook.
2.  In the top menu bar, click on Add-ons -> Secrets.
3.  A sidebar will appear. Click "Add a new secret".
4.  For W&B:
    - Label: Type wandb
    - Value: Paste your W&B API Key.
5.  For GitHub:
    - Label: Type github
    - Value: Paste your GitHub Personal Access Token.

GitHub Personal Access Token steps:

1. Log in to GitHub.
2. Click your profile picture in the top-right corner and select Settings.
3. On the left sidebar, scroll all the way to the bottom and click Developer settings.
4. Click Personal access tokens.
5. Choose Your Token Type - classic is fine for simple Kaggle setups
6. Click Generate new token -> Generate new token (classic). Note: Give it a name like "Kaggle Competition."
7. Scopes (Crucial): Check the box for repo. This is all you need to clone, pull, and push code.
8. Scroll to the bottom and click Generate token

### Weight and Biases

We use weight and biases for the logging. Below you can find a script to use it in Kaggle

```
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("wandb")

import os

os.environ["WANDB_API_KEY"] = api_key
!wandb login
```

### Uploading code

The following script allows you to clone the github repo in kaggle and to make imports from it to your notebook.

```

from kaggle_secrets import UserSecretsClient
import os
user_secrets = UserSecretsClient()
GITHUB = user_secrets.get_secret("github")

os.system(f"git clone https://{GITHUB}@github.com/riberoc/MLiP---Translate-Akkadian-to-English---Kaggle-.git /kaggle/working/akkadian-to-english")

import sys
from pathlib import Path

# Path to your repo
repo_path = Path("/kaggle/working/akkadian-to-english")
src_path = repo_path / "src"

# Add src to sys.path if not already there
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

```

### Inference

You cannot clone in kaggle due to the lack of internet.

`(akkadian-to-english) > $ pyCombiner  src/evaluation/kaggle.py -p .`

## Dataset

- We use the dataset provided by the kaggle team
- Akkademia dataset (test,train,valid) under `dataset/external`

## DIscussions

- Gemma with bad performance
- 32 precision is better than 16
- [leaders have overfitted](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/668619)
- [LLM for postprocessing](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/664079)
