from torch.utils.data import Dataset
import torch
import pandas as pd
from src.config import Columns, DatasetType, TrainingMode

"""
Usage: 
Call TextProcessor() before calling the dataset.
"""


class AkkadianEnglishDataset(Dataset):
    """
    This is the dataset that allows for a tokenizer
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_length=512,
        weight=None,
        training_mode: TrainingMode = TrainingMode.SUPERVISED,
        is_inference: bool = False,
        dataset_type: DatasetType = DatasetType.INTERNAL,
        task_prefix: str = "",  # NEW: pass "translate Akkadian to English: " for T5
        eng_tokenizer=None,
    ):
        """Constructs the dataset according to the `training_mode` and if the model is inference mode

        Based on the training_mode it decides whether to create a dataset for self-supervised training
        or supervised training.

        Args:
            df (pandas.dataframe): Is the dataframe. If self-supervised learning is choosen
                It is expected to have the columns for data augmentation .e.g NOISY_GAPS  and NOISY_TOKENS
            tokenizer (): The tokenizer that will turn the strings into tokens. This is based on
                the model that you are using
            max_length (int) : I am not even sure myself (mARCELO)
            training_mode: The dataset will look different depending on the training mode
            is_inference (bool): Whether you are running for inference or not
            dataest_type (str): This specifies whether the dataset instance contains the `internal` dataset (the one provided by Kaggle)
                or the external one (e.g., Akkademia dataset)

        Raises:
            ValueError: When given a training_mode that there is no constructor for it.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training_mode = training_mode
        self.is_inference = is_inference
        self.dataset_type = dataset_type
        self.task_prefix = task_prefix
        self.eng_tokenizer = eng_tokenizer
        self.weight = weight

        # Store the id_key so it's accessible in helper methods
        self.id_key = Columns.ID_TEST

        # Route the dataset creation based on mode
        if self.training_mode == TrainingMode.SELF_SUPERVISED:
            self._create_self_supervised_dataset(df)
        elif self.training_mode == TrainingMode.SUPERVISED:
            self._create_supervised_dataset(df)
        else:
            raise ValueError(
                f"No constructor for the training_mode: {self.training_mode} "
            )

    def _create_supervised_dataset(self, df):
        self.sample_ids = df[self.id_key].astype(str).tolist()
        self.input_texts = df[Columns.TRANSLITERATION].tolist()

        if not self.is_inference:
            self.target_texts = df[Columns.TRANSLATION].tolist()
        else:
            self.target_texts = []

    def _create_self_supervised_dataset(self, df):
        # Removed tgt_lang validation — not valid when mixing modes
        gaps_input = df[Columns.NOISY_GAPS]
        tokens_input = df[Columns.NOISY_TOKENS]
        self.input_texts = pd.concat(
            [gaps_input, tokens_input], ignore_index=True
        ).tolist()

        combined_targets = pd.concat(
            [df[Columns.TRANSLITERATION], df[Columns.TRANSLITERATION]],
            ignore_index=True,
        )
        self.target_texts = combined_targets.tolist()

        ids_gap = df[self.id_key].astype(str) + "_gap"
        ids_token = df[self.id_key].astype(str) + "_token"
        self.sample_ids = pd.concat([ids_gap, ids_token], ignore_index=True).tolist()

    def get_weights(self, internal_weight=0.2, external_weight=0.8):
        """Generate per-sample weights for this dataset based on its dataset_type.

        These weights are used by a PyTorch WeightedRandomSampler to control how
        frequently samples from different datasets are selected during training.

        If the dataset source is "internal", all samples receive `internal_weight`.
        Otherwise, all samples receive `external_weight`.

        This allows biasing training toward one dataset (e.g. the evaluation dataset)
        by making its samples more likely to be drawn.

        the values of `internal_weight` and `external_weight` are probabilities to be sampled

        Args:
            main_weight (float): Weight assigned to each sample if the dataset
                source is "internal". Higher values increase sampling frequency.
            external_weight (float): Weight assigned to each sample if the dataset
        """

        weight = (
            internal_weight
            if self.dataset_type == DatasetType.INTERNAL
            else external_weight
        )

        # If self.weight has been set. Use that instead
        if self.weight:
            weight = self.weight
        return [weight] * len(self)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        input_text = self.task_prefix + self.input_texts[index]

        model_input = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )

        if not self.is_inference:
            if self.eng_tokenizer:
                tokenizer = self.eng_tokenizer
            else:
                tokenizer = self.tokenizer
            labels = tokenizer(
                text_target=self.target_texts[index],
                max_length=self.max_length,
                padding=False,
                truncation=True,
            )["input_ids"]

            labels = [la if la != tokenizer.pad_token_id else -100 for la in labels]
            model_input["labels"] = labels

        # Add training mode so the model knows how to handle this batch
        model_input["training_mode"] = self.training_mode.value

        return model_input


class AkkadianTranslationDatasetT5(Dataset):
    """
    This class creates a dataset depending on the mode (train vs inference) and
    the task (translation vs reconstruction).

    Input requires already preprocessed data, meaning it should come after
    running TextProcessor() and the corresponding function on the text.
    Tokenizer needs to be initialized beforehand. It can be used like this:

    train_dataset = AkkadianTranslationDatasetT5(train_df,tokenizer)

    """

    def __init__(
        self, dataframe, tokenizer, max_length=512, mode="train", task="translation"
    ):
        """
        Intializes the dataset with a dataframe. Depending on the task, the
        input and target texts are created differently.

        Args:
            dataframe (pd.DataFrame): The dataframe with the train, validation
            or test data, already preprocessed. tokenizer (PreTrainedTokenize):
            The HuggingFace tokenizer from the pretrained model. max_length
            (int, optional): Sequence truncation length. Defaults to 512. mode
            (str, optional): "train" or "inference" Defaults to "train". task
            (str, optional): "translation" or "reconstruction". Defaults to
            "translation".
        """
        # self.sample_ids = dataframe["id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.task = task  # when denoising mode is true, the task is reconstruction rather than translation

        if task == "translation":
            # Input: transliteration with translation 
            self.input_texts = [
                "translate Akkadian to English: " + str(t)
                for t in dataframe["transliteration"]
            ]
            if self.mode == "train":
                # Target: English translation
                self.target_texts = [str(text) for text in dataframe["translation"]]
        else:
            # Input: Noisy transliteration with reconstruction prompt
            self.input_texts = [
                "reconstruct Akkadian from the noise: " + str(t)
                for t in dataframe["noisy_transliteration"]
            ]
            # Target: original transliteration
            self.target_texts = [str(text) for text in dataframe["transliteration"]]


        # Keep original order for bucket batching
        self.indices = list(range(len(self.input_texts)))

        # Precompute tokenized input lengths for bucket batching
        self.input_lengths = [
            min(
                len(
                    self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        add_special_tokens=True
                    )["input_ids"]
                ),
                self.max_length 
            )
            for text in self.input_texts
        ]

    
    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        """
        Retrieves and tokenizes a single data sample by its index. Processes the
        input text (transliteration) and target text (translation or original
        transliteration) into tensors. If in "train" mode, it includes labels.

        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - "input_ids": Tokenized input text as a tensor.
                - "attention_mask": Attention mask for the input text as a tensor.
                - "labels" (optional): Tokenized target text as a tensor, included only in "train" mode.
        """
        # Tokenize transliteration text (either normal or noisy)
        tokenized_inputs = self.tokenizer(
            self.input_texts[index],
            padding=False, # this happens in data collator
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Base model inputs
        item = {
            "input_ids": tokenized_inputs["input_ids"].squeeze(),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
            # Added for Bucket Batching
            "idx": self.indices[index], # for getting order
        }

        # Add target labels during training
        if self.mode == "train":
            tokenized_target = self.tokenizer(
                self.target_texts[index],
                padding=False, # this happens in data collator
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            labels = tokenized_target["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels

        return item

