from data_processing.dataset_prep import DatasetPrep
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BatchEncoding,
)
from dataclasses import dataclass

import wandb
import evaluate
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from typing import cast, List, Dict
import math
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from data_processing.datasets import AkkadianTranslationDatasetT5
from data_processing.processing import TextProcessor
from data_processing.augmentation import DataAugmentation
from data_processing.alignment import Aligner


class ByT5Trainer:
    """
    This class serves as a pipeline for finetuning the ByT5 model with the
    HuggingFace Seq2SeqTrainer. It includes functions for loading the data,
    calling the preprocessing functions and creating the datasets which can be
    directly used in the custom WeightedSeq2SeqTrainer.

    """

    def __init__(self, config: dict):
        """
        Initialize the ByT5Trainer with a configuration dictionary.

        Args:
            config (dict): Training configuration. Expected keys include:
                - base_model_name (str): HuggingFace model identifier
                - pretrained_model (str, optional): Path to a pretrained checkpoint
                - task (str): One of 'translation', 'reconstruction', 'span_corruption'
                - dropout (float): Dropout rate for the model
                - ...and others that can be seen in byt5_main.py

        """
        self.config = config
        self.task = self.config["task"]
        self.wandb_logger = self._setup_wandb()

        model_path = config.get("pretrained_model") or config["base_model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, dropout_rate=self.config["dropout"]
        )
        self.augmentor = DataAugmentation(
            morphemes_csv_path=self.config["morphemes_csv_path"],
            lexicon_csv_path=self.config["lexicon_csv_path"],
            verbose=True,
        )

        self.chrf_metric = evaluate.load("chrf")
        self.bleu_metric = evaluate.load("sacrebleu")

        # For tables
        self.eval_count = 0

    def _setup_wandb(self):
        return wandb.init(
            entity="scaly",
            project=self.config["model_name"],
            name=self.config["run_name"],
            config=self.config,
            reinit=True,
        )

    def preprocess(self, data):
        """
        Performs preprocessing on transliteration and translation columns in the
        given dataframe. For the reconstruction task, it also includes
        augmenting the data with noisy gap and token columns.

        Args:
            data (pd.DataFrame): Dataframe to be preprocessed

        Returns:
            pd.DataFrame: Modified dataframe
        """

        tp = TextProcessor()

        data["transliteration"] = data["transliteration"].apply(
            lambda x: tp.preprocess_transliteration_text(
                transliteration_text=x,
                separate_compounds=self.config["separate_compounds"],
                with_hyphens=self.config["with_hyphens"],
                named_determinatives=self.config["named_determinatives"],
                normalize_chars=self.config["normalize_chars"],
                diacritic_mode=self.config["diacritic_mode"],
            )
        )

        if self.task == "translation":
            data["translation"] = data["translation"].apply(
                lambda x: tp.preprocess_translation_text(x)
            )

        if self.task == "reconstruction":
            data["noisy_gaps"] = data["transliteration"].apply(
                lambda x: self.augmentor.add_gap_noise(x)
            )

            data["noisy_tokens"] = data["transliteration"].apply(
                lambda x: self.augmentor.add_token_noise(x)
            )

        return data

    def dataset_creation(self):
        """
        Routes dataset creation to the appropriate method based on the training configuration.

        Returns:
            Dataset: A dataset created by one of three methods:
                - `dataset_pretraining_dictionary()` if 'pretraining_dictionary' is configured.
                - `dataset_pretraining_external()` if 'pretraining_external' is configured.
                - `dataset_creation_data()` for standard training (default).

        """

        if self.config["pretraining_dictionary"]:
            return self.dataset_pretraining_dictionary()

        elif self.config["pretraining_external"]:
            return self.dataset_pretraining_external()

        else:
            return self.dataset_creation_data()

    def dataset_pretraining_dictionary(self):
        """
        Builds a dataset for pretraining using dictionary data only.

        Loads and preprocesses the dictionary data. No train/test split
        is performed and no sample weighting is applied.

        Returns:
            tuple:
                - dict_dataset (AkkadianTranslationDatasetT5): The dictionary dataset.
                - None: No test dataset.
                - None: No sample weights.
        """

        dictionary = pd.read_csv(self.config["dictionary_path"])
        dictionary = self.preprocess(dictionary)

        dict_dataset = AkkadianTranslationDatasetT5(
            dataframe=dictionary,
            max_length=self.config["max_length"],
            tokenizer=self.tokenizer,
            task=self.task,
        )

        return dict_dataset, None, None

    def dataset_pretraining_external(self):
        """
        Builds train and test datasets for pretraining using external data only.
        Loads and preprocesses the external training data, then splits it into
        train (90%) and test (10%) sets. No sample weighting is applied.

        Returns:
            tuple:
                - train_dataset (AkkadianTranslationDatasetT5): The training
                dataset.
                - test_dataset (AkkadianTranslationDatasetT5): The validation
                dataset.
                - None: Returned in place of sample weights, as no weighting is
                used.
        """
        external = pd.read_csv(self.config["external_train_data_path"])
        external = self.preprocess(external)

        # In this case we do want to evaluate
        train_df, test_df = train_test_split(external, test_size=0.1)

        #
        train_dataset = AkkadianTranslationDatasetT5(
            dataframe=train_df,
            max_length=self.config["max_length"],
            tokenizer=self.tokenizer,
            task=self.task,
        )

        test_dataset = AkkadianTranslationDatasetT5(
            dataframe=test_df,
            max_length=self.config["max_length"],
            tokenizer=self.tokenizer,
            task=self.task,
        )

        return train_dataset, test_dataset, None

    def dataset_creation_data(self):
        """
        Builds train and test datasets for standard training from internal,
        external, dictionary, and onomasticon data sources.

        Loads the internal training data and optionally a third dataset,
        external data, a dictionary, and an onomasticon based on the config. The
        internal data is split into train (90%) and test (10%) sets. Optionally
        applies name-swap augmentation to the training set. All sources are
        preprocessed and concatenated into a final training dataframe, with
        per-sample weights assigned based on data source. For span corruption
        tasks, the input length is expanded to account for noise.

        Returns:
            tuple:
            - train_dataset (AkkadianTranslationDatasetT5): The training dataset.
            - test_dataset (AkkadianTranslationDatasetT5): The validation dataset,
            built from the internal data split only.
            - sample_weights (torch.Tensor or None): Per-sample weights reflecting
            the configured source weights. None if task is 'span_corruption'.
        """
        sample_weights = None

        # Define internal and maybe external
        internal = pd.read_csv(self.config["internal_train_data_path"])
        internal_new = (
            pd.read_csv(self.config["third_data_path"])
            if self.config["third_data_path"]
            else None
        )

        # Concatenate two internal datasets
        if internal_new is not None:
            internal_new = internal_new.sample(
                n=self.config["amount_of_internal_new_data"], random_state=42
            )
            internal = pd.concat([internal, internal_new]).reset_index(drop=True)

        external = (
            pd.read_csv(self.config["external_train_data_path"])
            if self.config["use_external_data"]
            else None
        )
        dictionary = (
            pd.read_csv(self.config["dictionary_path"])
            if self.config["use_dictionary"]
            else None
        )

        onomasticon = (
            pd.read_csv(self.config["onomasticon_path"])
            if self.config["use_onomasticon"]
            else None
        )

        # Split the INTERNAL into train and test(validation) = we evaluate only on the internal data
        train_internal_df, test_df = train_test_split(internal, test_size=0.1)

        # Only do nameswapping on internal training data
        if self.config["name_swapping"]:
            train_internal_df = self.augmentor.name_swap_augmentation(train_internal_df)

        # Preprocess both datasets
        train_internal_df = self.preprocess(train_internal_df)
        test_df = self.preprocess(test_df)

        if external is not None:
            external = external.sample(
                n=self.config["amount_of_external_data"], random_state=42
            )
            external = self.preprocess(external)

        if dictionary is not None:

            if onomasticon is not None:
                dictionary = pd.concat([dictionary, onomasticon]).reset_index(drop=True)

            dictionary = self.preprocess(dictionary)

        # Building weights
        sources = [(train_internal_df, "internal_weight")]

        if external is not None:
            sources.append((external, "external_weight"))

        if dictionary is not None:
            sources.append((dictionary, "dictionary_weight"))

        train_df = pd.concat([df for df, _ in sources]).reset_index(drop=True)

        if self.task == "span_corruption":
            sample_weights = None
        else:
            sample_weights = torch.cat(
                [
                    torch.full((len(df),), self.config[weight_key])
                    for df, weight_key in sources
                ]
            )

        # For span corruption, the max_length needs to be expanded because corruption decreases it
        expanded_length = (
            self.compute_expanded_input_length(
                self.config["max_length"],
                self.config["noise_density"],
                self.config["mean_noise_span_length"],
            )
            if self.config["task"] == "span_corruption"
            else self.config["max_length"]
        )
        # Create datasets based on the task
        train_dataset = AkkadianTranslationDatasetT5(
            dataframe=train_df,
            max_length=expanded_length,
            tokenizer=self.tokenizer,
            task=self.task,
        )

        test_dataset = AkkadianTranslationDatasetT5(
            dataframe=test_df,
            max_length=expanded_length,
            tokenizer=self.tokenizer,
            task=self.task,
        )

        return train_dataset, test_dataset, sample_weights

    def compute_expanded_input_length(
        self, input_length, noise_density, mean_noise_span_length
    ):
        """
        Computes the pre-corruption input length needed to yield the desired
        post-corruption length after span corruption noise is applied.

        Span corruption replaces noise spans with sentinel tokens, which shortens
        the sequence. This method works backwards by incrementally expanding the
        input length until the simulated post-corruption length matches the target.

        Args:
            input_length (int): The desired sequence length after corruption.
            noise_density (float): Fraction of tokens to be corrupted (e.g. 0.15).
            mean_noise_span_length (float): Average length of each corrupted span.

        Returns:
            int: The expanded input length to use before corruption
        """
        expanded = input_length
        while True:
            num_noise_tokens = round(expanded * noise_density)
            num_noise_tokens = min(max(num_noise_tokens, 1), expanded - 1)
            num_spans = round(num_noise_tokens / mean_noise_span_length)
            num_spans = max(num_spans, 1)
            # post-corruption length = expanded - noise_tokens + spans (sentinels)
            result = expanded - num_noise_tokens + num_spans
            if result == input_length:
                return expanded
            expanded += 1

    def get_datacollator(self):
        """
        Returns the appropriate data collator based on the configured task.

        For span corruption, uses custom DataCollatorForT5MLM, computing the target
        sequence length as the number of corrupted tokens plus their sentinel
        tokens plus one EOS token. For all other tasks, uses the standard
        DataCollatorForSeq2Seq.

        Returns:
            DataCollatorForT5MLM: If task is 'span_corruption'.
            DataCollatorForSeq2Seq: For all other tasks.
        """

        def compute_target_length(input_length, noise_density, mean_noise_span_length):
            # need to compute the length of the target for the T5MLM data collator

            noisy_tokens = round(input_length * noise_density)
            special_tokens = round(noisy_tokens / mean_noise_span_length)

            return noisy_tokens + special_tokens + 1  # 1 is for EOS token

        if self.config["task"] == "span_corruption":
            return DataCollatorForT5MLM(
                tokenizer=self.tokenizer,
                noise_density=self.config["noise_density"],
                mean_noise_span_length=self.config["mean_noise_span_length"],
                input_length=self.config["max_length"],
                target_length=compute_target_length(
                    self.config["max_length"],
                    self.config["noise_density"],
                    self.config["mean_noise_span_length"],
                ),
                decoder_start_token_id=self.model.config.decoder_start_token_id,
            )
        else:
            return DataCollatorForSeq2Seq(self.tokenizer, self.model)

    def compute_metrics(self, eval_preds):
        """
        Computes ChrF, BLEU, and their geometric mean for a set of evaluation predictions.

        Decodes predicted and label token IDs, replacing -100 padding markers with
        the tokenizer's pad token before decoding. On the final evaluation epoch,
        logs a sample of up to 50 reference/hypothesis pairs to Weights & Biases.

        Args:
            eval_preds (tuple): A tuple of (predictions, labels) as returned by the
                Hugging Face Trainer.

        Returns:
            dict: A dictionary with keys 'chrf', 'bleu', and 'geo_mean'
        """
        self.eval_count += 1

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Ignore -100 in the labels.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Logging to WANDB
        if (
            self.eval_count * self.config["eval_every_n_epochs"]
            >= self.config["num_epochs"]
        ):
            table = wandb.Table(columns=["reference", "hypothesis"])
            for ref, hyp in zip(decoded_labels[:50], decoded_preds[:50]):
                table.add_data(ref, hyp)
            self.wandb_logger.log({"translation_samples": table})

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        chrf_result = self.chrf_metric.compute(
            predictions=decoded_preds, references=decoded_labels, word_order=2
        )
        bleu_result = self.bleu_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        if chrf_result != None and bleu_result != None:
            geo_mean = np.sqrt(
                chrf_result["score"] * bleu_result["score"] + 1e-8
            )  # Adding a small value to avoid multiplication by zero
            return {
                "chrf": chrf_result["score"],
                "bleu": bleu_result["score"],
                "geo_mean": geo_mean,
            }
        else:
            return {}

    def train(self):
        """
        Configures and runs the training loop based on the configured task and training mode.

        Builds train and eval datasets via dataset_creation(), then optionally wraps
        the model with LoRA adapters. Computes evaluation steps dynamically based on
        dataset size, number of devices, batch size, and gradient accumulation steps.

        Training behaviour varies by mode:
            - span_corruption: Trains without generation; uses eval loss as the best
                model metric.
            - pretraining_dictionary: Trains without evaluation; saves per epoch.
            - standard: Trains with generation and uses geometric mean of ChrF and
                BLEU as the best model metric.

        After training, if LoRA was used, the adapters are merged into the base model
        before saving. Both the model and tokenizer are saved to the configured output
        directory.
        """

        train_dataset, eval_dataset, sample_weights = self.dataset_creation()

        if self.config["use_lora"]:
            lora_config = LoraConfig(
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                target_modules=self.config["lora_target_modules"],
                lora_dropout=self.config["lora_dropout"],
                bias=self.config["lora_bias"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            peft_model = get_peft_model(self.model, lora_config)
            self.model = cast(PreTrainedModel, peft_model)

        # Calculate the amount of steps in each epoch
        num_devices = torch.cuda.device_count()
        steps_per_epoch = math.ceil(
            len(train_dataset)
            / (
                self.config["train_batch_size"]
                * num_devices
                * self.config["gradient_accumulation_steps"]
            )
        )  # 2 for two gpu's we're using on the cluster + account for gradient accumulation
        eval_steps = steps_per_epoch * self.config["eval_every_n_epochs"]

        if self.task == "span_corruption":
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.config["output_dir"],
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=eval_steps,
                save_steps=eval_steps,
                logging_strategy="epoch",
                learning_rate=self.config["learning_rate"],
                per_device_train_batch_size=self.config["train_batch_size"],
                per_device_eval_batch_size=self.config["eval_batch_size"],
                num_train_epochs=self.config["num_epochs"],
                predict_with_generate=False,  # no generations!
                logging_dir="./logs",
                report_to="wandb",
                logging_first_step=True,
                remove_unused_columns=False,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",  # geo mean makes no sense here, we use validation loss
                greater_is_better=False,  # for loss, lower is better
                fp16=self.config["fp16"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            )

            trainer = WeightedSeq2Seq(
                model=self.model,  # consider model_init for HYPERPARAMETER SEARCH
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=None,  # we just use loss
                data_collator=self.get_datacollator(),
                sample_weights=sample_weights,
            )

        elif self.config["pretraining_dictionary"]:
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.config["output_dir"],
                eval_strategy="no",
                save_strategy="epoch",
                logging_strategy="epoch",
                learning_rate=self.config["learning_rate"],
                per_device_train_batch_size=self.config["train_batch_size"],
                num_train_epochs=self.config["num_epochs"],
                logging_dir="./logs",
                report_to="wandb",
                logging_first_step=True,
                remove_unused_columns=False,
                save_total_limit=2,
                fp16=self.config["fp16"],
                gradient_accumulation_steps=self.config[
                    "gradient_accumulation_steps"
                ],  # run this
                gradient_checkpointing=self.config["gradient_checkpointing"],
            )

            # Custom Class - child from the hugging face Seq2SeqTrainer
            trainer = WeightedSeq2Seq(
                model=self.model,  # consider model_init for HYPERPARAMETER SEARCH
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=self.get_datacollator(),
            )

        else:
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.config["output_dir"],
                eval_strategy="steps",
                save_strategy="steps",
                eval_steps=eval_steps,
                save_steps=eval_steps,
                logging_strategy="epoch",
                learning_rate=self.config["learning_rate"],
                per_device_train_batch_size=self.config["train_batch_size"],
                per_device_eval_batch_size=self.config["eval_batch_size"],
                num_train_epochs=self.config["num_epochs"],
                predict_with_generate=True,
                generation_max_length=self.config["generation_max_length"],
                logging_dir="./logs",
                report_to="wandb",
                logging_first_step=True,
                remove_unused_columns=False,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="geo_mean",
                greater_is_better=True,
                fp16=self.config["fp16"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                gradient_checkpointing=self.config["gradient_checkpointing"],
                seed=42,
                data_seed=42,
            )

            # Custom Class - child from the hugging face Seq2SeqTrainer

            trainer = WeightedSeq2Seq(
                model=self.model,  # consider model_init for HYPERPARAMETER SEARCH
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics,
                data_collator=self.get_datacollator(),
                sample_weights=sample_weights,
            )

        trainer.train(resume_from_checkpoint=True)

        if self.config["use_lora"]:
            lora_model: PeftModel = trainer.model  # type: ignore
            merged_model = lora_model.merge_and_unload()  # type: ignore
            merged_model.save_pretrained(self.config["output_dir"])
        else:
            trainer.save_model(self.config["output_dir"])

        self.tokenizer.save_pretrained(self.config["output_dir"])


class WeightedSeq2Seq(Seq2SeqTrainer):
    """
    A Seq2SeqTrainer subclass that supports weighted sampling during training.

    When sample weights are provided, replaces the default training dataloader
    with one that uses a WeightedRandomSampler, allowing data sources with
    different importance weights to be sampled proportionally. Falls back to
    the standard dataloader if no weights are given.

    Args:
        *args: Positional arguments passed to Seq2SeqTrainer.
        sample_weights (torch.Tensor, optional): A 1D tensor of per-sample weights
            used to construct the WeightedRandomSampler. If None, standard
            sequential sampling is used.
        **kwargs: Keyword arguments passed to Seq2SeqTrainer.
    """

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self) -> DataLoader:
        """
        Builds the training dataloader, using weighted random sampling if
        sample weights were provided, otherwise delegating to the parent class.

        Returns:
            DataLoader: A dataloader with WeightedRandomSampler if sample weights
                are set, otherwise the default Seq2SeqTrainer dataloader.

        Raises:
            ValueError: If sample weights are set but no training dataset is found.
        """
        if self.sample_weights is None:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Training dataset is not set")

        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        train_dataset = cast(Dataset, self.train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# Data Collator for Span Corruption
class DataCollatorForT5MLM:
    """
    This class was fully copied from https://huggingface.co/flax-community/t5-base-dutch/blob/main/run_t5_mlm_flax.py
    Minimal changes: implementation of the shift_tokens_right function and changing how its called (removing the pad_token_id requirement).



    -----------------START OF THE ORIGINAL DOCS------------------------------------------
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    def __init__(
        self,
        tokenizer,
        noise_density,
        mean_noise_span_length,
        input_length,
        target_length,
        decoder_start_token_id,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = np.array(batch["input_ids"])
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # CUSTOM
        batch["input_ids"] = batch["input_ids"][:, : self.input_length]
        batch["labels"] = batch["labels"][:, : self.target_length]

        # if batch["input_ids"].shape[-1] != self.input_length:
        #     raise ValueError(
        #         f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
        #     )

        # if batch["labels"].shape[-1] != self.target_length:
        #     raise ValueError(
        #         f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
        #     )

        # to check that tokens are correctly proprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = self.shift_tokens_right(
            batch["labels"], self.decoder_start_token_id
        )

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def shift_tokens_right(self, input_ids, decoder_start_token_id):
        """
         Shifts token IDs one position to the right, inserting the decoder start
        token at the beginning. Used to construct decoder input IDs from labels
        for teacher forcing during seq2seq training.

        Args:
            input_ids (np.ndarray): 2D array of token IDs of shape (batch_size, seq_len).
            decoder_start_token_id (int): Token ID to insert at position 0 of each sequence.

        Returns:
            np.ndarray: Shifted token IDs of the same shape as input_ids.
        """
        shifted = np.zeros_like(input_ids)
        shifted[:, 1:] = input_ids[:, :-1]  # -1 to not include the last token
        shifted[:, 0] = decoder_start_token_id
        return shifted
