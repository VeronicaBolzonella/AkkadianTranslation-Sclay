from datetime import datetime
from typing import Dict

from data_processing.augmentation import DataAugmentation
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import ConcatDataset
from transformers import DataCollatorForSeq2Seq

from src.config import Columns, DatasetType, LEXICON_PATH, MORPHEMES_PATH, TrainingMode
from src.data_processing.alignment import Aligner
from src.data_processing.datasets import AkkadianEnglishDataset
from src.data_processing.processing import TextProcessor
from src.models.mBART.mBartFineTuner import mBartFineTuner
from src.utils import drop_empty_rows, stack_csvs_from_folder

L.seed_everything(42, workers=True)


class ModeAwareCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        # Extract training_mode before collation
        modes = [f.pop("training_mode") for f in features]

        # Collate the rest normally
        batch = self.base_collator(features)

        # Re-attach all samples in a batch share the same mode
        # because WeightedRandomSampler doesn't mix modes mid-batch
        batch["training_mode"] = modes

        return batch


class KaggleTrainer:
    """

    Example usage:

    See example in `mbart_kaggle_training.py`

    kt = KaggleTrainer(settings)

    kt.load_and_prepare_dataset(settings)
    kt.create_dataloaders(settings)
    kt.fit()
    """

    def __init__(self, config: Dict, checkpoint_dir: str = "/kaggle/working/mBART"):
        """Loads the model from a checkpoint using the `config`

        Args:
            checkpoint_dir : where the checkkpoint is
            config (dict): Example of a config dictionary is:
                        self.config = {
                            "learning_rate": 0.02,
                            "max_epochs": 200,
                            "batch_size": 4,
                            "training_type": "supervised",
                            "precision": 32,
                            "lora_r": 32,
                            "lora_alpha": 16,
                            "model_name": "mbart",
                            "early_stopping_patience": 30,
                            "save_model_every": 10,  # save the model every 10 epochs
                        }
        """
        self.checkpoint_dir = checkpoint_dir

        self.config = config
        self.wandb_logger = self._setup_wandb()
        self.model = self._create_model()
        self.datasets = []
        self.datasets_weights = []
        self.train_loader = None
        self.val_loader = None
        self.augmentor = DataAugmentation(
            morphemes_csv_path=str(MORPHEMES_PATH),
            lexicon_csv_path=str(LEXICON_PATH),
            verbose=True,
        )

    def _setup_wandb(self):
        wandb_logger = WandbLogger(
            entity="scaly",
            project=self.config["project_name"],
            name=self.config["model_name"],
            log_model=False,
            config=self.config,
        )
        return wandb_logger

    def _create_model(self):
        model = mBartFineTuner(
            lr=self.config["learning_rate"],
            src_lang="ak_XX",
            dropout=self.config["dropout"],
            attention_dropout=self.config["attention_dropout"],
            base_model_path=self.config["base_model_path"],
            eval_every=self.config["eval_every"],
            model_type=self.config["model_type"],
        )
        # tgt_lang is now handled per batch — no global assignment here
        return model

    def load_and_prepare_dataset(
        self,
        data_folder_path: str,
        model_type: str,
        weight: float,
        dataset_type: DatasetType,
        training_mode: TrainingMode,
        max_samples: int | None = None,
        name_swapping: bool = False,
        train_data_ratio: float = 0.9,
        **text_processor_args,
    ):
        """Applies preprocessing and data augmentation to the dataset.

        It stores the dataset and its weights in:
            - self.datasets
            - self.dataset_weights

        NOTE: You should call this function of each separete dataset you have.

        Args:
            data_folder_path: (str): The path where the dataset lives
            max_samples ([int,None]):  Max samples to use from the dataset. Useful for overfitting a single batch
            dataset_type (DatasetTypes): The type of the dataset. Depending on the type it will have a different weight.
                e.g., internal datasets have higher weight than external datasets.
        """

        df: pd.DataFrame = stack_csvs_from_folder(data_folder_path)
        df = drop_empty_rows(df)

        if max_samples is not None:
            df = df.head(max_samples)

        # The internal dataset must be aligned
        if dataset_type == DatasetType.INTERNAL and self.config["align"]:
            aligner = Aligner()
            df = aligner.align_data(df)

        train_df, val_df = self.split_and_augment(
            df,
            dataset_type=dataset_type,
            name_swapping=name_swapping,
            train_data_ratio=train_data_ratio,
            training_mode=training_mode,
        )
        tp = TextProcessor()

        for split in [train_df, val_df]:
            # For each value in the row, apply preprocess_input_text()
            split[Columns.TRANSLITERATION] = split[Columns.TRANSLITERATION].apply(
                lambda x: tp.preprocess_transliteration_text(
                    transliteration_text=x,
                    **text_processor_args,
                )
            )

            split[Columns.TRANSLATION] = split[Columns.TRANSLATION].apply(
                lambda x: tp.preprocess_translation_text(translation_text=x)
            )

        if model_type == "mt5" or model_type == "akk_300m":
            if model_type == "mt5":
                task_prefix = "Translate Akkadian to English: "
            elif model_type == "akk_300m":
                task_prefix = "Translate complex Akkadian transliteration to English "

            if training_mode == TrainingMode.SELF_SUPERVISED:
                task_prefix = (
                    "Reconstruct the original Akkadian sentence by removing noise:"
                )
        else:
            task_prefix = ""

        # Build datasets
        train_dataset = AkkadianEnglishDataset(
            train_df,
            self.model.tokenizer,
            training_mode=training_mode,
            is_inference=False,
            dataset_type=dataset_type,
            max_length=self.config["max_length"],
            task_prefix=task_prefix,
            eng_tokenizer=self.model.eng_tokenizer,
            weight=weight,
        )
        val_dataset = AkkadianEnglishDataset(
            val_df,
            self.model.tokenizer,
            training_mode=training_mode,
            is_inference=False,
            dataset_type=dataset_type,
            max_length=self.config["max_length"],
            task_prefix=task_prefix,
            eng_tokenizer=self.model.eng_tokenizer,
            weight=weight,
        )

        self.datasets.append(
            {
                "train_dataset": train_dataset,
                "train_weights": train_dataset.get_weights(),
                "val_dataset": val_dataset,
                "dataset_type": dataset_type,
            }
        )

    def split_and_augment(
        self,
        df,
        dataset_type: DatasetType,
        name_swapping: bool,
        train_data_ratio: float,
        training_mode: TrainingMode,
    ):
        """Splits the dataframe into validation and training. Applies augmentation to tranining

        If the dataset_type something other than "INTERNAL" do not have a validation split.
        This is prefered because it ensures the validation set does not change.

        """

        # If the dataset_type something other than "INTERNAL" do not have a validation split.
        if dataset_type != DatasetType.INTERNAL:
            train_data_ratio = 1

        # Shuffle !
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the DataFrame first
        train_len = int(train_data_ratio * len(df))
        train_df = df.iloc[:train_len].copy()

        val_df = df.iloc[train_len:].copy()

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Length of {dataset_type} dataset {len(df)}")
            print(f"Length of {dataset_type} train dataset {len(train_df)}")
            print(f"Length of {dataset_type} validation dataset {len(val_df)}")

        # Create the selfsupervise dataset
        if training_mode == TrainingMode.SELF_SUPERVISED:
            for split in [train_df, val_df]:
                split[Columns.NOISY_GAPS] = split[Columns.TRANSLITERATION].apply(
                    lambda x: self.augmentor.add_gap_noise(x)
                )
                split[Columns.NOISY_TOKENS] = split[Columns.TRANSLITERATION].apply(
                    lambda x: self.augmentor.add_token_noise(x)
                )

        # Augment only the training split
        if name_swapping and dataset_type == DatasetType.INTERNAL:
            print(f"Before nameswapping: {len(train_df)}")
            train_df = self.augmentor.name_swap_augmentation(train_df)
            print(f"After nameswapping: {len(train_df)}")

        if dataset_type != DatasetType.INTERNAL:
            assert len(val_df) == 0, (
                "For datasets other than internal, there should be no validation dataframe. length = 0"
            )
        return train_df, val_df

    def create_dataloaders(self, batch_size=4, num_workers=4):
        if not self.datasets:
            raise ValueError(
                "No datasets loaded. Call load_and_prepare_dataset() first."
            )

        train_datasets = []
        val_datasets = []

        for entry in self.datasets:
            train_datasets.append(entry["train_dataset"])
            val_datasets.append(entry["val_dataset"])

        train_data = ConcatDataset(train_datasets)
        val_data = ConcatDataset(val_datasets)

        # Build weights so each dataset's share of batches equals its weight / total_weight
        total_weight = sum(entry["train_dataset"].weight for entry in self.datasets)
        train_weights_all = []
        for entry in self.datasets:
            dataset_prob = entry["train_dataset"].weight / total_weight
            per_sample_weight = dataset_prob / len(entry["train_dataset"])
            train_weights_all.extend([per_sample_weight] * len(entry["train_dataset"]))

        print(f"Total training samples: {len(train_data)}")
        print(f"Total weights: {len(train_weights_all)}")

        sampler = WeightedRandomSampler(
            weights=train_weights_all,
            num_samples=len(train_weights_all),
            replacement=True,
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model.tokenizer,
            model=self.model,
        )
        mode_aware_collator = ModeAwareCollator(data_collator)

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=mode_aware_collator,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            collate_fn=mode_aware_collator,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False,
        )

    def _create_trainer(self):
        timestamp = datetime.now().strftime("%d-%Hh")

        monitor_configs = {
            "geo": ("val_geo_score", "max"),
            "val_loss": ("val_loss", "min"),
            "train_loss": ("train_loss", "min"),
        }

        if self.config["checkpoint_monitor"] not in monitor_configs:
            raise ValueError(
                f"Unknown checkpoint_monitor: {self.config['checkpoint_monitor']}"
            )

        monitor_metric, monitor_mode = monitor_configs[
            self.config["checkpoint_monitor"]
        ]

        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"metric={{{monitor_metric}:.3f}}-{{epoch:02d}}-{timestamp}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            every_n_epochs=self.config["save_model_every"],
            save_last=True,
        )
        # Build callbacks list
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            checkpoint,
        ]

        # Conditionally add early stopping
        if self.config["early_stopping_patience"] != 0:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                min_delta=0.1,
                patience=self.config["early_stopping_patience"],
                verbose=True,
                mode=monitor_mode,
            )
            callbacks.append(early_stop_callback)

        # Create Trainer
        trainer = L.Trainer(
            max_epochs=self.config["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.config["gpu_num"],  # use all the gpus available
            strategy=DDPStrategy(find_unused_parameters=True),
            logger=self.wandb_logger,
            log_every_n_steps=10,
            callbacks=callbacks,
            precision=self.config["precision"],
        )

        return trainer

    def fit(self):
        if not self.datasets:
            exit("Run first `load_and_prepare_dataset()`")

        if not self.train_loader:
            exit("Run first `create_dataloaders()`")

        trainer = self._create_trainer()
        self.model.train()

        layer = self.model.model.model.encoder.layers[0]
        assert layer.dropout > 0, f"Dropout not set! Got {layer.dropout}"
        assert self.model.training, "Model not in train mode!"

        trainer.fit(
            self.model,
            self.train_loader,
            self.val_loader,
            ckpt_path=self.config["checkpoint_path"],
        )
