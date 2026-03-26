import os
import tempfile

import pandas as pd
from src.data_processing.datasets import AkkadianEnglishDataset
import torch
from src.models.mBART.mBartFineTuner import mBartFineTuner
from src.data_processing.processing import TextProcessor
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from src.config import Columns


import lightning as L


class KaggleInference:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.model.eval()

    def load_and_prepare_dataset(
        self,
        test_data_path="/kaggle/input/deep-past-initiative-machine-translation/test.csv",
        **text_processor_args,
    ):
        """
        [TODO:description]

        :param tokenizer [TODO:type]:  Tokenizer from the model being used
        :param max_samples [TODO:type]: Max samples to use from the dataset. Useful for overfitting a single batch

        """
        self.df = pd.read_csv(test_data_path)
        tp = TextProcessor()

        df_copy = self.df.copy()

        # For each value in the row, apply preprocess_input_text()
        df_copy["transliteration"] = df_copy["transliteration"].apply(
            lambda x: tp.preprocess_transliteration_text(
                transliteration_text=x,
                **text_processor_args,
            )
        )
        self.dataset = AkkadianEnglishDataset(
            df_copy, self.tokenizer, max_length=512, is_inference=True
        )

    def create_dataloaders(self, batch_size=4, num_workers=4):
        """
        Returns the dataloaders for training
        :param batch_size
        :param num_workers
        """

        # The data collator tokenizes the batch before the foward pass.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

        # Create DataLoaders
        test_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=num_workers,
        )

        self.test_loader = test_loader

    def inference(
        self,
        use_mbr,
        mbr_num_beam_cands,
        mbr_num_sample_cands,
        early_stopping,
        length_penalty,
        no_repeat_ngram_size,
        num_beams,
        repetition_penalty,
    ):
        """
        Runs the full inference pipeline and returns all translated sentences
        without postprocessing.

        """
        self.model.hparams["use_mbr"] = use_mbr
        self.model.hparams["mbr_num_beam_cands"] = mbr_num_beam_cands
        self.model.hparams["mbr_num_sample_cands"] = mbr_num_sample_cands
        self.model.hparams["num_beams"] = num_beams
        self.model.hparams["length_penalty"] = length_penalty
        self.model.hparams["no_repeat_ngram_size"] = no_repeat_ngram_size
        self.model.hparams["early_stopping"] = early_stopping
        self.model.hparams["repetition_penalty"] = repetition_penalty

        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",  # use all the gpus available
        )
        predictions = trainer.predict(self.model, dataloaders=self.test_loader)

        if predictions is None:
            exit("Error while translating in inference")

        flat_translations = [
            str(sentence) for batch in predictions for sentence in batch
        ]

        # Build the Kaggle-valid DataFrame
        # We use self.test_df['id'] to ensure IDs match the translations 1:1
        results_df = pd.DataFrame(
            {"id": self.df["id"].values, "translation": flat_translations}
        )

        return results_df
