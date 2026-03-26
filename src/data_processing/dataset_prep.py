"""

Deprecated class, soon to be removed


"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import ALIGNMENT_PATH, LEXICON_PATH, MORPHEMES_PATH
import pandas as pd
from tqdm import tqdm
import argparse

from src.data_processing.alignment import Aligner
from data_processing.augmentation import DataAugmentation


class DatasetPrep:
    def __init__(
        self,
        lexicon_csv_path=str(LEXICON_PATH),
        morphemes_csv_path=str(MORPHEMES_PATH),
        alignment_csv_path=str(ALIGNMENT_PATH),
        verbose: bool = False,
    ):
        self.aligner = Aligner()
        self.augmenter = DataAugmentation(
            lexicon_csv_path=lexicon_csv_path if lexicon_csv_path else "",
            morphemes_csv_path=morphemes_csv_path if morphemes_csv_path else "",
        )
        self.V = verbose
        self.alignment_path = alignment_csv_path

    def align_data(
        self, train_df: pd.DataFrame, alignment_guide_df: pd.DataFrame, doc_limit=512
    ) -> pd.DataFrame:
        if self.V:
            print(f"Starting alignment for {len(train_df)} tablets...")

        # lookup for ids
        id_dict = {
            k: g.sort_values("first_word_number").reset_index(drop=True)
            for k, g in alignment_guide_df.groupby("text_uuid")
        }

        processed_rows = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Aligning"):
            aligned_segments, log = self.aligner.compare_and_align(
                row, id_dict, accept_worst_eng=True
            )
            processed_rows.extend(aligned_segments)

        all_data = pd.DataFrame(processed_rows)

        aligned_sentences = all_data[all_data["level"] == "sentence"].copy()

        short_documents = all_data[
            (all_data["level"] == "document")
            & (all_data["transliteration"].str.len() <= doc_limit)
        ].copy()

        final_df = pd.concat([aligned_sentences, short_documents], ignore_index=True)
        assert isinstance(final_df, pd.DataFrame)

        final_df = final_df.dropna(subset=["transliteration", "translation"])
        final_df = final_df[final_df["transliteration"].str.strip() != ""]

        if "oare_id" in final_df.columns:
            # named id for consistency
            final_df["id"] = final_df["oare_id"]

        if self.V:
            print(
                f"Alignment complete. Extracted {len(aligned_sentences)} sentences and {len(short_documents)} short docs."
            )

        assert isinstance(final_df, pd.DataFrame)
        return final_df

    
    def run_pipeline(
        self,
        train_path,
        do_align=True,
        do_augment=True,
        accept_worst_eng=False,
        output_path="",
    ):
        """Preps the dataset by (optionally) alligning the data and augmenting with name swaps

        Args:
            train_path (str): path to training set
            alignment_path (str, optional): path toalignment path. Defaults to None.
            do_align (bool, optional): if true data is aligned. Defaults to True.
            do_augment (bool, optional): if true data is alligned with pn and gn swapping. Defaults to True.
            accept_worst_eng (bool, optional): if true 500 extra sentences are aligned but some
                quality of translation is lost
            output_path (str): path to save new csv. If not empty, creates an output csv

        Returns:
            df: prepped train df
        """
        # Load base data
        print(f"Loading training data from {train_path}...")
        df = pd.read_csv(train_path)

        # Step 1: Alignment
        # if do_align and self.alignment_path:
        #     guide_df = pd.read_csv(self.alignment_path)
        #     df = self.align_data(df, guide_df)
        # elif do_align and not self.alignment_path:
        #     if self.V:
        #         print(
        #             "Warning: Alignment requested but no alignment file provided. Skipping."
        #         )

        # Step 2: Augmentation
        if do_augment:
            print("Starting data augmentation with name swapping...")
            print("Rows before augmentation:", len(df))
            df = self.augmenter.name_swap_augmentation(df)
            print("Rows after augmentation:", len(df))

        # Final Export
        # Standardize columns to the required output format
        final_cols = ["id", "transliteration", "translation"]
        # Handle cases where 'id' might be named 'oare_id'
        if "id" not in df.columns and "oare_id" in df.columns:
            df["id"] = df["oare_id"]

        output_df = df[[c for c in final_cols if c in df.columns]]
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Pipeline finished. Saved to {output_path}")

        return output_df


# example usage
def main():
    parser = argparse.ArgumentParser(
        description="Akkadian Dataset Preparation Pipeline"
    )
    parser.add_argument("--train", required=True, help="Path to raw train.csv")
    parser.add_argument("--alignment", help="Path to alignment CSV")
    parser.add_argument("--lexicon", help="Path to lexicon CSV")
    parser.add_argument("--morphemes", help="Path to morphemes CSV")
    parser.add_argument(
        "--output", default="prepared_train.csv", help="Output filename"
    )
    parser.add_argument(
        "--skip-align", action="store_true", help="Skip the alignment step"
    )
    parser.add_argument(
        "--skip-augment", action="store_true", help="Skip the augmentation step"
    )

    args = parser.parse_args()

    prep = DatasetPrep()
    print("Starting dataset preparation pipeline...")
    prep.run_pipeline(
        train_path=args.train,
        do_align=not args.skip_align,
        do_augment=not args.skip_augment,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
