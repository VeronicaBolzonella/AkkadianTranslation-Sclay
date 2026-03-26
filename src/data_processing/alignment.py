import os
import re
from src.config import ALIGNMENT_PATH
import pandas as pd
import numpy as np
from tqdm import tqdm


class Aligner:
    def __init__(self, alignment_path=ALIGNMENT_PATH, verbose: bool = False) -> None:
        self.verbose = verbose
        self.alignment_guide = pd.read_csv(alignment_path)

    def normalize_akkadian(self, transliteration):
        """
        Just temporary basic normalization of weird characters, more is not needed, helps cleaner matching because in the
        alignment data some characters are siplified
        """
        if not isinstance(transliteration, str):
            return ""
        transliteration = transliteration.lower()
        replacements = {
            "š": "s",
            "ṣ": "s",
            "ḫ": "h",
            "ṭ": "t",
            "ā": "a",
            "ē": "e",
            "ī": "i",
            "ū": "u",
            "(d)": "",
            ".": "",
            "-": "",
        }
        for char, rep in replacements.items():
            transliteration = transliteration.replace(char, rep)
        # remove all non-alphanumeric
        return re.sub(r"[^a-z0-9]", "", transliteration)

    def normalize_english_for_search(self, translation):
        """normalization for English to compare alignment dataset and normal dataset
        (e.g. different use of quotations)"""
        if not isinstance(translation, str):
            return ""
        return re.sub(r"[^a-z0-9]", "", translation.lower())

    def split_target_nuclear(self, target):
        """
        Split english translations into segments at: punctuation, colons, and quoted speech boundaries
        (give most matches).
        """
        if pd.isna(target):
            return []

        text = re.sub(
            r":(?!\s)", ": ", target
        )  # add space after colons when missing (Luca found some that did)

        # same for other punctuations just to be safe, have't checked if it makes a difference
        text = re.sub(r'([?!"\'])([A-Z])', r"\1 \2", text)

        # split pattern for all punctuations mentioned
        pattern = r'(?<=[.!?])\s+|(?<=[:;])\s+|(?<=[?!]")\s*(?=[A-Z])'

        # Added in iteration 2:
        #   - Splitting on unicode capitals as well (not only ASCII)
        #   - Added ." as split pattern as well
        #   - Worsens performance
        # pattern = r'(?<=[.!?])\s+|(?<=[:;])\s+|(?<=[.?!]")\s*(?=[A-Z\u0100-\uffff])'

        # Iteration 3:
        #   - Split pattern without colons
        # pattern = r'(?<=[.!?])\s+|(?<=;)\s+|(?<=[?!]")\s*(?=[A-Z])'

        return [t.strip() for t in re.split(pattern, text) if t.strip()]

    def slice_transliteration(self, source_text, alignment_rows):
        """
        Use the word indices from alignment data to cut the transliteration.
        """
        words = source_text.split()
        sliced_sources = []

        for i in range(len(alignment_rows)):
            curr_start_idx = alignment_rows.iloc[i]
            next_idx = (
                alignment_rows.iloc[i + 1] if i + 1 < len(alignment_rows) else None
            )  # none at last part

            if pd.isna(curr_start_idx["first_word_number"]):
                # check: csv doesn't have a word number for this sentence, can't split it
                sliced_sources.append("")
                continue

            start_index = int(curr_start_idx["first_word_number"]) - 1

            if next_idx is not None and not pd.isna(next_idx["first_word_number"]):
                end_index = int(next_idx["first_word_number"]) - 1
            else:
                end_index = len(words)

            sliced_sources.append(" ".join(words[start_index:end_index]))

        return sliced_sources

    def verify_by_akkadian_word(self, train_row, alignment_rows):
        """
        Safety check.
        Verify if the word at the index provided by the alignment data matches the word in the actual transliteration.
        """
        train_words = train_row["transliteration"].split()

        for j, (_, row) in enumerate(alignment_rows.iterrows()):
            idx = int(row["first_word_number"]) - 1
            expected = self.normalize_akkadian(row["first_word_spelling"])

            if idx >= len(train_words):
                # Returning logging info
                return False, {
                    "mismatch_type": "index_out_of_range",
                    "sentence_row_idx": int(j),
                    "word_index": int(idx),
                    "n_train_words": int(len(train_words)),
                    "expected": expected,
                }

            actual = self.normalize_akkadian(train_words[idx])
            if actual != expected:
                return False, {
                    "mismatch_type": "word_mismatch",
                    "sentence_row_idx": int(j),
                    "word_index": int(idx),
                    "expected": expected,
                    "actual": actual,
                    "train_word_raw": train_words[idx],
                    "guide_word_raw": row["first_word_spelling"],
                }
        return True, {}

    def compare_and_align(
        self,
        train_row,
        guide_by_id,
        accept_worst_eng: bool = False,
        debug: bool = False,
    ):
        """
        Align logic for a doc:
        1. if no information on sentence, keep as is.
        2. If akkadian words dont match info is wrong: keep as is
        3. try splitting using original dataset -> count matches -> perfect alignment
        4. Fallback to alignment English if split count fails (if accept_worst_eng=True).

        Note:
            38 are not matchin in step 2
            191 are matched perfectly
            worst english acceptance aligns over 500 more sentences
        """
        oare_id = train_row["id"]

        # Logging for debugging
        log = {
            "oare_id": oare_id,
            "status": None,
            "n_oare_rows": None,
            "n_split_eng": None,
            "akk_ok": None,
            "akk_fail_info": None,
            "split_preview": None,
            "slice_preview": None,
            "note": None,
        }

        # point 1: no info available
        if oare_id not in guide_by_id:
            row_data = train_row.to_dict()
            row_data["level"] = "document"
            row_data["alignment_status"] = "NO_SENTENCE_DATA"  # for stats

            log["status"] = "NO_SENTENCE_DATA"
            return [row_data], log

        sent_rows = guide_by_id[oare_id]

        # Checking for duplicates
        duplicate_mask = sent_rows["first_word_number"].duplicated(keep="first")
        if duplicate_mask.any():
            # If there are any duplicate word indexes, keep only the first row
            sent_rows = sent_rows.loc[~duplicate_mask].reset_index(drop=True)
        log["n_oare_rows"] = len(sent_rows)

        akk_ok, fail_info = self.verify_by_akkadian_word(train_row, sent_rows)
        log["akk_ok"] = akk_ok

        # point 2: akk mismatch
        # if not self.verify_by_akkadian_word(train_row, sent_rows):
        if not akk_ok:
            row_data = train_row.to_dict()
            row_data["level"] = "document"
            row_data["alignment_status"] = "FAIL_AKKADIAN_MISMATCH"  # for stats
            log["status"] = "FAIL_AKKADIAN_MISMATCH"
            log["akk_fail_info"] = fail_info
            return [row_data], log

        sliced_sources = self.slice_transliteration(
            train_row["transliteration"], sent_rows
        )
        split_sentences = self.split_target_nuclear(train_row["translation"])

        log["n_split_eng"] = len(split_sentences)
        log["split_preview"] = split_sentences[:5]
        log["slice_preview"] = sliced_sources[:5]

        if len(split_sentences) == len(sent_rows):
            # point 3: perdect metch
            status = "SUCCESS"
            final_targets = split_sentences
        else:
            # point 4: possible fallback to csv eng
            if accept_worst_eng:
                # use CSV eng
                status = "SUCCESS_CSV_FALLBACK"
                final_targets = sent_rows["translation"].tolist()
            else:
                row_data = train_row.to_dict()
                row_data["level"] = "document"
                row_data["alignment_status"] = "ENG_MISMATCH"  # for stats
                log["status"] = "ENG_MISMATCH"
                return [row_data], log

        aligned_data = []
        for i in range(len(sent_rows)):
            aligned_data.append(
                {
                    "oare_id": oare_id,
                    "level": "sentence",
                    "sentence_idx": i,
                    "transliteration": sliced_sources[i],
                    "translation": final_targets[i],
                    "alignment_status": status,
                }
            )

        log["status"] = status
        return aligned_data, log

    def align_data(self, train_df: pd.DataFrame, doc_limit=512) -> pd.DataFrame:
        if self.verbose:
            print(f"Starting alignment for {len(train_df)} tablets...")

        # lookup for ids
        id_dict = {
            k: g.sort_values("first_word_number").reset_index(drop=True)
            for k, g in self.alignment_guide.groupby("text_uuid")
        }

        processed_rows = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Aligning"):
            aligned_segments, log = self.compare_and_align(
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

        if self.verbose:
            print(
                f"Alignment complete. Extracted {len(aligned_sentences)} sentences and {len(short_documents)} short docs."
            )

        assert isinstance(final_df, pd.DataFrame)
        return final_df


# example usage
def main():
    aligner = Aligner()
    print("Loading datasets...")
    train_df = pd.read_csv(f"dataset/train.csv")
    sentences_oare = pd.read_csv(f"dataset/Sentences_Oare_FirstWord_LinNum.csv")

    print("Indexing alignment aid...")
    guide_by_id = {
        k: g.sort_values("first_word_number").reset_index(drop=True)
        for k, g in sentences_oare.groupby("text_uuid")
    }

    print("Starting alignment process...")
    all_rows = []
    log_rows = []

    for _, row in train_df.iterrows():
        processed_rows, log = aligner.compare_and_align(row, guide_by_id)
        all_rows.extend(processed_rows)
        log_rows.append(log)

    aligned_df = pd.DataFrame(all_rows)
    log_df = pd.DataFrame(log_rows)

    # Statistics
    print("\n" + "=" * 30)
    print("FINAL ALIGNMENT SUMMARY")
    print("=" * 30)
    print(aligned_df["alignment_status"].value_counts())

    sent_count = len(aligned_df[aligned_df["level"] == "sentence"])
    doc_count = len(aligned_df[aligned_df["level"] == "document"])

    print(f"\nFinal Sentence-Level Rows: {sent_count}")
    print(f"Final Document-Level Rows: {doc_count}")
    print(f"Total Combined Training Rows: {len(aligned_df)}")

    print(log_df[log_df["status"] == "ENG_MISMATCH"].head(10))

    return aligned_df, log_df


if __name__ == "__main__":
    final_dataset, log_dataset = main()
    # Save the final dataset to CSV for further analysis
    # final_dataset.to_csv('dataset/aligned_dataset.csv', index=False)
    # log_dataset.to_csv("dataset/log_dataset.csv", index=False)
    print("Dataset saved to dataset/aligned_dataset.csv")

    # (Luca:) I have the datasets in the following folders (for testing):

    # Datasets without duplicate starting indexes

    # final_dataset.to_csv('dataset/alignment_datasets/aligned_dataset_1_no_dup.csv', index=False)
    # log_dataset.to_csv("dataset/alignment_datasets/log_dataset_1_no_dup.csv", index=False)

    # Datasets with updated pattern (unicode handling and ." splitting)

    # final_dataset.to_csv('dataset/alignment_datasets/aligned_dataset_2_updated_pattern.csv', index=False)
    # log_dataset.to_csv("dataset/alignment_datasets/log_dataset_2_updated_pattern.csv", index=False)
