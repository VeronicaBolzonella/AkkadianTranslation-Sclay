from src.config import (
    EXTERNAL_DATASET_INPUTS,
    INTERNAL_DATASET_INPUTS,
    Columns,
    ALIGNMENT_PATH,
)
from src.data_processing.processing import TextProcessor
from src.data_processing.alignment import Aligner
from src.utils import stack_csvs_from_folder, drop_empty_rows
import pandas as pd
import io
import csv
import time
import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def load_data() -> pd.DataFrame:
    df_internal = stack_csvs_from_folder(INTERNAL_DATASET_INPUTS)
    return drop_empty_rows(df_internal)


def align(row, guide_by_id):
    oare_id = row["id"]

    if oare_id not in guide_by_id:
        doc_row = row.to_dict()
        doc_row["level"] = "document"
        doc_row["alignment_status"] = "NO_GUIDE"

        return [doc_row]
    sent_rows = guide_by_id[oare_id]

    # Pre-calculate word count to handle the "last interval" logic
    words_trans = row["transliteration"].split()
    max_words = len(words_trans)

    # Get the (start, end) points for slicing
    intervals = slice_transliteration(row["transliteration"], sent_rows)

    final_segments = []

    for i, (_, guide_row) in enumerate(sent_rows.iterrows()):
        start_idx, end_idx = intervals[i]

        # Ensure the end_idx defaults to the maximum if it was the last segment
        if end_idx is None or end_idx > max_words:
            end_idx = max_words

        segment = row.to_dict().copy()
        segment["level"] = "sentence"
        segment["sentence_idx"] = i

        # The data you'll pass to the LLM
        segment["original_transliteration"] = row["transliteration"]
        segment["slice_interval"] = (start_idx, end_idx)
        segment["guide_translation"] = guide_row["translation"]
        segment["translation"] = row["translation"]

        # Add a visual check for your own debugging
        segment["text_at_interval"] = (
            " ".join(words_trans[start_idx:end_idx]) if start_idx is not None else ""
        )

        segment["alignment_status"] = "READY_FOR_LLM"
        final_segments.append(segment)

    return final_segments


def slice_transliteration(source_text, alignment_rows):
    words = source_text.split()
    total_len = len(words)
    intervals = []

    for i in range(len(alignment_rows)):
        curr_row = alignment_rows.iloc[i]

        # Start point
        if pd.isna(curr_row["first_word_number"]):
            start = None
        else:
            start = int(curr_row["first_word_number"]) - 1

        # End point logic: Look at the next row's start, or use total length
        if i + 1 < len(alignment_rows):
            next_row = alignment_rows.iloc[i + 1]
            if not pd.isna(next_row["first_word_number"]):
                end = int(next_row["first_word_number"]) - 1
            else:
                end = total_len
        else:
            # This is the last interval
            end = total_len

        intervals.append((start, end))

    return intervals


sentences_oare = pd.read_csv(ALIGNMENT_PATH)
sentences_oare = sentences_oare.dropna(subset=["translation"])
train_df = load_data()

guide_by_id = {
    k: g.sort_values("first_word_number").reset_index(drop=True)
    for k, g in sentences_oare.groupby("text_uuid")
}

all_rows = []

for _, row in train_df.iterrows():
    processed_rows = align(row, guide_by_id)
    all_rows.extend(processed_rows)

final_df = pd.DataFrame(all_rows)


final_df.head()


# To see the distribution of all levels (document vs sentence)
print(final_df["level"].value_counts())


def generate_alignment_prompt(oare_id, final_df, train_df):
    # Get the high-quality full translation from the original training data
    full_train_row = train_df[train_df["id"] == oare_id].iloc[0]
    full_train_text = full_train_row["translation"]
    orig_trans = full_train_row["transliteration"]

    # Get the segmented guide rows
    segments = final_df[final_df["id"] == oare_id].sort_values("sentence_idx")

    segment_block = ""
    for _, row in segments.iterrows():
        segment_block += (
            f"Segment {row['sentence_idx']}:\n"
            f"- Suggested Interval: {row['slice_interval']}\n"
            f"- Guide Translation: {row['guide_translation']}\n\n"
        )

    prompt = f"""
### ALIGNMENT TASK: {oare_id}

**Full Original Transliteration:**
{orig_trans}

**Full High-Quality Training Translation:**
{full_train_text}

**Proposed Segments to Align:**
{segment_block}

**Instructions:**
Your instructions are the following:
- Align the transliteration to the high-quality translation using the guide translations as a reference for sentence breaks.
- The interval refers to the places where the sentence should break in the original transliteration.
- The intervals might have errors and be slightly misplaced.
- You must return your output in a csv the following format: oare_id,aligned_transliteration,aligned_translation,comment.
- Include the header row: oare_id,aligned_transliteration,aligned_translation,comment

where each row corresponds to an aligned transliteration and translation
"""
    return prompt


SYSTEM_PROMPT = "You are an expert in Old Assyrian cuneiform texts. You follow instructions properly"


def process_all(
    oare_ids, final_df, train_df, output_path: str = "alignments_master.csv"
):
    # --- Phase 1: Submit ---
    requests = [
        {
            "custom_id": oare_id,
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": generate_alignment_prompt(
                            oare_id, final_df, train_df
                        ),
                    }
                ],
            },
        }
        for oare_id in oare_ids
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}")
    with open("batch_id.txt", "w") as f:
        f.write(batch.id)

    # --- Phase 2: Wait ---
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        print(f"Status: {batch.processing_status} | {batch.request_counts}")
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    results = list(client.messages.batches.results(batch.id))
    for entry in results[:3]:
        print(f"--- {entry.custom_id} ---")
        print(f"stop_reason: {entry.result.message.stop_reason}")
        print(f"content blocks: {entry.result.message.content}")
        print()

    # --- Phase 3: Save ---
    fieldnames = [
        "oare_id",
        "aligned_transliteration",
        "aligned_translation",
        "comment",
    ]
    failed = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    results = list(
        client.messages.batches.results(batch.id)
    )  # materialise so we know total
    for i, entry in enumerate(results):
        oare_id = entry.custom_id
        print(f"Saving {i + 1}/{len(results)}: {oare_id}")  # ← progress line

        if entry.result.type != "succeeded":
            print(f"  FAILED: {oare_id} — {entry.result.type}")
            failed.append(oare_id)
            continue

        if entry.result.message.stop_reason == "max_tokens":
            print(f"WARNING: truncated for {oare_id}")
            failed.append(oare_id)
            continue

        raw_csv = entry.result.message.content[0].text.strip()
        if raw_csv.startswith("```"):
            raw_csv = "\n".join(raw_csv.split("\n")[1:])
        if raw_csv.endswith("```"):
            raw_csv = raw_csv.rsplit("```", 1)[0]
        raw_csv = raw_csv.strip()

        rows = list(csv.DictReader(io.StringIO(raw_csv)))
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerows(rows)

    if failed:
        print(f"\n{len(failed)} failed: {failed}")
    print(f"Done. Results in {output_path}")


oare_ids = final_df[final_df["level"] == "sentence"]["id"].unique().tolist()

oare_ids = oare_ids.head(1)

process_all(oare_ids, final_df, train_df)
