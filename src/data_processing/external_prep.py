"""
This file takes an external dataset and puts it in the same format as the datasets provided by the Kaggle competition

You are intended to modify the main() for your purposes.
"""

from src.config import PROJECT_ROOT
from pathlib import Path
import shutil


def akkademia(translation_file, transliteration_file, output_name, id_suffix="extra"):
    """Processes the Akkademia datasets

    Join two text files line-by-line into a CSV with columns:
    id, transliteration, translation.

    Args:
        trans_file (str): Path to transliteration file.
        transl_file (str): Path to translation file.
        output_file (str): Path to output CSV file.
        id_suffix (str): Text appended to sequential ID (default: "extra").
    Raises:
        FileNotFoundError: If one of the files does not exist.
        ValueError: If the files do not have the same number of lines.
    """

    # Read both files first to check lengths
    with open(translation_file, encoding="utf-8") as f1:
        trans_lines = f1.readlines()

    with open(transliteration_file, encoding="utf-8") as f2:
        transl_lines = f2.readlines()

    # Check same number of lines
    if len(trans_lines) != len(transl_lines):
        raise ValueError(
            f"Files must have the same number of lines. "
            f"{translation_file}: {len(trans_lines)} lines, "
            f"{transliteration_file}: {len(transl_lines)} lines."
        )

    # Write output
    with open(output_name, "w", encoding="utf-8") as out:
        out.write("id,translation,transliteration\n")

        for i, (t1, t2) in enumerate(zip(trans_lines, transl_lines), start=1):
            t1 = t1.strip().replace('"', '""')
            t2 = t2.strip().replace('"', '""')
            out.write(f'{i}{id_suffix},"{t1}","{t2}"\n')


def process_all_akkademia(result_name):
    path_dataset = Path(PROJECT_ROOT) / "dataset/external"

    splits = ["train", "valid", "test"]
    generated_files = []

    for split in splits:
        output_name = path_dataset / f"{split}.csv"
        transliteration = path_dataset / f"{split}.tr"
        translation = path_dataset / f"{split}.en"

        akkademia(
            transliteration_file=str(transliteration),
            translation_file=str(translation),
            output_name=str(output_name),
            id_suffix=f"{split}_akkademia",
        )

        generated_files.append(output_name)

    # merge into one CSV
    merged_output = path_dataset / result_name

    with open(merged_output, "w", encoding="utf-8") as outfile:
        first_file = True

        for file in generated_files:
            with open(file, encoding="utf-8") as infile:
                if first_file:
                    outfile.write(infile.read())  # keep header
                    first_file = False
                else:
                    next(infile)  # skip header
                    outfile.write(infile.read())

    print(f"Created merged dataset: {merged_output}")


def main():
    process_all_akkademia(result_name="combined_akkademia")


if __name__ == "__main__":
    main()
