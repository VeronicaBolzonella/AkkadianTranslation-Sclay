import pandas as pd
from pathlib import Path


def stack_csvs_from_folder(folder_path) -> pd.DataFrame:
    """
    Read all CSV files in a folder with columns:
    id, transliteration, translation
    and stack them into a single DataFrame.

    Args:
        folder_path (str or Path): Path to folder containing CSV files.

    Raises:
        FileNotFoundError: [TODO:description]
        ValueError: [TODO:description]
        ValueError: [TODO:description]
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {folder}")

    required_cols = {"id", "transliteration", "translation"}
    dfs = []

    for file in csv_files:
        df = pd.read_csv(
            file, keep_default_na=False
        )  # The second param prevents the akkadian"NA" from being reaad as Null

        # check column format
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"{file} must contain columns: {required_cols}. "
                f"Found: {set(df.columns)}"
            )

        df = df[["id", "transliteration", "translation"]]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def drop_empty_rows(df) -> pd.DataFrame:
    # Find rows with any empty/NaN cell
    empty_mask = df.isnull().any(axis=1) | (df == "").any(axis=1)

    # Print the IDs of rows being dropped
    dropped_ids = df.loc[empty_mask, "id"].tolist()
    if dropped_ids:
        print(f"Dropping {len(dropped_ids)} rows with empty cells:")
        for id_ in dropped_ids:
            print(f"  id: {id_}")
    else:
        print("No empty rows found.")

    # Drop the rows and reset index
    clean_df = df[~empty_mask].reset_index(drop=True)
    print(f"Rows before: {len(df)}, Rows after: {len(clean_df)}")

    return clean_df
