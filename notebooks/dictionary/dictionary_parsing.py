"""
parse_dictionary.py
-------------------
Parse the eBL Akkadian dictionary CSV into training datapoints.

Each word with multiple numbered definitions produces one datapoint per sense.
Only entries that contain a quoted translation gloss are kept.

Usage:
    from parse_dictionary import parse_file
    datapoints = parse_file("eBL_Dictionary.csv")

Each datapoint dict:
    transliteration  - the lemma, e.g. "abattu"
    homograph        - Roman numeral index, e.g. "I" (or None)
    sense            - numbered sense (int) or None if entry has no sub-senses
    translation      - primary quoted gloss, e.g. "river-gravel"
    all_translations - all quoted glosses found in this sense
    definition       - full definition text for this sense
"""

import csv
import re


def _extract_translations(text):
    """Return all quoted glosses from a definition string."""
    return re.findall(r'"([^"]+)"', text)


def _split_definitions(raw):
    """
    Split a definition into numbered senses.
    Handles run-together numbers like 'plant3.' → 'plant 3.'
    Returns list of (sense_number_or_None, text).
    """
    text = re.sub(r"(?<=[^\s\d])(\d+)\.", r" \1.", raw)
    parts = re.split(r"(?<!\w)(\d+)\.\s*", text)

    if len(parts) == 1:
        return [(None, raw.strip())]

    preamble = parts[0].strip()
    results = []
    i = 1
    while i < len(parts) - 1:
        sense_text = parts[i + 1].strip()
        full = f"{preamble} {sense_text}".strip() if preamble else sense_text
        results.append((int(parts[i]), full))
        i += 2
    return results


def _parse_headword(raw):
    """Split 'abattu I' → ('abattu', 'I')."""
    m = re.match(r"^(.+?)\s+(I{1,3}V?|VI{0,3}|IX|X{1,3})$", raw.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (raw.strip(), None)


def parse_file(path):
    """
    Parse a dictionary CSV file.

    Returns a list of datapoint dicts, one per sense that has a translation.
    Words with N numbered senses produce up to N datapoints.
    """
    datapoints = []

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row

        for row in reader:
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue

            lemma, homograph = _parse_headword(row[0])

            for sense_num, sense_text in _split_definitions(row[1]):
                translations = _extract_translations(sense_text)
                if not translations:
                    continue  # no gloss found — skip this sense

                datapoints.append(
                    {
                        "transliteration": lemma,
                        "homograph": homograph,
                        "sense": sense_num,  # int or None
                        "translation": translations[0],  # primary gloss
                        "all_translations": translations,  # all glosses in sense
                        "definition": sense_text,
                    }
                )

    return datapoints


"""
expand_translations.py
----------------------
Run after parse_dictionary.py.
Splits datapoints where the translation contains multiple meanings
separated by semicolons (always) or commas (optional).
Deduplicates entries with the same transliteration + homograph + translation.

Usage:
    from expand_translations import expand

    expanded = expand(datapoints)                        # semicolons only
    expanded = expand(datapoints, split_commas=True)     # semicolons + commas
    expanded = expand(datapoints, split_commas=True, min_length=3)
"""


def expand(datapoints, split_commas=False, min_length=3):
    """
    Parameters
    ----------
    datapoints   : list of dicts from parse_file()
    split_commas : also split on commas (default False)
    min_length   : minimum character length to keep a fragment after splitting
                   only applied when there are multiple parts — never drops
                   a translation that is the only part (default 3)
    """
    seen = set()
    result = []

    for dp in datapoints:
        parts = [p.strip() for p in dp["translation"].split(";") if p.strip()]

        if split_commas:
            comma_parts = []
            for part in parts:
                comma_parts.extend(p.strip() for p in part.split(",") if p.strip())
            parts = comma_parts

        # Only filter short fragments when splitting produced multiple parts.
        # Never drop the translation entirely if it's the only thing there.
        if len(parts) > 1:
            parts = [p for p in parts if len(p) >= min_length]

        if not parts:
            parts = [dp["translation"]]  # fallback: keep original untouched

        candidates = (
            [{**dp, "all_translations": parts}]
            if len(parts) == 1
            else [
                {**dp, "translation": part, "all_translations": [part]}
                for part in parts
            ]
        )

        for candidate in candidates:
            key = (
                candidate["transliteration"],
                candidate["homograph"],
                candidate["translation"],
            )
            if key not in seen:
                seen.add(key)
                result.append(candidate)

    return result
