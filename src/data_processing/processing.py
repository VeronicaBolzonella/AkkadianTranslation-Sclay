"""
Public API:
    TextProcessor

All other classes/constants are internal implementation details.
Do not use them directly.


You are expected create a class `TextProcessor` and use its `preprocessing()` and `postprocessing()` methods

See the `main()` for an example.

Some code (especially some regex) from:
    Akkadian to English Translation Inference
    OPTIMIZED VERSION - Multi-Layer Optimization Applied
    https://www.kaggle.com/code/sword4949/deep-past-challenge-ver2?scriptVersionId=294896184&cellId=1

"""

import os
import re
import pandas as pd
import torch
import wandb

from .patterns import CompiledPatterns

BASE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(BASE_DIR, "..", "..", "dataset")

PREPROCESSING_CONFIG = {
    "test_data_path": os.path.join(BASE_DIR, "test.csv"),
    "train_data_path": os.path.join(BASE_DIR, "train.csv"),
    "computation_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class TextProcessor:
    """Provide pre and post processing functions"""

    def __init__(self):
        self.patterns = CompiledPatterns()

    def _handle_determinatives(self, match):
        """replace or leave intact determinatives and log to wandb if missing"""
        det = match.group(0)

        replacement = self.patterns.DETERMINATIVE_MAP.get(det, None)

        if replacement:
            # Surround with spaces so it doesn't stick to the word
            return f" {replacement} "

        return det
    
    def _underscore_capitalize(self, match):
        content = match.group(1)
        # Check if it ends with "-ta" (case insensitive check optional, 
        # but based on your requirement we keep "-ta" lowercase)
        if content.lower().endswith("-ta"):
            # Capitalize everything except the last 3 characters ("-ta")
            return content[:-3].upper() + "-ta"
        else:
            return content.upper()
    
    def _ascii_to_diacritics(self, text: str) -> str:
        # Step 1: unicode subscripts → regular digits
        text = text.translate(self.patterns.SUBSCRIPT_TABLE)
        # Step 2: ASCII consonants → diacritics
        text = text.replace("sz", "š").replace("SZ", "Š")
        text = text.replace("s,", "ṣ").replace("S,", "Ṣ")
        text = text.replace("t,", "ṭ").replace("T,", "Ṭ")
        # Step 3: numbered vowels → diacritic vowels
        text = self.patterns.V2.sub(
            lambda m: m.group(1).translate(self.patterns.ACUTE), text
        )
        text = self.patterns.V3.sub(
            lambda m: m.group(1).translate(self.patterns.GRAVE), text
        )
        return text

    def preprocess_transliteration_text(
        self,
        transliteration_text,
        separate_compounds: bool = False,
        with_hyphens: bool = False,
        named_determinatives: bool = False,
        normalize_chars: bool = False,
        diacritic_mode: bool = False,
    ):
        """Preprocessess transliteration text

        Args:
            transliteration_text: text to be preprocessed
            separate_compounds (bool, optional): if true separates logograms compounds into two space-separated words.
                Defaults to False.
            with_hyphens (bool, optional): if true maintains hypheneted words, else removes hyphens and joins syllabi
                (e.g. a-na -> ana). Defaults to False.
            named_determinatives (bool, optional): if true substitute determiantives with meaningful word,
                else leave intact. Defaults to False.

        Returns:
            text: preprocessed text

        Notes:
            preprocessing steps:
            - changes subscripts, superscripts and accents to unifified letters (e.g. "á" to "a2")
            - Add special token to indicate start (<lgg_s>) and end (<lgg_e>) of logograms
            - Lowercase everything
            - change remaining subscripts to normal numbers
            - change gaps to <gap> and <big_gap>
            - handle determinatives
            - optionally remove hyphens
            - optionally separate logogram compounds with spaces
            - remove forbidden characters
            - remove multispaces
        """
        if pd.isna(transliteration_text):
            return ""

        text = str(transliteration_text)
        text = text.translate(self.patterns.FORBIDDEN_CHARS_INPUT)

        text = self.patterns.KUB_PATTERN.sub("KÙ.BABBAR", text)
        text = self.patterns.MODIFIERS.sub("", text)
        text = self.patterns.ROUND_BRACKETS.sub(r"\1", text)
        # 2. Square -> just remove symbols: [asu-ta] -> asu-ta
        text = self.patterns.SQUARE_BRACKETS.sub(r"\1", text)
        text = self.patterns.UNDERSCORE_LOGOGRAMS.sub(self._underscore_capitalize, text)
        
        # normalize gaps
        # text = self.patterns.BIG_GAP_INPUT.sub("<big_gap>", text)
        # BEFORE logograms or problem with PN
        text = self.patterns.GAPS.sub("<gap>", text)
        text = self.patterns.MERGE_GAPS.sub(" <gap> ", text)
        text = self.patterns.MERGE_BIG_GAPS.sub(" <gap> ", text)
        text = self.patterns.MERGE_MIXED_GAPS.sub(" <gap> ", text)

        # OPTIONAL: separate logograms compound with space
        if separate_compounds:
            text = self.patterns.LOGOGRAMS_COMPOUND.sub(r"\1 \2", text)
        # add special tokens for logograms start and end, and lowecase
        text = self.patterns.LOGOGRAM.sub(lambda m: f"<lgg_s>{m.group(1)}<lgg_e>", text)

        # OPTIONAL: sub determinatives with word
        if named_determinatives:
            # replace if found, leave and log if not found
            text = self.patterns.DETERMINATIVE_PATTERN.sub(
                self._handle_determinatives, text
            )

        assert not (diacritic_mode and normalize_chars)
        # normalize characters
        # normalize subscripts
        if normalize_chars:
            text = text.translate(self.patterns.SUBSCRIPT_TABLE)
            for acc, num in self.patterns.ACCENT_MAP.items():
                text = text.replace(acc, num)
        elif diacritic_mode:
            text = self._ascii_to_diacritics(text)

        if not named_determinatives:
            text = self.patterns.DET_UPPER_RE.sub(r"\1", text)  
            text = self.patterns.DET_LOWER_RE.sub(r"{\1}", text)  

        text = text.lower()

        # OPTIONAL: remove hyphens
        if not with_hyphens:
            text = self.patterns.HYPHENS.sub("", text)

        text = self.patterns.MULTI_SPACE.sub(" ", text).strip()

        return text

    def preprocess_translation_text(self, translation_text):
        if pd.isna(translation_text):
            return ""
        text = str(translation_text)

        text = text.replace("(?)", "")
        text = text.replace("(!)", "")

        # handle gaps
        text = self.patterns.GAPS.sub("<gap>", text)
        text = self.patterns.MERGE_GAPS.sub(" <gap> ", text)
        text = self.patterns.MERGE_BIG_GAPS.sub(" <gap> ", text)
        text = self.patterns.MERGE_MIXED_GAPS.sub(" <gap> ", text)
        
        # remove §
        text = text.replace("§", "")

        # handle number things
        for pattern, replacement in self.patterns.MONEY_PATTERNS:
            text = pattern.sub(replacement, text)

        text = self.patterns.MONTH_NUMERAL_REGEX.sub(
            lambda m: f"{m.group(1)} {self.patterns.ROMAN_MAP.get(m.group(2).upper(), m.group(2))}",
            text,
        )

        for symbol, pattern in self.patterns.FRACTION_CONVERSION.items():
            text = pattern.sub(f" {symbol}", text)

        text = re.sub(self.patterns.ONE_CLEAN_REGEX, "1", text, flags=re.I)
        text = self.patterns.OTHER_NUMBERS_PATTER.sub(
            lambda m: self.patterns.NUMBERS_MAP[m.group(0).lower()], text
        )

        # cleanups and abbreviations
        for pattern, replacement in self.patterns.WORD_EXPANSIONS:
            text = pattern.sub(replacement, text)

        text = self.patterns.ENG_SLASH_CHOICE.sub(r"\1", text)
        text = self.patterns.ENG_STRAY_CLEANUP.sub("", text)
        
        # fix quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')  
        text = text.replace('\u2018', "'").replace('\u2019', "'")

        # fix punctuation spacing
        text = self.patterns.SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = self.patterns.MULTI_SPACE.sub(" ", text).strip()

        return text


    def postprocess_translation_output(self, translation_text):
        """Postprocess translation text

        Args:
            translation_text: translated text

        Returns:
            text: processed tranlation text

        Notes:
            postprocessing steps:
            - clean and merge gaps
            - remove repeated words and n-grams (common hallucination)
            - remove extra spaces and punctuations
            - change all text numbers to digits (e.g. one to 1)
            - change all fractions to decimals (e.g. one third to 0.33333)
        """
        if not isinstance(translation_text, str) or not translation_text.strip():
            return ""

        text = str(translation_text)

        # substitute gaps and merge multiple gaps
        text = self.patterns.GAPS.sub("<gap>", text)
        text = self.patterns.MERGE_GAPS.sub(" <gap> ", text)
        text = self.patterns.MERGE_BIG_GAPS.sub(" <gap> ", text)
        text = self.patterns.MERGE_MIXED_GAPS.sub(" <gap> ", text)

        # remove repeated words and ngrams (up to 4)
        text = self.patterns.REPEATED_WORDS.sub(r"\1", text)
        for ngram_pattern in self.patterns.NGRAM_PATTERNS:
            text = ngram_pattern.sub(r"\1", text)

        # remove extra spaces
        text = self.patterns.SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = self.patterns.REPEATED_PUNCT.sub(r"\1", text)
        text = self.patterns.MULTI_SPACE.sub(" ", text).strip().strip("-").strip()

        text = self.patterns.MONTH_NUMERAL_REGEX.sub(
            lambda m: f"{m.group(1)} {self.patterns.ROMAN_MAP.get(m.group(2).upper(), m.group(2))}",
            text,
        )

        # change numbers to digints (one -> 1) from 1 to 10
        text = re.sub(self.patterns.ONE_CLEAN_REGEX, "1", text, flags=re.I)

        # Process all other numbers (two through hundred)
        text = self.patterns.OTHER_NUMBERS_PATTER.sub(
            lambda m: self.patterns.NUMBERS_MAP[m.group(0).lower()], text
        )

        # change fractions to decimals
        for symbol, pattern in self.patterns.FRACTION_CONVERSION.items():
            text = pattern.sub(symbol, text)

        return text

