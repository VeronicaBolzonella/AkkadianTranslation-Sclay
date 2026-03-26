import random
import re
import pandas as pd
import nltk
from nltk.corpus import words
from typing import Any

"""
Data Augmentation must happen after alligning so only one sentence is processed (and names are not confused with start 
of sentence), and before preprocessing in order to preserve original characters and capitalization.
"""

class DataAugmentation:
    def __init__(
        self,
        lexicon_csv_path: str = "",
        morphemes_csv_path: str = "",
        onomasticon_csv_path: str = "",
        verbose=False,
    ):
        self.lexicon_csv_path = lexicon_csv_path
        self.morphemes_csv_path = morphemes_csv_path
        self.onomasticon_csv_path = onomasticon_csv_path
        self.verbose = verbose
        self.gaps = ["<gap>", "<big_gap>"]
        
        self.gn_bank = {
            "ku-bu-ur-na-at": "Kuburnat",
            "du-ur-ḫu-mì-it": "Durhumit",
            "za-al-pá-a": "Zalpe",
            "pu-ru-uš-ḫa-dim": "Purušhaddum",
            "dur₄-ḫu-mì-it": "Durhumit",
            "e-lu-ḫu-ut": "Eluhut",
            "kà-ni-iš": "Kanesh",
            "bu-ru-uš-ḫa-tum": "Burušhattum",
            "ḫu-ra-ma": "Hurama",
            "wa-aḫ-šu-ša-na": "Wahšušana",
            "lu-ḫu-za-tí-a": "Luhuzattiya",
        }
        self.akk_gn_list = list(self.gn_bank.keys())

        # norm forms from lexicon
        self.pn_form_to_norm = {} 
        self.pn_set = set()      
        self.gn_form_to_norm = {}
        self.gn_set = set()

        # exact pns from onomasticon
        self.pn_akk_to_eng = {}   
        self.pn_akk_set = set()   
        self.onomasticon_df = None

        self.affixes = set()
        self.suffixes = set()

        if lexicon_csv_path != "":
            df = pd.read_csv(lexicon_csv_path)
            pn_df = df[df["type"] == "PN"].copy()
            for _, row in pn_df.iterrows():
                form = str(row["form"]).lower()
                norm = str(row["norm"]) if pd.notna(row["norm"]) else form
                self.pn_form_to_norm[form] = norm
                self.pn_set.add(form)

            gn_df = df[df["type"] == "GN"].copy()
            for _, row in gn_df.iterrows():
                form = str(row["form"]).lower()
                norm = str(row["norm"]) if pd.notna(row["norm"]) else form
                self.gn_form_to_norm[form] = norm
                self.gn_set.add(form)

        if onomasticon_csv_path != "":
            self.onomasticon_df = pd.read_csv(onomasticon_csv_path)
            
            for _, row in self.onomasticon_df.iterrows():
                akk = str(row["transliteration"])
                eng = str(row["translation"])
                self.pn_akk_to_eng[akk.lower()] = eng
                self.pn_akk_set.add(akk.lower())
            # cleaned version for sampling
            self.onomasticon_df = self.onomasticon_df

        if morphemes_csv_path != "":
            morph_df = pd.read_csv(morphemes_csv_path)
            for _, row in morph_df.iterrows():
                word = str(row["word"]).lower()
                morph_type = str(row["type"]).lower() if pd.notna(row["type"]) else ""
                word_clean = re.sub(r"\s+[ivxlcdm]+$", "", word, flags=re.IGNORECASE).strip()
                if morph_type == "aff":
                    self.affixes.add(word_clean)
                elif morph_type == "suff":
                    self.suffixes.add(word_clean)

        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words")
        self.english_vocab = set(w.lower() for w in words.words())

    def _get_noise_indices(self, tokens, noise_percent: float = 0.1):
        # returns 10% of indices to be corrupted
        n_to_corrupt = max(1, int(len(tokens) * noise_percent))
        return random.sample(range(len(tokens)), n_to_corrupt)

    def add_token_noise(self, text: str, tokens_type: str = "word_tokens", noise_percent: float = 0.2):
        """Adds noise to a given percentage of tokens, of which:
        - with 20% prob token is switched with next token
        - with 20% prob token is removed
        - with 80% prob token is remain intact

        Args:
            text (str): string to corrups
            tokens_type (str, optional): Level of corruption (word or char). Defaults to "word_tokens".
            noise_percent (float, optional): percentage of string to be corrupted. Defaults to 0.2.

        Raises:
            ValueError: noise level not implemented (token or char)

        Returns:
            str: corrupted text
        """
        if tokens_type == "word_tokens":
            tokens = text.split()
        elif tokens_type == "char_tokens":
            tokens = list(text)
        else:
            raise ValueError("Unsupported token type")
        indices = self._get_noise_indices(tokens, noise_percent=noise_percent)
        for idx in indices:
            strategy = random.uniform(0, 1)
            if strategy < 0.2:
                # swap token
                if idx < len(tokens) - 1:
                    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
            elif strategy < 0.4:
                # remove token
                tokens[idx] = ""
        return " ".join([t for t in tokens if t != ""])

    def add_gap_noise(self, text: str, tokens_type: str = "word_tokens", noise_percent: float = 0.1):
        """For a given percentage of tokens either removes (10%) or substitutes with a gap (10%) 

        Args:
            text (str): text to corrupt
            tokens_type (str, optional): Level of corruption (word or char). Defaults to "word_tokens".
            noise_percent (float, optional): percentage of string to be corrupted. Defaults to 0.2.

        Raises:
            ValueError: noise level not implemented (token or char)

        Returns:
            str: corrupted text
        """
        if tokens_type == "word_tokens":
            tokens = text.split()
        elif tokens_type == "char_tokens":
            tokens = list(text)
        else:
            raise ValueError("Unsupported token type")
        if len(tokens) < 2: return text # intact
        indices = self._get_noise_indices(tokens, noise_percent=noise_percent)
        for idx in indices:
            strategy = random.uniform(0, 1)
            if strategy < 0.8: tokens[idx] = random.choice(self.gaps) # sub with gap
            elif strategy < 0.9: tokens[idx] = "" # remove
        return " ".join([t for t in tokens if t != ""]) if tokens_type == "word_tokens" else "".join([t for t in tokens if t != ""])

    def name_swap_augmentation(self, df, swap_pn=True, swap_gn=True):
        """Swaps names of people or places with random new names, creates a df with the new datapoints with 
        swapped names

        Args:
            df (DataFrame): original dataframe
            swap_pn (bool, optional): If true swaps people names. Defaults to True.
            swap_gn (bool, optional): If true swaps geographical names. Defaults to True.

        Returns:
            df: augmented df with new swapped rows
        """
        if self.verbose:
            print(f"Starting augmentation on {len(df)} rows...")
        augmented_rows = []

        for idx, row in df.iterrows():
            original_id = row.get("id", f"row_{idx}")
            translit = str(row["transliteration"])
            transl = str(row["translation"])

            if swap_pn:
                # swap people names
                success, new_akk, new_eng = self.swap_pn(translit, transl)
                if success:
                    new_row = row.to_dict()
                    new_row["id"] = f"{original_id}-pn_swap"
                    new_row["transliteration"] = new_akk
                    new_row["translation"] = new_eng
                    augmented_rows.append(new_row)

            if swap_gn:
                # swapped geo names
                success, new_akk, new_eng = self.swap_gn(translit, transl)
                if success:
                    new_row = row.to_dict()
                    new_row["id"] = f"{original_id}-gn_swap"
                    new_row["transliteration"] = new_akk
                    new_row["translation"] = new_eng
                    augmented_rows.append(new_row)

        if not augmented_rows:
            return df
        return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    def _find_english_name(self, eng_norm, translation):
        pattern = r"\b" + re.escape(eng_norm) + r"\b"
        return re.search(pattern, translation, flags=re.IGNORECASE) is not None

    def _find_akk_name_with_morpheme(self, token, name_set):
        # search for akk name after removing prefix and suffix, returns (pre-, name, suf-)
        token_lower = token.lower()
        if token_lower in name_set:
            return token, "", ""
        for affix in self.affixes:
            if token_lower.startswith(affix):
                remainder = token_lower[len(affix) :]
                if remainder in name_set:
                    return token[len(affix):], token[:len(affix)], ""
                for suffix in self.suffixes:
                    if remainder.endswith(suffix):
                        potential_name = remainder[: -len(suffix)]
                        if potential_name in name_set:
                            return token[len(affix):-len(suffix)], token[:len(affix)], token[-len(suffix):]
        for suffix in self.suffixes:
            if token_lower.endswith(suffix):
                potential_name = token_lower[: -len(suffix)]
                if potential_name in name_set:
                    return token[:-len(suffix)], "", token[-len(suffix):]
        return None, None, None

    # replaces with a random name from Onomasticon.
    def swap_pn(self, transliteration: str, translation: str):
        """Identifies and swaps personal names. 
        Tries Onomasticon method first; if success is false, tries Lexicon method.
        Always replaces with a random entry from Onomasticon."""
        
        akk_tokens = transliteration.split()
        unique_found = {} 

        # look for name in onomasticon
        for token in akk_tokens:
            base, pref, suff = self._find_akk_name_with_morpheme(token, self.pn_akk_set)
            if base:
                base_low = base.lower()
                eng_name = self.pn_akk_to_eng[base_low]
                if self._find_english_name(eng_name, translation):
                    if base_low not in unique_found:
                        unique_found[base_low] = (eng_name, set())
                    unique_found[base_low][1].add((base, pref, suff))

        # look for name in lexicon if name was not in onomasticon
        if not unique_found:
            for token in akk_tokens:
                base, pref, suff = self._find_akk_name_with_morpheme(token, self.pn_set)
                if base:
                    base_low = base.lower()
                    eng_name = self.pn_form_to_norm.get(base_low, base)
                    if self._find_english_name(eng_name, translation):
                        if base_low not in unique_found:
                            unique_found[base_low] = (eng_name, set())
                        unique_found[base_low][1].add((base, pref, suff))

        # both methods failed to find any names, return False
        if not unique_found:
            return False, transliteration, translation

        # swap with random name from nomasticon 
        new_transliteration, new_translation = transliteration, translation
        for base_low, (old_eng, forms) in unique_found.items():
            
            if self.onomasticon_df is not None and not self.onomasticon_df.empty:
                rand_row = self.onomasticon_df.sample(n=1).iloc[0]
                rep_akk, rep_eng = str(rand_row["transliteration"]), str(rand_row["translation"])
            else:
                return False, transliteration, translation

            for b_orig, p, s in forms:
                old_f = p + b_orig + s
                new_f = p + rep_akk + s
                new_transliteration = re.sub(r"\b" + re.escape(old_f) + r"\b", new_f, new_transliteration)
            
            new_translation = re.sub(r"\b" + re.escape(old_eng) + r"\b", rep_eng, new_translation, flags=re.IGNORECASE)

        return True, new_transliteration, new_translation

    def swap_gn(self, transliteration: str, translation: str):
        """Identifies GN using Lexicon and swaps using the hardcoded GN bank."""
        akk_tokens = transliteration.split()
        unique_found = {}

        for token in akk_tokens:
            base, pref, suff = self._find_akk_name_with_morpheme(token, self.gn_set)
            if base:
                base_low = base.lower()
                eng_norm = self.gn_form_to_norm.get(base_low, base)
                if self._find_english_name(eng_norm, translation):
                    if base_low not in unique_found:
                        unique_found[base_low] = (eng_norm, set())
                    unique_found[base_low][1].add((base, pref, suff))

        if not unique_found:
            return False, transliteration, translation

        new_translit, new_transl = transliteration, translation
        for base_low, (old_eng, forms) in unique_found.items():
            rep_akk = random.choice(self.akk_gn_list)
            rep_eng = self.gn_bank[rep_akk]

            for b_orig, p, s in forms:
                old_f = p + b_orig + s
                new_f = p + rep_akk + s
                new_translit = re.sub(r"\b" + re.escape(old_f) + r"\b", new_f, new_translit)
            
            new_transl = re.sub(r"\b" + re.escape(old_eng) + r"\b", rep_eng, new_transl, flags=re.IGNORECASE)

        return True, new_translit, new_transl