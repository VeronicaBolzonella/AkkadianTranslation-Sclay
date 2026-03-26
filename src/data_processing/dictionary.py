
import pandas as pd
import re

from processing import TextProcessor

"""Does not work because dictionary has I, II, ... for multiple meanings of the same word 
and -... for suffixes, idk how to handle sorry 
"""

class DictionaryModel:
    def __init__(self, csv_path, TextProcessor):
        """
        Initializes the model by loading the dictionary from the CSV.
        """
        self.df = pd.read_csv(csv_path)
        self.df = pd.read_csv(csv_path).dropna(subset=['word', 'definition'])
        self.processor = TextProcessor()

        # preprocess
        self.df['clean_word'] = self.df['word'].apply(self.normalize_text)
        self.df['clean_definition'] = self.df['definition'].apply(self.normalize_definition)
        
        # Create multiple lookup dictionaries
        self.exact_lookup = {}  # For exact matches (including multi-word)
        self.suffix_lookup = {}  # For suffixes starting with -
        
        for _, row in self.df.iterrows():
            clean_word = row['clean_word']
            definition = row['clean_definition']
            
            if clean_word.startswith('-'):
                # It's a suffix - store without the hyphen
                self.suffix_lookup[clean_word[1:]] = definition
            else:
                # Regular word - store as-is
                self.exact_lookup[clean_word] = definition
    

    def normalize_definition(self, text):
        if not isinstance(text, str):
            # print("NOT SUPPORTED")
            return ""
        pattern = re.compile(r"\"(.+?)\"")
        match = pattern.search(text)

        if match:
            # print("CLEANED: ", match.group(1))
            return match.group(1)
        
    def normalize_text(self, text):
        """
        Standardizes Akkadian using the TextProcessor
        """
        text = self.processor.preprocess_input_text(text)
        return text

    def tokenize(self, sentence):
        """
        Splits a sentence into words, trying longest matches first.
        """
        if not isinstance(sentence, str):
            return []
        
        sentence = self.normalize_text(sentence)
        tokens = []
        words = sentence.split()
        i = 0
        
        while i < len(words):
            # Try matching increasingly longer phrases (greedy matching)
            best_match = None
            best_length = 0
            
            # Try 1, 2, 3, 4 word combinations
            for length in range(1, min(5, len(words) - i + 1)):
                phrase = ' '.join(words[i:i+length])
                if phrase in self.exact_lookup:
                    best_match = phrase
                    best_length = length
            
            if best_match:
                tokens.append(best_match)
                i += best_length
            else:
                tokens.append(words[i])
                i += 1
        
        return tokens

    
    def check_suffix_match(self, token):
        """
        Check if token ends with any known suffix.
        Returns (root, suffix, suffix_meaning) or None.
        """
        for suffix, meaning in self.suffix_lookup.items():
            if token.endswith(suffix) and len(token) > len(suffix):
                root = token[:-len(suffix)]
                return (root, suffix, meaning)
        return None
    
    def translate_token(self, token):
        """
        Looks up a single token with suffix handling.
        """
        # Direct exact lookup
        if token in self.exact_lookup:
            return self.exact_lookup[token]
        
        # Check for suffix match
        suffix_match = self.check_suffix_match(token)
        if suffix_match:
            root, suffix, suffix_meaning = suffix_match
            
            # Try to find the root
            if root in self.exact_lookup:
                root_meaning = self.exact_lookup[root]
                return f"{root_meaning}+{suffix_meaning}"
            else:
                return f"[{root}]+{suffix_meaning}"
        
        # Not found
        return f"[{token}]"
    
    def translate(self, sentence):
        """
        Translates a full sentence word by word.
        """
        tokens = self.tokenize(sentence)
        translation = [self.translate_token(t) for t in tokens]
        
        return {
            "original": sentence,
            "tokens": tokens,
            "translated_tokens": translation,
            "joined": " ".join(translation)
        }
    