"""
All patterns and maps useful for data processing
"""

import re


class CompiledPatterns:
    """Pre-compile all regex patterns for pre and post processing"""

    ################################################################################
    ################################################################################
    ##                                                                            ##
    ##                            AKKADIAN WRITING                                ##
    ##                                                                            ##
    ################################################################################
    ################################################################################

    # Logograms as fully capitalized words
    CHARS_CAPITAL = r"[A-ZŠŠṢṢṬṬḪḪÁÉÍÚÀÈÌÙ]"

    LOGOGRAM_STR = rf"(?!(?:PN)\b){CHARS_CAPITAL}+(?:[\.·]{CHARS_CAPITAL}+)*"

    LOGOGRAM = re.compile(rf"\b({LOGOGRAM_STR})\b")
    LOGOGRAMS_COMPOUND = re.compile(rf"\b({LOGOGRAM_STR})[\.·]({LOGOGRAM_STR})\b")

    # in scraped dataset some logograms are marked by underscores
    UNDERSCORE_LOGOGRAMS = re.compile(r"_([^_]+)_")

    ALL_CHARS = r"a-zA-Z0-9šŠṣṢṭṬḫḪáéíúàèìù<>,·\."
    HYPHENS = re.compile(rf"(?<=[{ALL_CHARS}])\-(?=[{ALL_CHARS}])")

    # Global translation tables
    ACCENT_MAP = {
        "á": "a2",
        "é": "e2",
        "í": "i2",
        "ú": "u2",
        "Á": "A2",
        "É": "E2",
        "Í": "I2",
        "Ú": "U2",
        "à": "a3",
        "è": "e3",
        "ì": "i3",
        "ù": "u3",
        "À": "A3",
        "È": "E3",
        "Ì": "I3",
        "Ù": "U3",
        "š": "sz",
        "Š": "SZ",
        "ṣ": "s,",
        "Ṣ": "S,",
        "ṭ": "t,",
        "Ṭ": "T,",
        "ḫ": "h",
        "Ḫ": "H",
    }

    NORM_PATTERN = re.compile(
        "|".join(re.escape(k) for k in sorted(ACCENT_MAP.keys(), key=len, reverse=True))
    )

    SUBSCRIPT_TABLE = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    # matches normalized and lower case form
    DETERMINATIVE_MAP = {
        "{d}": " god",
        "{mul}": " stars",
        "{ki}": " earth",
        "{lú}": " person",
        "{é}": " building",
        "{uru}": " settlement",
        "{kur}": " land",
        "{munus}": " feminine",
        "{m}": " masculine",
        "{giš}": " wood",
        "TUG": " textile",
        "{dub}": " tablet",
        "{id2}": " river",
        "{mušen}": " bird",
        "{na4}": " stone",
        "{kuš}": " skin",
        "{ú}": " plant",
    }

    UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
    UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"

    DETERMINATIVE_PATTERN = re.compile("|".join(re.escape(k) for k in sorted(DETERMINATIVE_MAP.keys(), key=len, reverse=True)))
    DET_UPPER_RE = re.compile(r"\(([" + UNICODE_UPPER + r"0-9]{1,6})\)")
    DET_LOWER_RE = re.compile(r"\{([" + UNICODE_LOWER + r"]{1,4})\}")

    # fix host said idk why
    # KÙ.B. —> KÙ.BABBAR
    KUB_PATTERN = re.compile(r"k[ùu]3?[.\-](?:babbar|b\.)", re.I)

    # scraped dataset has modifier annotations
    MODIFIERS = re.compile("@(?:[cfgstnzkrnv]|<DIGITS>)")


    # below patterns are from the notebook
    # https://www.kaggle.com/code/giovannyrodrguez/lb-35-9-ensembling-post-processing-baseline
    V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
    V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
    ACUTE = str.maketrans(
        {"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"}
    )
    GRAVE = str.maketrans(
        {"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"}
    )

    ################################################################################
    ################################################################################
    ##                                                                            ##
    ##                                   GAPS                                     ##
    ##                                                                            ##
    ################################################################################
    ################################################################################

    # gaps: x, [x], [xxx...], …, .., (n broken lines)    
    # will be replaced with <gap>
    # probably useless
    GAPS = re.compile(
        r"PN|"                                
        r"\[x*\]|"                             
        r"\[\.\.\.\]|"                        
        r"\sx\s|"                             
        r"\(x\)|"                             
        r"\(\d*\s*broken\s+lines?\b\s*\)|"    
        r"\.{2,}|"                            
        r"…+",                                 
        re.I
    )

    # REMOVED IN NEW VERSION moved to <gap> for all
    # BIG_GAP_INPUT = re.compile(r"(\[x\]|\(x\)|(?<!\w)x(?!\w)|xx+|\s+x\s+)")

    # merge sequences consisting only of <gap>
    # matches: <gap> <gap>, -<gap> <gap>, <gap>-<gap>, etc.
    MERGE_GAPS = re.compile(r"-?<gap>(?:\s*[\-\s]*<gap>)+-?")

    # MERGE_BIG_GAPS: Merges sequences consisting only of <big_gap>
    # matches: <big_gap> <big_gap>, -<big_gap> <big_gap>-, etc.
    MERGE_BIG_GAPS = re.compile(r"-?<big_gap>(?:\s*[\-\s]*<big_gap>)+-?")

    # MERGE_MIXED_GAPS: Merges sequences containing both <gap> and <big_gap>
    # matches: <gap> <big_gap>, -<big_gap> <gap> <big_gap>-, etc.
    MERGE_MIXED_GAPS = re.compile(
        r"-?(?:<gap>|<big_gap>)(?:\s*[\-\s]*(?:<gap>|<big_gap>))+-?"
    )

    # GRAMMAR_ANNOTATIONS = re.compile(
    #     r"\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)", re.I
    # )

    ################################################################################
    ################################################################################
    ##                                                                            ##
    ##                                CLANUPS                                     ##
    ##                                                                            ##
    ################################################################################
    ################################################################################

    # post processing cleanup
    REPEATED_WORDS = re.compile(r"\b(\w+)(?:\s+\1\b)+")
    SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,:])")
    REPEATED_PUNCT = re.compile(r"([.,])\1+")
    MULTI_SPACE = re.compile(r"\s+")

    # N-gram dedup (pre-compiled for sizes 4,3,2)
    NGRAM_PATTERNS = [
        re.compile(r"\b((?:\w+\W+){" + str(n - 1) + r"}\w+)(?:\W+\1\b)+")
        for n in range(6, 1, -1)
    ]

    # BRACKETS_MAP = {"{": "(", "}": ")"}  # not used anymore after dataset change
    # BRACKETS_PATTERN = re.compile(r"[{}]")

    FORBIDDEN_CHARS_INPUT = str.maketrans("", "", "\"—–⌈⌋⌊+'/;#!?")
    SQUARE_BRACKETS = re.compile(r"\[(.*?)\]") # to be removed except in gaps
    ROUND_BRACKETS = re.compile(r"[\(\)](.*?)[\(\)]")

    # FORBIDDEN_CHARS_OUTPUT = str.maketrans("", "", "()—–⌈⌋⌊+/;") # chenged by host


    ENG_STRAY_CLEANUP = re.compile(
        r"\(?\b(fem|sing|plur|plural|pl|masc)\b\.?\)?|" # matches (fem.), (fem), fem., or fem
        r"\.{2,}|"  # matches 1 or 2 periods
        r"\(\?\)"  # matches (?)
        r"<<|>>|",  # matches << or >>
        re.I,
    )

    ENG_SLASH_CHOICE = re.compile(r"(\b[\w\s'-]+)\s*/\s*([\w\s'-]+\b)")

    WORD_EXPANSIONS = [
        # match '-gold' only if it's not preceded by another letter
        # this keeps 'gold' as is, but changes '-gold', not sure if thats what they meant
        (re.compile(r"(?<!\w)-gold\b", re.I), "pašallum gold"),
        # match '-tax' only if it's not preceded by another letter
        (re.compile(r"(?<!\w)-tax\b", re.I), "šadduātum tax"),
        # match 'textile' or 'textiles'
        (re.compile(r"(?<!\w)-textiles?\b", re.I), "kutānum textiles"),
    ]

    ################################################################################
    ################################################################################
    ##                                                                            ##
    ##                            NUMBERS AND SUCH                                ##
    ##                                                                            ##
    ################################################################################
    ################################################################################

    FRACTION_CONVERSION = {
        "½": re.compile(r"\b(0\.5|one[- ]half)\b", re.I),
        "¼": re.compile(r"\b(0\.25|one[- ]fourth|one[- ]quarter)\b", re.I),
        "⅓": re.compile(r"\b(0\.33\d*|one[- ]third)\b", re.I),
        "⅚": re.compile(r"\b(0\.83\d*|five[- ]sixths)\b", re.I),
        "⅝": re.compile(r"\b(0\.625|five[- ]eighths)\b", re.I),
        "⅔": re.compile(r"\b(0\.66\d*|0\.67|two[- ]thirds)\b", re.I),
        "¾": re.compile(r"\b(0\.75|three[- ]fourths|three[- ]quarters)\b", re.I),
        "⅙": re.compile(r"\b(0\.16\d*|0\.17|one[- ]sixth)\b", re.I),
    }

    NUMBERS_MAP = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
    }

    # one" with exceptions:
    # don't match if preceded by 'some', 'any', 'no', 'every', 'each'
    # don't match if followed by 'another'
    # are there other issues with one?
    ONE_CLEAN_REGEX = (
        r"(?<!some\s)(?<!any\s)(?<!no\s)(?<!every\s)(?<!each\s)"
        r"\bone\b"
        r"(?!\s+another)"
    )
    other_nums = "|".join([k for k in NUMBERS_MAP.keys() if k != "one"])
    OTHER_NUMBERS_PATTER = re.compile(rf"\b({other_nums})\b", re.I)

    # roman numbers
    MONTH_NUMERAL_REGEX = re.compile(
        r"\b(Month|Months?|Nisannu|Ajaru|Simanu|Abu|Ululu|Taszritu|Arahsamnu|Kislimu|Tebetu|Szabatu|Addaru)\s+([IVX]+)\b",
        re.I,
    )

    # map for the 12 months
    ROMAN_MAP = {
        "I": "1",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
        "XI": "11",
        "XII": "12",
    }

    # DO NOT change order plzz: Longest/most specific patterns first
    MONEY_PATTERNS = [
        # 5 11/12 shekels
        (re.compile(r"5\s+11\s*/\s*12\s*shekels?", re.I), "6 shekels less 15 grains"),
        # 1/12 shekel (handles: 1/12 shekel, 1/12 shekels, 1/12 (shekel), etc.)
        (re.compile(r"1\s*/\s*12\s*\(?shekels?\)?", re.I), "⅔ shekel 15 grains"),
        # 5/12 shekel
        (re.compile(r"5\s*/\s*12\s*\(?shekels?\)?", re.I), "⅓ shekel 15 grains"),
        
        # 7/12 shekel
        (re.compile(r"7\s*/\s*12\s*\(?shekels?\)?", re.I), "½ shekel 15 grains"),
    ]

