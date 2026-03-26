"""
alignment_reviewer.py
─────────────────────
Sentence-alignment pipeline for Akkadian / English parallel texts.

USAGE
─────
    from alignment_reviewer import build_patterns, process_dataframe, BatchReviewer, build_final_dataset

    # Pass your own compiled patterns (recommended — stays in sync with your highlighter)
    akk_re, eng_re = build_my_patterns()
    clean, flagged = process_dataframe(
        df,
        translit_col=Columns.TRANSLITERATION,
        transl_col=Columns.TRANSLATION,
        substring='KIŠIB',          # case-insensitive, checked in both columns
        akk_pattern=akk_re, akk_group=1,
        eng_pattern=eng_re, eng_group=2,
        ratio_min=0.7, ratio_max=1.4,
        pos_threshold=0.15,
    )

    reviewer = BatchReviewer(flagged)
    reviewer.show()

    dataset = build_final_dataset(clean + flagged)
    # → list of {'row_index', 'segment_idx', 'akk', 'eng'}
    import pandas as pd
    df_aligned = pd.DataFrame(dataset)
"""

import re
import html
import copy
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

WORD_RATIO_MIN = 0.8
WORD_RATIO_MAX = 1.2
POSITION_THRESHOLD = 0.10
MERGE_GAP_SIZE = 5


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════


@dataclass
class MatchSpan:
    text: str
    start: int
    end: int
    position_pct: float


@dataclass
class SuspicionFlag:
    kind: str  # 'count_mismatch' | 'position' | 'ratio' | 'no_match'
    description: str
    akk_match_idx: Optional[int] = None
    eng_match_idx: Optional[int] = None


@dataclass
class DatapointResult:
    row_index: int
    akk_text: str
    eng_text: str
    akk_matches: List[MatchSpan]
    eng_matches: List[MatchSpan]
    flags: List[SuspicionFlag]
    source_id: Optional[any] = field(default=None)  # value from your id column

    # user adjustments — None means "use the original regex matches"
    custom_akk_matches: Optional[List[MatchSpan]] = field(default=None)
    custom_eng_matches: Optional[List[MatchSpan]] = field(default=None)

    skip_akk_indices: List[int] = field(default_factory=list)
    skip_eng_indices: List[int] = field(default_factory=list)
    discard: bool = False
    approved: bool = False

    def effective_akk(self) -> List[MatchSpan]:
        return (
            self.custom_akk_matches
            if self.custom_akk_matches is not None
            else self.akk_matches
        )

    def effective_eng(self) -> List[MatchSpan]:
        return (
            self.custom_eng_matches
            if self.custom_eng_matches is not None
            else self.eng_matches
        )


# ═══════════════════════════════════════════════════════════════
# BUILT-IN PATTERN BUILDER  (mirrors highlight_witnessed_by)
# ═══════════════════════════════════════════════════════════════


def build_patterns() -> Tuple[re.Pattern, re.Pattern]:
    """
    Returns (akk_re, eng_re) matching the logic of highlight_witnessed_by.
    akk_re : group(1) = match text
    eng_re : group(1) = boundary prefix, group(2) = match text
    """
    name_chars = r"[^\s,]+"
    gap_token = (
        r"(?:\s*<[^>]*>"
        r"(?:\s+(?!IGI\b|DUMU\b|KISIB\b|KU\.BABBAR\b|witnessed\b|by\b)[^\s<]+)?)*"
    )
    conj_akk = r"(?:\s+(?:u|and)\s+(?!KISIB)" + name_chars + r")?"
    relation_akk = (
        r"(?:\s+(?:DUMU|a-hu-u)\s+[^\s,;<]+(?:\s*<[^>]*>)*"
        r"(?:\s+(?!DUMU\b|IGI\b|a-hu-u\b|a-ha-ma\b|a-na\b|i-na\b)[^\s,;]*-[^\s,;]+"
        r"(?:\s*<[^>]*>)*)*)*"
    )
    akk_name = (
        r"[^\s]*[-\.][^\s]*[-\.][^\s]+"
        r"|[^\s]*-[^\s]+(?:\s+(?!DUMU\b|IGI\b|a-hu-u\b)[^\s]*-[^\s]+)*"
        r"|[^\s]*\.[^\s]+"
    )
    akk_igi_unit = (
        rf"IGI\s+(?!GIR\b|a-we-li\b)"
        rf"(?:(?:{akk_name}){gap_token}(?:{relation_akk})"
        rf"|(?:DUMU|a-hu-u)\s+{name_chars}{gap_token}{relation_akk})"
        rf"{conj_akk}\s*(?!\d)"
    )
    akk_igi_multi = rf"(?:{akk_igi_unit}\s*){{2,}}"
    akk_igi_single = rf"{akk_igi_unit}$"
    akk_igi = rf"(?:{akk_igi_multi}|{akk_igi_single})"
    akk_kisib = rf"(?:KISIB\s+{name_chars}{relation_akk}{conj_akk}\s*)+"
    akk_pattern = rf"(?:{akk_kisib}|{akk_igi})"
    akk_re = re.compile(rf"({akk_pattern})", re.IGNORECASE | re.MULTILINE)

    occupations = r"(?:packer|scribe|merchant|smith|weaver|potter|herald|priestess)"
    name_or_gap = rf"(?:{name_chars}|{gap_token})"
    possessive_witness = (
        r"(?:his|her)\s+(?:son|daughter|brother|sister|mother|wife|partner)"
    )
    relation_eng = (
        r"(?:[,\s]+(?:"
        r"(?:his|her)\s+(?:brother|sister|son|daughter|mother|father|wife|partner)"
        r"|the\s+[^\s,;.]+(?:\s+of\s+" + name_or_gap + r")?"
        r"|(?:younger|elder|older)\s+(?:son|daughter)"
        r"|[^\s,;]+'s\s+(?:younger|elder|older)?\s*(?:son|daughter|"
        + occupations
        + r")"
        r"|(?:s\.|son|\(grand\)son|grandson|brother|sister|mother|father|daughter|wife|"
        + occupations
        + r")\s+of\s+"
        + gap_token
        + name_chars
        + r"))*"
    )
    conj_eng = (
        r"(?:"
        r"(?:,\s*(?!(?:and|u|by)\s|<[^>]+>)" + name_chars + r")*"
        r"(?:,?\s*(?:and|u)\s+(?!witnessed\s+by|by\s)" + name_chars + r")?)"
    )
    possessive_prefix = (
        r"(?:(?:his|her)\s+(?:son|daughter|brother|sister|mother|wife)\s+)?"
    )
    eng_unit = (
        rf"witnessed\s+by\s+"
        rf"(?:{possessive_witness}|{possessive_prefix}{name_chars})"
        rf"{gap_token}{relation_eng}{conj_eng}"
    )
    trailing_collective = (
        r"(?:,\s*the\s+(?:sons|daughters|brothers|sisters)\s+of\s+"
        + name_or_gap
        + r")?"
    )
    eng_pattern = (
        rf"{eng_unit}"
        rf"(?:[\s,;]+(?:and\s+)?(?:witnessed\s+)?by\s+"
        rf"(?:{possessive_witness}|{possessive_prefix}(?:(?![,;]){name_chars})?)"
        rf"{gap_token}{relation_eng}{conj_eng}{gap_token})*"
        rf"{trailing_collective}{gap_token}[,;]?"
    )
    eng_full = (
        rf'(^\s*|\n\s*|[.,;:"\u201C\u201D]\s*|<[^>]*>\s*)'
        rf"(?=witnessed\s+by)({eng_pattern})"
    )
    eng_re = re.compile(eng_full, re.IGNORECASE | re.MULTILINE)
    return akk_re, eng_re


AKK_RE, ENG_RE = build_patterns()


# ═══════════════════════════════════════════════════════════════
# SPAN ADJUSTMENT HELPERS
# ═══════════════════════════════════════════════════════════════


def _words_before(text: str, pos: int) -> List[Tuple[int, int]]:
    """Return list of (start, end) for whitespace-separated tokens before pos."""
    tokens = list(re.finditer(r"\S+", text[:pos]))
    return [(m.start(), m.end()) for m in tokens]


def _words_after(text: str, pos: int) -> List[Tuple[int, int]]:
    """Return list of (start, end) for whitespace-separated tokens from pos onward."""
    tokens = list(re.finditer(r"\S+", text[pos:]))
    return [(pos + m.start(), pos + m.end()) for m in tokens]


def span_extend_left(full_text: str, span: MatchSpan) -> MatchSpan:
    """Include the next word to the LEFT of the span."""
    before = _words_before(full_text, span.start)
    if not before:
        return span
    word_start, word_end = before[-1]
    new_start = word_start
    new_text = full_text[new_start : span.end]
    return MatchSpan(new_text, new_start, span.end, new_start / max(len(full_text), 1))


def span_shrink_left(full_text: str, span: MatchSpan) -> MatchSpan:
    """Remove the first word from the LEFT of the span."""
    tokens = list(re.finditer(r"\S+", span.text))
    if len(tokens) <= 1:
        return span  # can't shrink to nothing
    second_tok_start = span.start + tokens[1].start()
    new_text = full_text[second_tok_start : span.end]
    return MatchSpan(
        new_text, second_tok_start, span.end, second_tok_start / max(len(full_text), 1)
    )


def span_extend_right(full_text: str, span: MatchSpan) -> MatchSpan:
    """Include the next word to the RIGHT of the span."""
    after = _words_after(full_text, span.end)
    if not after:
        return span
    word_start, word_end = after[0]
    new_end = word_end
    new_text = full_text[span.start : new_end]
    return MatchSpan(new_text, span.start, new_end, span.position_pct)


def span_shrink_right(full_text: str, span: MatchSpan) -> MatchSpan:
    """Remove the last word from the RIGHT of the span."""
    tokens = list(re.finditer(r"\S+", span.text))
    if len(tokens) <= 1:
        return span
    last_tok_start = span.start + tokens[-1].start()
    new_end = last_tok_start
    while new_end > span.start and full_text[new_end - 1] in " \t":
        new_end -= 1
    new_text = full_text[span.start : new_end]
    return MatchSpan(new_text, span.start, new_end, span.position_pct)


# ═══════════════════════════════════════════════════════════════
# MERGE CLOSE SPANS
# ═══════════════════════════════════════════════════════════════


def merge_close_match_spans(
    text: str, spans: List[MatchSpan], gap_size: int = MERGE_GAP_SIZE
) -> List[MatchSpan]:
    if len(spans) < 2:
        return spans
    merged = [spans[0]]
    for nxt in spans[1:]:
        prev = merged[-1]
        gap_text = text[prev.end : nxt.start]
        gap_words = len(gap_text.split())
        has_digit_or_dot = bool(re.search(r"\d|\.", gap_text))
        prev_ends_with_dot = prev.text.rstrip().endswith(".")
        if gap_words <= gap_size and not has_digit_or_dot and not prev_ends_with_dot:
            merged[-1] = MatchSpan(
                text=text[prev.start : nxt.end],
                start=prev.start,
                end=nxt.end,
                position_pct=prev.position_pct,
            )
        else:
            merged.append(nxt)
    return merged


# ═══════════════════════════════════════════════════════════════
# MATCH EXTRACTION
# ═══════════════════════════════════════════════════════════════


def extract_matches(
    text: str,
    lang: str,
    pattern: Optional[re.Pattern] = None,
    group: Optional[int] = None,
    gap_size: int = MERGE_GAP_SIZE,
) -> List[MatchSpan]:
    text = unicodedata.normalize("NFC", text)
    text_len = max(len(text), 1)
    if pattern is None:
        pattern = AKK_RE if lang == "akk" else ENG_RE
    if group is None:
        group = 1 if lang == "akk" else 2
    spans = []
    for m in pattern.finditer(text):
        try:
            s, e, t = m.start(group), m.end(group), m.group(group)
        except IndexError:
            continue
        if t:
            spans.append(MatchSpan(t, s, e, s / text_len))
    if gap_size > 0:
        spans = merge_close_match_spans(text, spans, gap_size)
    return spans


# ═══════════════════════════════════════════════════════════════
# SPLIT & WORD RATIO
# ═══════════════════════════════════════════════════════════════


def _word_count(text: str) -> int:
    return len(text.split())


def _word_ratio(a: str, b: str) -> Optional[float]:
    wb = _word_count(b)
    return (_word_count(a) / wb) if wb else None


def split_at_matches(
    text: str, matches: List[MatchSpan], skip_indices: List[int]
) -> List[str]:
    active = [m for i, m in enumerate(matches) if i not in skip_indices]
    if not active:
        return [text]
    segments, cursor = [], 0
    for m in active:
        segments.append(text[cursor : m.start])
        segments.append(m.text)
        cursor = m.end
    segments.append(text[cursor:])
    return segments


# ═══════════════════════════════════════════════════════════════
# SUSPICION DETECTION
# ═══════════════════════════════════════════════════════════════


def detect_flags(
    akk_text: str,
    eng_text: str,
    akk_matches: List[MatchSpan],
    eng_matches: List[MatchSpan],
    ratio_min: float = WORD_RATIO_MIN,
    ratio_max: float = WORD_RATIO_MAX,
    pos_threshold: float = POSITION_THRESHOLD,
) -> List[SuspicionFlag]:
    flags: List[SuspicionFlag] = []

    if len(akk_matches) != len(eng_matches):
        flags.append(
            SuspicionFlag(
                kind="count_mismatch",
                description=(
                    f"Akkadian has {len(akk_matches)} match(es), "
                    f"English has {len(eng_matches)} match(es)."
                ),
            )
        )

    for i, (am, em) in enumerate(zip(akk_matches, eng_matches)):
        diff = abs(am.position_pct - em.position_pct)
        if diff > pos_threshold:
            flags.append(
                SuspicionFlag(
                    kind="position",
                    description=(
                        f"Match pair {i}: Akkadian at {am.position_pct:.1%}, "
                        f"English at {em.position_pct:.1%} (delta {diff:.1%})"
                    ),
                    akk_match_idx=i,
                    eng_match_idx=i,
                )
            )

    if len(akk_matches) == len(eng_matches) and akk_matches:
        for i, (a_seg, e_seg) in enumerate(
            zip(
                split_at_matches(akk_text, akk_matches, []),
                split_at_matches(eng_text, eng_matches, []),
            )
        ):
            r = _word_ratio(a_seg, e_seg)
            if r is not None and not (ratio_min <= r <= ratio_max):
                flags.append(
                    SuspicionFlag(
                        kind="ratio",
                        description=f"Segment {i}: word ratio {r:.2f} (outside [{ratio_min}, {ratio_max}])",
                        akk_match_idx=i if i < len(akk_matches) else None,
                        eng_match_idx=i if i < len(eng_matches) else None,
                    )
                )
    return flags


# ═══════════════════════════════════════════════════════════════
# PROCESS DATAFRAME
# ═══════════════════════════════════════════════════════════════


def process_dataframe(
    df,
    translit_col: str,
    transl_col: str,
    substring: str = "",
    ratio_min: float = WORD_RATIO_MIN,
    ratio_max: float = WORD_RATIO_MAX,
    pos_threshold: float = POSITION_THRESHOLD,
    gap_size: int = MERGE_GAP_SIZE,
    akk_pattern: Optional[re.Pattern] = None,
    akk_group: int = 1,
    eng_pattern: Optional[re.Pattern] = None,
    eng_group: int = 2,
    id_col: Optional[str] = None,  # name of your id column
    must_match_akkadian=True,
    must_match_english=True,
) -> Tuple[List[DatapointResult], List[DatapointResult]]:
    """
    Filter rows by substring (case-insensitive, both columns), extract matches,
    detect suspicious alignments.  Returns (clean, flagged).
    """
    if substring:
        mask = df[translit_col].str.contains(substring, case=False, na=False) | df[
            transl_col
        ].str.contains(substring, case=False, na=False)
        filtered = df[mask]
        print(f"Rows matching '{substring}': {len(filtered)}")
    else:
        filtered = df

    clean: List[DatapointResult] = []
    flagged: List[DatapointResult] = []

    for idx, row in filtered.iterrows():
        akk = unicodedata.normalize("NFC", str(row[translit_col]))
        eng = unicodedata.normalize("NFC", str(row[transl_col]))

        akk_matches = extract_matches(
            akk, "akk", pattern=akk_pattern, group=akk_group, gap_size=gap_size
        )
        eng_matches = extract_matches(
            eng, "eng", pattern=eng_pattern, group=eng_group, gap_size=gap_size
        )

        no_akk = must_match_akkadian and not akk_matches
        no_eng = must_match_english and not eng_matches

        # Skip rows where both sides have no matches — truly nothing to work with
        if no_akk and no_eng:
            continue

        flags = detect_flags(
            akk, eng, akk_matches, eng_matches, ratio_min, ratio_max, pos_threshold
        )

        # If only one side is missing, flag it so the reviewer can add matches manually
        if no_akk:
            flags.append(
                SuspicionFlag(
                    kind="no_match",
                    description=(
                        "Akkadian side has no matches — add them manually in the reviewer."
                    ),
                )
            )
        if no_eng:
            flags.append(
                SuspicionFlag(
                    kind="no_match",
                    description=(
                        "English side has no matches — add them manually in the reviewer."
                    ),
                )
            )

        result = DatapointResult(
            row_index=idx,
            akk_text=akk,
            eng_text=eng,
            akk_matches=akk_matches,
            eng_matches=eng_matches,
            flags=flags,
            source_id=row[id_col] if id_col and id_col in df.columns else None,
        )
        if flags:
            flagged.append(result)
        else:
            result.approved = True
            clean.append(result)

    print(
        f"Processed {len(clean) + len(flagged)} rows — "
        f"{len(clean)} clean  |  {len(flagged)} flagged"
    )
    return clean, flagged


# ═══════════════════════════════════════════════════════════════
# APPLY DECISION / FINAL DATASET
# ═══════════════════════════════════════════════════════════════


def split_with_exhaustion(
    akk_text: str,
    eng_text: str,
    akk_matches: List[MatchSpan],
    eng_matches: List[MatchSpan],
    skip_akk: List[int],
    skip_eng: List[int],
) -> List[Tuple[str, str]]:
    """
    Split both texts at their respective matches, stopping as soon as either
    side runs out of remaining text after a match (Option B):
    the final match on the exhausted side absorbs all remaining text on
    the other side so the last pair is always balanced.
    """
    active_akk = [m for i, m in enumerate(akk_matches) if i not in skip_akk]
    active_eng = [m for i, m in enumerate(eng_matches) if i not in skip_eng]

    if not active_akk or not active_eng:
        return [(akk_text, eng_text)]
    pairs = []
    akk_cur = 0
    eng_cur = 0

    for akk_m, eng_m in zip(active_akk, active_eng):
        akk_pre = akk_text[akk_cur : akk_m.start]
        eng_pre = eng_text[eng_cur : eng_m.start]
        if akk_pre or eng_pre:
            pairs.append((akk_pre, eng_pre))

        akk_cur = akk_m.end
        eng_cur = eng_m.end

        akk_remaining = akk_text[akk_cur:]
        eng_remaining = eng_text[eng_cur:]

        akk_exhausted = len(akk_remaining.strip()) == 0
        eng_exhausted = len(eng_remaining.strip()) == 0

        if akk_exhausted or eng_exhausted:
            akk_match_text = akk_m.text + akk_remaining
            eng_match_text = eng_m.text + eng_remaining
            pairs.append((akk_match_text, eng_match_text))
            return pairs

        pairs.append((akk_m.text, eng_m.text))

    akk_tail = akk_text[akk_cur:]
    eng_tail = eng_text[eng_cur:]
    if akk_tail or eng_tail:
        pairs.append((akk_tail, eng_tail))

    return pairs


def apply_decision(
    result: DatapointResult, mode: str = "split"
) -> Optional[List[Tuple[str, str]]]:
    """
    mode='split'        : keep everything — pre, match, inter, post all become segments
    mode='matches_only' : only the matched spans are kept, all other text is discarded
    """
    if result.discard:
        return None

    active_akk = [
        m
        for i, m in enumerate(result.effective_akk())
        if i not in result.skip_akk_indices
    ]
    active_eng = [
        m
        for i, m in enumerate(result.effective_eng())
        if i not in result.skip_eng_indices
    ]

    if mode == "matches_only":
        if not active_akk or not active_eng:
            return None
        return [(a.text, e.text) for a, e in zip(active_akk, active_eng)]

    # mode == 'split'
    return split_with_exhaustion(
        result.akk_text,
        result.eng_text,
        result.effective_akk(),
        result.effective_eng(),
        result.skip_akk_indices,
        result.skip_eng_indices,
    )


def build_final_dataset(
    results: List[DatapointResult], split_fn=None, mode: str = "split"
) -> List[dict]:
    """
    mode='split'        : keep everything — pre, match, inter, post all become segments
    mode='matches_only' : only the highlighted match spans are kept as datapoints

    split_fn overrides apply_decision entirely if provided.
    """
    rows = []
    for r in results:
        if not r.approved and r.flags:
            print(
                f"  [INFO] row {r.row_index} was not reviewed — saving with original matches"
            )
        pairs = split_fn(r) if split_fn is not None else apply_decision(r, mode=mode)
        if pairs is None:
            continue
        for seg_i, (akk, eng) in enumerate(pairs):
            rows.append(
                {
                    "source_id": r.source_id,
                    "row_index": r.row_index,
                    "segment_idx": seg_i,
                    "akk": akk,
                    "eng": eng,
                }
            )
    return rows


# ═══════════════════════════════════════════════════════════════
# JUPYTER BATCH REVIEWER
# ═══════════════════════════════════════════════════════════════


class BatchReviewer:
    """
    Interactive Jupyter widget for reviewing flagged alignment datapoints.

    Features per match:
      - Skip checkbox          : exclude this match from splitting
      - Extend / shrink buttons: adjust match boundaries word-by-word
      - Reset button           : restore original regex match
      - Add match widget       : manually add a match by substring search
    """

    _AKK_CLR = "#1a3a5c"
    _ENG_CLR = "#1a3d2b"
    _FLAG_BG = "#3d2e00"
    _TEXT_CLR = "#e8e8e8"

    def __init__(
        self,
        flagged: List[DatapointResult],
        ratio_min: float = WORD_RATIO_MIN,
        ratio_max: float = WORD_RATIO_MAX,
        pos_threshold: float = POSITION_THRESHOLD,
    ):
        self.flagged = flagged
        self.current_idx = 0
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.pos_threshold = pos_threshold

        if not flagged:
            print("No flagged items to review.")
            self.ui = widgets.HTML("<i>Nothing to review.</i>")
            return

        self._build_skeleton()

    def show(self):
        display(self.ui)
        if self.flagged:
            self._render()

    def get_approved(self) -> List[DatapointResult]:
        return [r for r in self.flagged if r.approved and not r.discard]

    def get_discarded(self) -> List[DatapointResult]:
        return [r for r in self.flagged if r.discard]

    # ── skeleton ──────────────────────────────────────────────

    def _build_skeleton(self):
        self._progress = widgets.HTML()
        self._prev_btn = widgets.Button(
            description="Prev", layout=widgets.Layout(width="80px")
        )
        self._next_btn = widgets.Button(
            description="Next", layout=widgets.Layout(width="80px")
        )
        self._prev_btn.on_click(lambda _: self._navigate(-1))
        self._next_btn.on_click(lambda _: self._navigate(+1))
        self._summ_btn = widgets.Button(
            description="Summary",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self._summ_btn.on_click(lambda _: self._render_summary())
        self._discard_all_btn = widgets.Button(
            description="Discard All",
            button_style="danger",
            layout=widgets.Layout(width="110px"),
        )
        self._discard_all_btn.on_click(self._on_discard_all)
        self._flags_out = widgets.Output()  # static: flag list
        self._text_out = widgets.Output()  # dynamic: annotated text + segments
        self._ctrl_out = widgets.Output()  # controls
        self._summ_out = widgets.Output()  # summary
        self._apply_out = widgets.Output()  # persistent apply confirmation
        self.ui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self._prev_btn,
                        self._next_btn,
                        self._progress,
                        self._summ_btn,
                        self._discard_all_btn,
                    ]
                ),
                self._apply_out,
                self._flags_out,
                self._text_out,
                self._ctrl_out,
                self._summ_out,
            ]
        )

    # ── navigation ────────────────────────────────────────────

    def _navigate(self, delta: int):
        # auto-apply current item before moving
        if hasattr(self, "_pending_apply"):
            self._pending_apply()
        new = self.current_idx + delta
        if 0 <= new < len(self.flagged):
            self.current_idx = new
            with self._apply_out:
                clear_output()
            self._render()

    def _render(self):
        r = self.flagged[self.current_idx]
        reviewed = sum(1 for x in self.flagged if x.approved or x.discard)
        status = (
            "approved"
            if (r.approved and not r.discard)
            else "discarded"
            if r.discard
            else "pending"
        )
        self._progress.value = (
            f"&nbsp;<b>{self.current_idx + 1} / {len(self.flagged)}</b>"
            f"&nbsp; row <code>{r.row_index}</code>"
            f"&nbsp; [{status}]"
            f"&nbsp; <small>{reviewed}/{len(self.flagged)} reviewed</small>&nbsp;"
        )
        with self._flags_out:
            clear_output(wait=True)
            display(HTML(self._flags_html(r)))
        with self._text_out:
            clear_output(wait=True)
            display(HTML(self._text_html(r, r.effective_akk(), r.effective_eng())))
        with self._ctrl_out:
            clear_output(wait=True)
            self._render_controls(r)

    def _refresh_text(
        self,
        r: DatapointResult,
        working_akk: List[MatchSpan],
        working_eng: List[MatchSpan],
        skip_akk: List[int],
        skip_eng: List[int],
    ):
        """Redraw just the annotated text + segments table with current working spans."""
        with self._text_out:
            clear_output(wait=True)
            display(
                HTML(self._text_html(r, working_akk, working_eng, skip_akk, skip_eng))
            )

    # ── info panel ────────────────────────────────────────────

    def _flags_html(self, r: DatapointResult) -> str:
        flag_rows = "".join(
            f"<li><code>{f.kind}</code> — {html.escape(f.description)}</li>"
            for f in r.flags
        )
        return (
            f"<div style='background:{self._FLAG_BG};padding:10px;"
            f"border-radius:4px;margin-bottom:10px;color:{self._TEXT_CLR};'>"
            f"<b>Flags:</b><ul style='margin:4px 0 0 16px;'>{flag_rows}</ul></div>"
        )

    def _text_html(
        self,
        r: DatapointResult,
        akk_matches: List[MatchSpan],
        eng_matches: List[MatchSpan],
        skip_akk: Optional[List[int]] = None,
        skip_eng: Optional[List[int]] = None,
    ) -> str:
        if skip_akk is None:
            skip_akk = r.skip_akk_indices
        if skip_eng is None:
            skip_eng = r.skip_eng_indices
        akk_ann = self._annotate(r.akk_text, akk_matches, self._AKK_CLR)
        eng_ann = self._annotate(r.eng_text, eng_matches, self._ENG_CLR)
        side_by_side = (
            "<table style='width:100%;border-collapse:collapse;table-layout:fixed;"
            "background:#1e1e1e;color:#e8e8e8;'><tr>"
            f"<th style='width:50%;background:#2a2a2a;padding:6px;text-align:left;color:#e8e8e8;'>"
            f"Akkadian ({len(akk_matches)} match(es))</th>"
            f"<th style='width:50%;background:#2a2a2a;padding:6px;text-align:left;color:#e8e8e8;'>"
            f"English ({len(eng_matches)} match(es))</th></tr><tr>"
            f"<td style='padding:10px;vertical-align:top;font-family:monospace;"
            f"font-size:0.85em;white-space:pre-wrap;word-break:break-word;color:#e8e8e8;'>{akk_ann}</td>"
            f"<td style='padding:10px;vertical-align:top;font-family:monospace;"
            f"font-size:0.85em;white-space:pre-wrap;word-break:break-word;color:#e8e8e8;'>{eng_ann}</td>"
            "</tr></table>"
        )
        return side_by_side + self._segments_html(
            r, akk_matches, eng_matches, skip_akk, skip_eng
        )

    @staticmethod
    def _annotate(text: str, matches: List[MatchSpan], colour: str) -> str:
        events: List[Tuple[int, str, int]] = []
        for i, m in enumerate(matches):
            events.append((m.start, "open", i))
            events.append((m.end, "close", i))
        events.sort(key=lambda x: (x[0], 0 if x[1] == "open" else 1))
        parts, cursor = [], 0
        for pos, kind, idx in events:
            parts.append(html.escape(text[cursor:pos]))
            if kind == "open":
                parts.append(
                    f'<span style="background:{colour};border:1px solid #555;'
                    f'border-radius:3px;padding:1px 4px;color:#fff;">'
                    f'<sup style="font-weight:bold;color:#ccc;">[{idx}]</sup>'
                )
            else:
                parts.append("</span>")
            cursor = pos
        parts.append(html.escape(text[cursor:]))
        return "".join(parts)

    def _segments_html(
        self,
        r: DatapointResult,
        akk_matches: List[MatchSpan],
        eng_matches: List[MatchSpan],
        skip_akk: List[int],
        skip_eng: List[int],
    ) -> str:
        akk_segs = split_at_matches(r.akk_text, akk_matches, skip_akk)
        eng_segs = split_at_matches(r.eng_text, eng_matches, skip_eng)
        rows = ""
        for i in range(max(len(akk_segs), len(eng_segs))):
            a = akk_segs[i] if i < len(akk_segs) else ""
            e = eng_segs[i] if i < len(eng_segs) else ""
            ratio = _word_ratio(a, e)
            if ratio is None:
                ratio_str, rc = "N/A", "#888"
            elif self.ratio_min <= ratio <= self.ratio_max:
                ratio_str, rc = f"{ratio:.2f} ok", "#4caf50"
            else:
                ratio_str, rc = f"{ratio:.2f} !", "#f44336"
            rows += (
                f"<tr><td style='padding:4px;text-align:center;color:#888;'>{i}</td>"
                f"<td style='padding:4px;font-family:monospace;font-size:0.82em;"
                f"color:#e8e8e8;max-width:300px;overflow:hidden;text-overflow:ellipsis;"
                f"white-space:nowrap;' title='{html.escape(a)}'>"
                f"{html.escape(a[:90])}{'...' if len(a) > 90 else ''}</td>"
                f"<td style='padding:4px;font-family:monospace;font-size:0.82em;"
                f"color:#e8e8e8;max-width:300px;overflow:hidden;text-overflow:ellipsis;"
                f"white-space:nowrap;' title='{html.escape(e)}'>"
                f"{html.escape(e[:90])}{'...' if len(e) > 90 else ''}</td>"
                f"<td style='padding:4px;color:{rc};font-weight:bold;'>{ratio_str}</td></tr>"
            )
        return (
            "<details open style='margin-top:10px;'>"
            "<summary style='cursor:pointer;font-weight:bold;color:#e8e8e8;'>"
            "Resulting segments (updates after Apply)</summary>"
            "<table style='width:100%;border-collapse:collapse;margin-top:6px;"
            "background:#1e1e1e;color:#e8e8e8;'>"
            "<tr style='background:#2a2a2a;'>"
            "<th>#</th><th>Akkadian segment</th><th>English segment</th>"
            "<th>Word ratio</th></tr>" + rows + "</table></details>"
        )

    # ── add-match widget ──────────────────────────────────────

    def _make_add_match_widget(
        self,
        lang: str,
        full_text: str,
        working_list: List[MatchSpan],
        akk_skip_cbs: list,
        eng_skip_cbs: list,
        r: DatapointResult,
        working_akk: List[MatchSpan],
        working_eng: List[MatchSpan],
    ) -> widgets.VBox:
        """
        Widget that lets the user find a substring in the text and add it as a
        new MatchSpan. The new span immediately gets extend/shrink/reset controls
        after the controls panel is re-rendered.
        """
        search_input = widgets.Text(
            placeholder="Type substring to find in text…",
            layout=widgets.Layout(width="400px"),
        )
        occurrence_w = widgets.BoundedIntText(
            value=1,
            min=1,
            max=50,
            step=1,
            description="Occurrence:",
            layout=widgets.Layout(width="160px"),
            style={"description_width": "80px"},
        )
        status_lbl = widgets.HTML()

        add_btn = widgets.Button(
            description="+ Add match",
            button_style="success",
            layout=widgets.Layout(width="110px"),
        )

        def _on_add(_):
            query = search_input.value
            if not query:
                status_lbl.value = (
                    "<span style='color:#f44;'>Enter a substring first.</span>"
                )
                return

            # Find the nth occurrence
            n = occurrence_w.value
            start = 0
            pos = -1
            for _ in range(n):
                pos = full_text.find(query, start)
                if pos == -1:
                    status_lbl.value = (
                        f"<span style='color:#f44;'>Occurrence {n} not found.</span>"
                    )
                    return
                start = pos + 1

            found_start = pos
            found_end = found_start + len(query)

            new_span = MatchSpan(
                text=query,
                start=found_start,
                end=found_end,
                position_pct=found_start / max(len(full_text), 1),
            )
            working_list.append(new_span)

            # ── CRITICAL FIX ────────────────────────────────────────────────
            # Save both working lists to r.custom_* BEFORE re-rendering
            # controls. _render_controls reloads from r.custom_*, so without
            # this the just-added span is lost the moment the panel redraws.
            r.custom_akk_matches = copy.deepcopy(working_akk)
            r.custom_eng_matches = copy.deepcopy(working_eng)
            # ────────────────────────────────────────────────────────────────

            idx_added = len(working_list) - 1
            status_lbl.value = (
                f"<span style='color:#4caf50;'>✔ Added [{idx_added}] "
                f"@ {found_start}–{found_end} — "
                f"use extend/shrink to adjust, then Apply & Refresh to save.</span>"
            )

            # Re-render controls so the new span gets its own row with buttons
            with self._ctrl_out:
                clear_output(wait=True)
                self._render_controls(r)
            # Also refresh the annotated text panel
            skip_akk_now = [i for i, cb in akk_skip_cbs if cb.value]
            skip_eng_now = [i for i, cb in eng_skip_cbs if cb.value]
            self._refresh_text(r, working_akk, working_eng, skip_akk_now, skip_eng_now)

        add_btn.on_click(_on_add)

        label = widgets.HTML(
            f"<b style='color:#aaa;'>Add {lang.upper()} match manually:</b>"
        )
        return widgets.VBox(
            [
                label,
                widgets.HBox([search_input, occurrence_w, add_btn]),
                status_lbl,
            ],
            layout=widgets.Layout(
                margin="8px 0",
                padding="6px",
                border="solid 1px #444",
            ),
        )

    # ── controls panel ────────────────────────────────────────

    def _render_controls(self, r: DatapointResult):
        # Initialise from any previously saved custom matches so edits accumulate
        working_akk = copy.deepcopy(
            r.custom_akk_matches if r.custom_akk_matches is not None else r.akk_matches
        )
        working_eng = copy.deepcopy(
            r.custom_eng_matches if r.custom_eng_matches is not None else r.eng_matches
        )

        decision_w = widgets.RadioButtons(
            options=[
                ("Approve — apply splits as shown", "approve"),
                ("Discard — keep full text as a single pair", "discard"),
            ],
            value="discard" if r.discard else "approve",
            layout=widgets.Layout(margin="0 0 12px 0"),
        )

        # ── per-match widgets ──────────────────────────────────

        def make_match_row(
            lang: str,
            idx: int,
            working_list: List[MatchSpan],
            full_text: str,
            skip_indices_ref: list,
        ):
            """Build a widget row for one match with skip + adjust buttons."""

            preview_lbl = widgets.HTML(
                value=self._match_preview_html(idx, working_list[idx])
            )
            skip_cb = widgets.Checkbox(
                value=(idx in skip_indices_ref),
                description=f"Skip [{idx}]",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="100px"),
            )

            btn_style = widgets.Layout(width="36px", height="28px")
            ext_l = widgets.Button(
                description="◀+", layout=btn_style, tooltip="Extend left"
            )
            shr_l = widgets.Button(
                description="▶-", layout=btn_style, tooltip="Shrink left"
            )
            shr_r = widgets.Button(
                description="-◀", layout=btn_style, tooltip="Shrink right"
            )
            ext_r = widgets.Button(
                description="+▶", layout=btn_style, tooltip="Extend right"
            )
            reset = widgets.Button(
                description="↺",
                layout=widgets.Layout(width="32px", height="28px"),
                tooltip="Reset to original regex match",
                button_style="warning",
            )

            step_w = widgets.BoundedIntText(
                value=1,
                min=1,
                max=50,
                step=1,
                description="",
                layout=widgets.Layout(width="52px", height="28px"),
                style={"description_width": "0px"},
            )

            orig_match = (
                r.akk_matches[idx]
                if lang == "akk" and idx < len(r.akk_matches)
                else r.eng_matches[idx]
                if lang == "eng" and idx < len(r.eng_matches)
                else None
            )

            def _live_refresh():
                preview_lbl.value = self._match_preview_html(idx, working_list[idx])
                skip_akk_now = [i for i, cb in akk_skip_cbs if cb.value]
                skip_eng_now = [i for i, cb in eng_skip_cbs if cb.value]
                self._refresh_text(
                    r, working_akk, working_eng, skip_akk_now, skip_eng_now
                )

            def _apply_n(fn, _):
                for _ in range(step_w.value):
                    working_list[idx] = fn(full_text, working_list[idx])
                _live_refresh()

            ext_l.on_click(lambda _, f=span_extend_left: _apply_n(f, _))
            shr_l.on_click(lambda _, f=span_shrink_left: _apply_n(f, _))
            shr_r.on_click(lambda _, f=span_shrink_right: _apply_n(f, _))
            ext_r.on_click(lambda _, f=span_extend_right: _apply_n(f, _))

            def on_reset(_):
                if orig_match is not None:
                    working_list[idx] = copy.deepcopy(orig_match)
                _live_refresh()

            reset.on_click(on_reset)

            skip_cb.observe(
                lambda _: self._refresh_text(
                    r,
                    working_akk,
                    working_eng,
                    [i for i, cb in akk_skip_cbs if cb.value],
                    [i for i, cb in eng_skip_cbs if cb.value],
                ),
                names="value",
            )

            row = widgets.HBox(
                [
                    skip_cb,
                    widgets.HTML(
                        value="<span style='color:#aaa;font-size:0.85em;margin:0 4px;'>step:</span>"
                    ),
                    step_w,
                    widgets.HTML(
                        value="<span style='color:#aaa;font-size:0.85em;margin:0 4px;'>|</span>"
                    ),
                    ext_l,
                    shr_l,
                    shr_r,
                    ext_r,
                    reset,
                    preview_lbl,
                ],
                layout=widgets.Layout(align_items="center", margin="2px 0"),
            )

            return row, skip_cb

        # Build rows for each language
        akk_skip_cbs = []
        akk_rows = []
        for i in range(len(working_akk)):
            row, cb = make_match_row(
                "akk", i, working_akk, r.akk_text, r.skip_akk_indices
            )
            akk_rows.append(row)
            akk_skip_cbs.append((i, cb))

        eng_skip_cbs = []
        eng_rows = []
        for i in range(len(working_eng)):
            row, cb = make_match_row(
                "eng", i, working_eng, r.eng_text, r.skip_eng_indices
            )
            eng_rows.append(row)
            eng_skip_cbs.append((i, cb))

        apply_btn = widgets.Button(
            description="Apply & Refresh",
            button_style="primary",
            layout=widgets.Layout(margin="12px 0 0 0"),
        )

        def _on_apply(_=None):
            r.custom_akk_matches = copy.deepcopy(working_akk)
            r.custom_eng_matches = copy.deepcopy(working_eng)
            r.skip_akk_indices = [i for i, cb in akk_skip_cbs if cb.value]
            r.skip_eng_indices = [i for i, cb in eng_skip_cbs if cb.value]
            r.discard = decision_w.value == "discard"
            r.approved = True

            # Build confirmation HTML — persists above the re-rendered controls
            action = "DISCARD" if r.discard else "APPROVED"
            action_colour = "#c0392b" if r.discard else "#27ae60"

            def _match_rows(label, matches, skipped):
                if not matches:
                    return f"<tr><td colspan='4' style='color:#888;padding:4px;'>{label}: no matches saved</td></tr>"
                rows = ""
                for i, m in enumerate(matches):
                    skip_note = (
                        " <em style='color:#f39c12;'>(skipped)</em>"
                        if i in skipped
                        else ""
                    )
                    preview = html.escape(
                        m.text[:80] + ("…" if len(m.text) > 80 else "")
                    )
                    rows += (
                        f"<tr>"
                        f"<td style='padding:3px 8px;color:#aaa;'>{label} [{i}]{skip_note}</td>"
                        f"<td style='padding:3px 8px;color:#888;font-size:0.8em;'>{m.start}–{m.end}</td>"
                        f"<td style='padding:3px 8px;color:#888;font-size:0.8em;'>{m.position_pct:.1%}</td>"
                        f"<td style='padding:3px 8px;font-family:monospace;font-size:0.82em;"
                        f"color:#e8e8e8;'>{preview}</td>"
                        f"</tr>"
                    )
                return rows

            table = (
                "<table style='width:100%;border-collapse:collapse;margin-top:6px;"
                "background:#1a1a1a;color:#e8e8e8;font-size:0.9em;'>"
                "<tr style='background:#2a2a2a;'>"
                "<th style='padding:4px 8px;text-align:left;'>Match</th>"
                "<th style='padding:4px 8px;text-align:left;'>Span</th>"
                "<th style='padding:4px 8px;text-align:left;'>Position</th>"
                "<th style='padding:4px 8px;text-align:left;'>Text</th>"
                "</tr>"
                + _match_rows("AKK", r.custom_akk_matches, r.skip_akk_indices)
                + _match_rows("ENG", r.custom_eng_matches, r.skip_eng_indices)
                + "</table>"
            )

            confirmation_html = (
                f"<div style='background:#1e2d1e;border:1px solid {action_colour};"
                f"border-radius:4px;padding:8px 12px;margin-bottom:6px;'>"
                f"<span style='color:{action_colour};font-weight:bold;'>✔ {action}</span>"
                f"<span style='color:#aaa;margin-left:10px;font-size:0.85em;'>"
                f"row {r.row_index} — "
                f"{len(r.custom_akk_matches)} akk match(es), "
                f"{len(r.custom_eng_matches)} eng match(es) saved"
                f"</span>" + table + "</div>"
            )

            with self._apply_out:
                clear_output(wait=True)
                display(HTML(confirmation_html))

            self._render()

        # Store so _navigate can call it before switching items
        self._pending_apply = _on_apply

        apply_btn.on_click(_on_apply)

        remove_btn = widgets.Button(
            description="🗑 Remove completely",
            button_style="danger",
            layout=widgets.Layout(margin="12px 0 0 8px"),
        )

        def _on_remove(_):
            self.flagged.pop(self.current_idx)
            if not self.flagged:
                with self._flags_out:
                    clear_output()
                with self._text_out:
                    clear_output()
                with self._ctrl_out:
                    clear_output(wait=True)
                    display(widgets.HTML("<i>No more items.</i>"))
                return
            self.current_idx = min(self.current_idx, len(self.flagged) - 1)
            self._render()

        remove_btn.on_click(_on_remove)

        children = [
            widgets.HTML("<b>Decision</b>"),
            decision_w,
            widgets.HTML(
                "<b>Akkadian matches</b> "
                "<small style='color:#aaa;'>"
                "◀+ extend left &nbsp; ▶- shrink left &nbsp; "
                "-◀ shrink right &nbsp; +▶ extend right &nbsp; ↺ reset"
                "</small>"
            ),
            *(
                akk_rows
                if akk_rows
                else [widgets.HTML("<i style='color:#888;'>No matches yet.</i>")]
            ),
            self._make_add_match_widget(
                "akk",
                r.akk_text,
                working_akk,
                akk_skip_cbs,
                eng_skip_cbs,
                r,
                working_akk,
                working_eng,
            ),
            widgets.HTML(
                "<b>English matches</b> "
                "<small style='color:#aaa;'>"
                "◀+ extend left &nbsp; ▶- shrink left &nbsp; "
                "-◀ shrink right &nbsp; +▶ extend right &nbsp; ↺ reset"
                "</small>"
            ),
            *(
                eng_rows
                if eng_rows
                else [widgets.HTML("<i style='color:#888;'>No matches yet.</i>")]
            ),
            self._make_add_match_widget(
                "eng",
                r.eng_text,
                working_eng,
                akk_skip_cbs,
                eng_skip_cbs,
                r,
                working_akk,
                working_eng,
            ),
            widgets.HBox([apply_btn, remove_btn]),
        ]

        display(widgets.VBox(children))

    @staticmethod
    def _match_preview_html(idx: int, m: MatchSpan) -> str:
        preview = m.text[:70] + ("..." if len(m.text) > 70 else "")
        return (
            f"<span style='font-family:monospace;font-size:0.82em;color:#bbb;'>"
            f"@{m.position_pct:.1%} &nbsp;"
            f"<span style='color:#e8e8e8;'>{html.escape(preview)}</span></span>"
        )

    def _on_discard_all(self, _):
        for r in self.flagged:
            r.discard = True
            r.approved = True
        self._render()

    # ── summary panel ─────────────────────────────────────────

    def _render_summary(self):
        with self._summ_out:
            clear_output(wait=True)
            approved = self.get_approved()
            discarded = self.get_discarded()
            pending = [r for r in self.flagged if not r.approved and not r.discard]
            rows = ""
            for i, r in enumerate(self.flagged):
                if r.discard:
                    status = "discarded"
                elif r.approved:
                    notes = []
                    if r.skip_akk_indices:
                        notes.append(f"skip Akk {r.skip_akk_indices}")
                    if r.skip_eng_indices:
                        notes.append(f"skip Eng {r.skip_eng_indices}")
                    if r.custom_akk_matches is not None:
                        notes.append("akk adjusted")
                    if r.custom_eng_matches is not None:
                        notes.append("eng adjusted")
                    status = "approved" + (f" ({', '.join(notes)})" if notes else "")
                else:
                    status = "pending"
                flag_kinds = ", ".join(set(f.kind for f in r.flags))
                rows += (
                    f"<tr><td style='padding:4px;text-align:center;'>{i + 1}</td>"
                    f"<td style='padding:4px;'>{r.row_index}</td>"
                    f"<td style='padding:4px;'><code>{flag_kinds}</code></td>"
                    f"<td style='padding:4px;'>{status}</td></tr>"
                )
            display(
                HTML(
                    f"<div style='margin-top:10px;padding:8px;background:#2a2a2a;"
                    f"border-radius:4px;color:#e8e8e8;'>"
                    f"<b>Summary:</b> {len(approved)} approved | "
                    f"{len(discarded)} discarded | {len(pending)} pending</div>"
                    f"<table style='width:100%;border-collapse:collapse;margin-top:8px;"
                    f"background:#1e1e1e;color:#e8e8e8;'>"
                    f"<tr style='background:#2a2a2a;'>"
                    f"<th>#</th><th>Row idx</th><th>Flag types</th><th>Status</th></tr>"
                    + rows
                    + "</table>"
                )
            )
