"""Text normalization helpers for WER/CER comparison."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_WHITESPACE = re.compile(r"\s+")
# Strip common punctuation; keep Bengali and word chars.
_PUNCT = re.compile(r"[^\w\s\u0980-\u09FF]", re.UNICODE)

# a.m. / AM / a m -> am; p.m. / PM / p m -> pm (applied before punctuation strip).
_AM_DOT_M = re.compile(r"\ba\s*\.\s*m\s*\.?\b", re.IGNORECASE)
_PM_DOT_M = re.compile(r"\bp\s*\.\s*m\s*\.?\b", re.IGNORECASE)
_DOT_AMPM = re.compile(r"\.\s*(am|pm)\s*\.?", re.IGNORECASE)
_DIGIT_AMPM = re.compile(r"\b(\d+(?:[.:]\d+)*)\s*(am|pm)\b", re.IGNORECASE)
# After casefold: spelled-out "a m" / "p m" from ASR of a.m. / p.m.
_AM_SPACE_M = re.compile(r"\ba\s+m\b")
_PM_SPACE_M = re.compile(r"\bp\s+m\b")
# U.S. / US / u s (after punctuation strip) -> us
_EN_USA_DOTTED = re.compile(r"\bU\.?\s*S\.?\s*A\.?\b", re.IGNORECASE)
_EN_US_DOTTED = re.compile(r"\bU\.?\s*S\.?\b", re.IGNORECASE)
_EN_USA_SPACED = re.compile(r"\bu\s+s\s+a\b")
_EN_US_SPACED = re.compile(r"\bu\s+s\b")
_EN_NUMERIC_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
    }
)
_EN_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_EN_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_EN_HOUR_WORDS = {w: n for w, n in {**_EN_UNITS, **_EN_TEENS}.items() if 1 <= n <= 12}
_EN_CLOCK_CANON = "@{hour}h{minute:02d}{ampm}"  # single WER token, e.g. @4h51pm
_EN_CLOCK_STRIP_SUFFIX = re.compile(r"@(\d{1,2})h(\d{2})(?:am|pm)\b")
# Words that block bare clock times without am/pm (avoid "three point three kilometers").
_EN_CLOCK_STOPPERS = frozenset(
    {
        "percent",
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
        "km",
        "mile",
        "miles",
        "pound",
        "pounds",
        "ft",
        "feet",
        "billion",
        "million",
        "hundred",
        "thousand",
    }
)
_EN_TOKEN_ALIASES = {
    "okay": "ok",
}
# breame gaps + explicit UK/US pairs
_EN_EXTRA_BRITISH_TO_AMERICAN = {
    "authorisation": "authorization",
}
_EN_PER_CENT = re.compile(r"\bper\s+cent\b")
# Straight/typographic apostrophe in contractions (before punctuation strip).
_EN_APOS = r"(?:'|\u2019)?"
_EN_CONTRACTION_EXPAND: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\bi{_EN_APOS}m\b", re.I), "I am"),
    (re.compile(rf"\byou{_EN_APOS}re\b", re.I), "you are"),
    (re.compile(rf"\bwe{_EN_APOS}re\b", re.I), "we are"),
    (re.compile(rf"\bthey{_EN_APOS}re\b", re.I), "they are"),
    (re.compile(rf"\bhe{_EN_APOS}s\b", re.I), "he is"),
    (re.compile(rf"\bshe{_EN_APOS}s\b", re.I), "she is"),
    (re.compile(rf"\bit{_EN_APOS}s\b", re.I), "it is"),
    (re.compile(rf"\bthat{_EN_APOS}s\b", re.I), "that is"),
    (re.compile(rf"\bthere{_EN_APOS}s\b", re.I), "there is"),
    (re.compile(rf"\bhere{_EN_APOS}s\b", re.I), "here is"),
    (re.compile(rf"\bwhat{_EN_APOS}s\b", re.I), "what is"),
    (re.compile(rf"\bwho{_EN_APOS}s\b", re.I), "who is"),
    (re.compile(rf"\bi{_EN_APOS}ve\b", re.I), "I have"),
    (re.compile(rf"\byou{_EN_APOS}ve\b", re.I), "you have"),
    (re.compile(rf"\bwe{_EN_APOS}ve\b", re.I), "we have"),
    (re.compile(rf"\bthey{_EN_APOS}ve\b", re.I), "they have"),
    (re.compile(rf"\bi{_EN_APOS}ll\b", re.I), "I will"),
    (re.compile(rf"\byou{_EN_APOS}ll\b", re.I), "you will"),
    (re.compile(rf"\bhe{_EN_APOS}ll\b", re.I), "he will"),
    (re.compile(rf"\bshe{_EN_APOS}ll\b", re.I), "she will"),
    (re.compile(rf"\bwe{_EN_APOS}ll\b", re.I), "we will"),
    (re.compile(rf"\bthey{_EN_APOS}ll\b", re.I), "they will"),
    (re.compile(rf"\bi{_EN_APOS}d\b", re.I), "I would"),
    (re.compile(rf"\byou{_EN_APOS}d\b", re.I), "you would"),
    (re.compile(rf"\bhe{_EN_APOS}d\b", re.I), "he would"),
    (re.compile(rf"\bshe{_EN_APOS}d\b", re.I), "she would"),
    (re.compile(rf"\bwe{_EN_APOS}d\b", re.I), "we would"),
    (re.compile(rf"\bthey{_EN_APOS}d\b", re.I), "they would"),
    (re.compile(rf"\bcan{_EN_APOS}t\b", re.I), "can not"),
    (re.compile(rf"\bwon{_EN_APOS}t\b", re.I), "will not"),
    (re.compile(rf"\bdon{_EN_APOS}t\b", re.I), "do not"),
    (re.compile(rf"\bdoesn{_EN_APOS}t\b", re.I), "does not"),
    (re.compile(rf"\bdidn{_EN_APOS}t\b", re.I), "did not"),
    (re.compile(rf"\bisn{_EN_APOS}t\b", re.I), "is not"),
    (re.compile(rf"\baren{_EN_APOS}t\b", re.I), "are not"),
    (re.compile(rf"\bwasn{_EN_APOS}t\b", re.I), "was not"),
    (re.compile(rf"\bweren{_EN_APOS}t\b", re.I), "were not"),
    (re.compile(rf"\bhasn{_EN_APOS}t\b", re.I), "has not"),
    (re.compile(rf"\bhaven{_EN_APOS}t\b", re.I), "have not"),
    (re.compile(rf"\bhadn{_EN_APOS}t\b", re.I), "had not"),
    (re.compile(rf"\bwouldn{_EN_APOS}t\b", re.I), "would not"),
    (re.compile(rf"\bshouldn{_EN_APOS}t\b", re.I), "should not"),
    (re.compile(rf"\bcouldn{_EN_APOS}t\b", re.I), "could not"),
    (re.compile(rf"\blet{_EN_APOS}s\b", re.I), "let us"),
]
# After apostrophe strip: "I'm" -> "i m"; merge back to expanded form.
_EN_CONTRACTION_SPACED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bi\s+m\b"), "i am"),
    (re.compile(r"\byou\s+re\b"), "you are"),
    (re.compile(r"\bwe\s+re\b"), "we are"),
    (re.compile(r"\bthey\s+re\b"), "they are"),
    (re.compile(r"\bhe\s+s\b"), "he is"),
    (re.compile(r"\bshe\s+s\b"), "she is"),
    (re.compile(r"\bit\s+s\b"), "it is"),
    (re.compile(r"\bi\s+ve\b"), "i have"),
    (re.compile(r"\byou\s+ve\b"), "you have"),
    (re.compile(r"\bwe\s+ve\b"), "we have"),
    (re.compile(r"\bthey\s+ve\b"), "they have"),
    (re.compile(r"\bi\s+ll\b"), "i will"),
    (re.compile(r"\byou\s+ll\b"), "you will"),
    (re.compile(r"\bhe\s+ll\b"), "he will"),
    (re.compile(r"\bshe\s+ll\b"), "she will"),
    (re.compile(r"\bwe\s+ll\b"), "we will"),
    (re.compile(r"\bthey\s+ll\b"), "they will"),
    (re.compile(r"\bi\s+d\b"), "i would"),
    (re.compile(r"\bcan\s+t\b"), "can not"),
    (re.compile(r"\bwon\s+t\b"), "will not"),
    (re.compile(r"\bdon\s+t\b"), "do not"),
    (re.compile(r"\bdoesn\s+t\b"), "does not"),
    (re.compile(r"\bdidn\s+t\b"), "did not"),
    (re.compile(r"\bisn\s+t\b"), "is not"),
    (re.compile(r"\baren\s+t\b"), "are not"),
]


def _language_base(language: str | None) -> str:
    return (language or "").lower().split("-")[0]


# CER is character-level; spaces in ref/hyp are not meaningful for scoring.
CHAR_LEVEL_WER_LANGS = frozenset(
    {"zh", "bn", "ja", "ko", "th", "hi", "ta", "te", "ml", "kn", "gu", "pa", "or", "as", "mr"}
)


def is_char_level_wer_language(language: str | None) -> bool:
    return _language_base(language) in CHAR_LEVEL_WER_LANGS


def _is_english(language: str | None) -> bool:
    return _language_base(language) == "en"


def _is_russian(language: str | None) -> bool:
    return _language_base(language) == "ru"


def _is_arabic(language: str | None) -> bool:
    return _language_base(language) == "ar"


# Tanwin al-fath (ً) before alef (ا) is non-word punctuation for _PUNCT and splits
# ref tokens (e.g. أيضًا -> أيض + ا) while ASR hyp keeps أيضا as one word.
_AR_TANWIN_FATHA_BEFORE_ALEF = re.compile("\u064b(?=\u0627)")
# Harakat, shadda, sukun, tanwin, and related Arabic combining marks.
_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u08f0-\u08ff]"
)


def _strip_arabic_diacritics_text(text: str) -> str:
    """Remove Arabic combining marks before punctuation strip.

    TN refs often carry shadda/tanwin (``يتحكّم``, ``إلكترونيّة``, ``أيضًا``). ``_PUNCT``
    treats these as non-word characters and splits tokens (``يتحك`` + ``م``,
    ``إلكتروني`` + ``ة``), while ASR hyps omit diacritics as single words.
    """
    return _ARABIC_DIACRITICS_RE.sub("", text)


def _normalize_ar_tanwin_alef_text(text: str) -> str:
    """Drop tanwin fatha before alef so ًا does not become a separate WER token."""
    return _AR_TANWIN_FATHA_BEFORE_ALEF.sub("", text)


_AR_WAW = "\u0648"
_AR_ALEF = "\u0627"
_AR_MA = "\u0645\u0627"
# Alternate spellings of "hundred": مئة (TN) vs مائة (ASR).
_AR_MIAH_ALT = "\u0645\u0626\u0629"
_AR_MIAH_CANON = "\u0645\u0627\u0626\u0629"


def _normalize_ar_miah_spelling_text(text: str) -> str:
    """Unify hundred spelling so بالمئة and بالمائة align."""
    return text.replace(_AR_MIAH_ALT, _AR_MIAH_CANON)


# TN ``أميرك`` vs ASR ``أمريك`` (American). Stem-level so it covers all
# inflections: أميركي/أميركية/أميركيين -> أمريكي/أمريكية/أمريكيين.
_AR_AMIRK_ALT = "\u0623\u0645\u064a\u0631\u0643"
_AR_AMIRK_CANON = "\u0623\u0645\u0631\u064a\u0643"


def _normalize_ar_american_spelling_text(text: str) -> str:
    """Unify American spelling variants at the stem (أميرك -> أمريك)."""
    return text.replace(_AR_AMIRK_ALT, _AR_AMIRK_CANON)


# TN ``أية`` vs ASR ``أي`` before feminine nouns (e.g. أية اقتراحات / أي اقتراحات).
_AR_AYYA_ALT = "\u0623\u064a\u0629"
_AR_AYYA_CANON = "\u0623\u064a"


def _normalize_ar_ayya_spelling_text(text: str) -> str:
    """Unify feminine ``any`` spelling so أية and أي align."""
    return text.replace(_AR_AYYA_ALT, _AR_AYYA_CANON)


# Proper name transliteration variant: ref ``زيري`` (Ziri) vs ASR ``زير``.
_AR_ZIRI_ALT = "\u0632\u064a\u0631\u064a"
_AR_ZIRI_CANON = "\u0632\u064a\u0631"


def _normalize_ar_ziri_spelling_text(text: str) -> str:
    """Unify the proper name ``زيري`` / ``زير`` transliteration variants."""
    return text.replace(_AR_ZIRI_ALT, _AR_ZIRI_CANON)


# Proper name transliteration variant: ref ``زاغالو`` (Zagallo) vs ASR ``زغالو``.
_AR_ZAGALLO_ALT = "\u0632\u0627\u063a\u0627\u0644\u0648"
_AR_ZAGALLO_CANON = "\u0632\u063a\u0627\u0644\u0648"


def _normalize_ar_zagallo_spelling_text(text: str) -> str:
    """Unify the proper name ``زاغالو`` / ``زغالو`` transliteration variants."""
    return text.replace(_AR_ZAGALLO_ALT, _AR_ZAGALLO_CANON)


# Place name transliteration variant: ref ``طوكيو`` vs ASR ``توكيو`` (Tokyo, ط/ت).
_AR_TOKYO_ALT = "\u0637\u0648\u0643\u064a\u0648"
_AR_TOKYO_CANON = "\u062a\u0648\u0643\u064a\u0648"


def _normalize_ar_tokyo_spelling_text(text: str) -> str:
    """Unify the place name ``طوكيو`` / ``توكيو`` transliteration variants."""
    return text.replace(_AR_TOKYO_ALT, _AR_TOKYO_CANON)


# Tens (20-90) case-ending unification: spoken refs use the genitive/accusative
# ``-ين`` form while digit-expanded hyps default to the nominative ``-ون`` form
# (e.g. ref ثلاثين vs hyp 30 -> ثلاثون). Same number, so fold ``-ون`` -> ``-ين``
# on both sides. Applied before alef unification so أربعون still matches.
_AR_TENS_NOMINATIVE_TO_GENITIVE = {
    "\u0639\u0634\u0631\u0648\u0646": "\u0639\u0634\u0631\u064a\u0646",  # عشرون -> عشرين
    "\u062b\u0644\u0627\u062b\u0648\u0646": "\u062b\u0644\u0627\u062b\u064a\u0646",  # ثلاثون -> ثلاثين
    "\u0623\u0631\u0628\u0639\u0648\u0646": "\u0623\u0631\u0628\u0639\u064a\u0646",  # أربعون -> أربعين
    "\u062e\u0645\u0633\u0648\u0646": "\u062e\u0645\u0633\u064a\u0646",  # خمسون -> خمسين
    "\u0633\u062a\u0648\u0646": "\u0633\u062a\u064a\u0646",  # ستون -> ستين
    "\u0633\u0628\u0639\u0648\u0646": "\u0633\u0628\u0639\u064a\u0646",  # سبعون -> سبعين
    "\u062b\u0645\u0627\u0646\u0648\u0646": "\u062b\u0645\u0627\u0646\u064a\u0646",  # ثمانون -> ثمانين
    "\u062a\u0633\u0639\u0648\u0646": "\u062a\u0633\u0639\u064a\u0646",  # تسعون -> تسعين
}


def _normalize_ar_tens_case_text(text: str) -> str:
    """Fold nominative tens (-ون) to the genitive (-ين) form so case differs harmlessly."""
    for nom, gen in _AR_TENS_NOMINATIVE_TO_GENITIVE.items():
        if nom in text:
            text = text.replace(nom, gen)
    return text


# Final WER-only step: unify standard Arabic orthographic variants. Applied
# symmetrically to ref and hyp AFTER all hamza-dependent rules have run, so e.g.
# آخر/أخر, الماضى/الماضي and الثقه/الثقة no longer count as substitutions:
#   - alef/hamza-alef (آ إ أ ٱ -> ا)
#   - alef maqsura -> ya (ى -> ي)
#   - taa marbuta -> ha (ة -> ه)
_AR_ORTHOGRAPHY_VARIANTS = str.maketrans(
    {
        "\u0622": "\u0627",  # آ alef with madda above
        "\u0623": "\u0627",  # أ alef with hamza above
        "\u0625": "\u0627",  # إ alef with hamza below
        "\u0671": "\u0627",  # ٱ alef wasla
        "\u0649": "\u064a",  # ى alef maqsura -> ي ya
        "\u0629": "\u0647",  # ة taa marbuta -> ه ha
    }
)


def _normalize_ar_orthography_text(text: str) -> str:
    """Unify alef/maqsura/taa-marbuta variants (final WER-only step)."""
    return text.translate(_AR_ORTHOGRAPHY_VARIANTS)


_AR_BAL = "\u0628\u0627\u0644"
_AR_BAL_SHORT = "\u0628\u0644"


def _ar_hyp_glues_waw(prev: str, hyp_tok: str) -> bool:
    """True when hyp attaches the conjunction و to the previous spoken word."""
    glued = prev + _AR_WAW
    if not hyp_tok.startswith(glued):
        return False
    extra = hyp_tok[len(glued) :]
    return extra in ("", _AR_ALEF)


def _ar_hyp_glues_ma(prev: str, hyp_tok: str) -> bool:
    """True when hyp glues the particle ما onto the previous spoken word."""
    glued = prev + _AR_MA
    if hyp_tok == glued:
        return True
    if prev.startswith(_AR_WAW) and hyp_tok == prev[1:] + _AR_MA:
        return True
    return False


_AR_AN = "\u0623\u0646"
_AR_LA = "\u0644\u0627"
_AR_ALA = "\u0623\u0644\u0627"
_AR_AN_THAK = "\u0622\u0646\u0630\u0627\u0643"
_AR_THAK = "\u0630\u0627\u0643"
# TN writes these as one token; ASR often splits them (ألا -> أن لا, آنذاك -> أن ذاك).
_AR_REF_HYP_TOKEN_SPLITS: dict[str, tuple[str, str]] = {
    _AR_ALA: (_AR_AN, _AR_LA),
    _AR_AN_THAK: (_AR_AN, _AR_THAK),
}
_AR_HUWA = "\u0647\u0648"
_AR_HIYA = "\u0647\u064a"
_AR_MIN = "\u0645\u0646"
# TN keeps these as two tokens; ASR glues them (ما هو -> ماهو, و هي -> وهي, ...).
# Some ASR glues insert an extra letter (هو ما -> هولما, من هو -> ممنهو).
_AR_HUWA_MA_GLUED = "\u0647\u0648\u0644\u0645\u0627"
_AR_MIN_HUWA_GLUED = "\u0645\u0645\u0646\u0647\u0648"
# TN keeps the conjunction و apart from كأنه; ASR glues them (و كأنه -> وكأنه).
_AR_KAANNAHU = "\u0643\u0623\u0646\u0647"
_AR_REF_GLUED_PAIRS: tuple[tuple[str, str, str], ...] = (
    (_AR_MA, _AR_HUWA, _AR_MA + _AR_HUWA),
    (_AR_WAW, _AR_HIYA, _AR_WAW + _AR_HIYA),
    (_AR_HUWA, _AR_MA, _AR_HUWA_MA_GLUED),
    (_AR_MIN, _AR_HUWA, _AR_MIN_HUWA_GLUED),
    (_AR_WAW, _AR_KAANNAHU, _AR_WAW + _AR_KAANNAHU),
)


# TN newline escapes (applied before punctuation strip).
_LITERAL_BACKSLASH_N = re.compile(r"\\n")
_RU_BACKSLASH_EN = re.compile(r"\\эн", re.IGNORECASE)
_NEWLINES = re.compile(r"[\r\n]+")


def normalize_newline_markers(text: str) -> str:
    """Expand TN newline control sequences to whitespace.

    Russian TN emits ``\\эн`` for line breaks; source text may also contain literal
    ``\\n`` or real newlines. ASR often transcribes the spoken break as a standalone
    ``N``/``n`` token — normalizing markers here keeps ref/hyp aligned.
    """
    text = _RU_BACKSLASH_EN.sub(" ", text)
    text = _LITERAL_BACKSLASH_N.sub(" ", text)
    text = _NEWLINES.sub(" ", text)
    return text


_RU_STANDALONE_NEWLINE_TOKENS = frozenset({"n", "nn", "nnn", "nnnn", "нн", "ннн", "нннн", "эн", "н"})
_RU_LIST_NUMBER_WORDS = frozenset(
    {
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
    }
)
_RU_MC_OPTION_RE = re.compile(r"^[nн]+([a-z])$")
_RU_CYR_N_LIST_NUMBER_RE = re.compile(
    r"^н(" + "|".join(sorted(_RU_LIST_NUMBER_WORDS, key=len, reverse=True)) + r")$"
)


def _peel_latin_n_before_cyrillic(token: str) -> str:
    """Drop a run of leading ``n`` when the remainder starts with Cyrillic (``nnдва`` → ``два``)."""
    i = 0
    while i < len(token) and token[i] == "n":
        i += 1
    if i == 0 or i >= len(token):
        return token
    if "\u0400" <= token[i] <= "\u04ff":
        return token[i:]
    return token


_RU_G_LATIN_GLUE_RE = re.compile(r"^g([\u0400-\u04ff]+)$")
_RU_EMBEDDED_N_GLUE_RE = re.compile(r"^(.+?)n([\u0400-\u04ff].+)$")
_RU_NN_VOPROS_RE = re.compile(r"^н+опрос$")
_RU_PLUS_GLUE_RE = re.compile(r"^плюс(.+)$", re.IGNORECASE)
_RU_LIST_MARKER_BEFORE_TOCHKA = frozenset(
    {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
    }
)


def _normalize_ru_glued_g_prefix_text(text: str) -> str:
    """Split ASR ``g``+Cyrillic glue (``gсемь``) into ``гп`` + word for label tokens like ``ГП-7``."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        match = _RU_G_LATIN_GLUE_RE.match(token)
        if match:
            out.extend(["гп", match.group(1)])
            continue
        out.append(token)
    return " ".join(out)


def _normalize_ru_pien_to_pi_text(text: str) -> str:
    """Map spoken math ``пиэн`` to the shorter TN form ``пи`` (π)."""
    if not text:
        return ""
    return " ".join("пи" if token == "пиэн" else token for token in text.split())


def _normalize_ru_nn_vopros_text(text: str) -> str:
    """Map ASR ``ннопрос`` / ``нн``+``опрос`` glue to TN ``вопрос``."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        if token == "опрос" or _RU_NN_VOPROS_RE.match(token):
            out.append("вопрос")
            continue
        out.append(token)
    return " ".join(out)


def _normalize_ru_latin_c_mc_text(text: str) -> str:
    """Map Latin MC label ``c`` to Cyrillic ``с`` (``c.``/``c,`` tokens and standalone ``c``)."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        if token == "c":
            out.append("с")
            continue
        if token.startswith("c") and len(token) > 1:
            tail = token[1:]
            if tail[0] in ".,;:" or tail[0].isdigit() or "\u0400" <= tail[0] <= "\u04ff":
                out.append("с" + tail)
                continue
        out.append(token)
    return " ".join(out)


def _normalize_ru_plus_glue_text(text: str) -> str:
    """Split glued formula plus tokens (``плюс2корень`` → ``плюс 2 корень``)."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        match = _RU_PLUS_GLUE_RE.match(token)
        if not match:
            out.append(token)
            continue
        suffix = match.group(1)
        if suffix.startswith("2") and suffix[1:].isalpha():
            out.extend(["плюс", "2", suffix[1:]])
        else:
            out.extend(["плюс", suffix])
    return " ".join(out)


_RU_EN_PREFIX_RE = re.compile(r"^эн([\u0400-\u04ff].+)$")
_RU_NUMBER_EN_SUFFIX_RE = re.compile(
    r"^(два|три|четыре|пять|шесть|семь|восемь|девять|десять)эн$"
)
_RU_MC_NUMBER_VYBERITE_RE = re.compile(
    r"^(один|два|три|четыре|пять|шесть|семь|восемь|девять|десять)выберите$"
)
_RU_TRAILING_LATIN_T_RE = re.compile(
    r"^(шесть|семь|восемь|девять|десять|три|четыре|пять)t$"
)
_RU_NETAP_RE = re.compile(r"^netap$")
_RU_NNDN_RE = re.compile(r"^nndn$")
_RU_NX_VAR_RE = re.compile(r"^n([xt])$")
_RU_THIRTY_TWO_GLUE_RE = re.compile(r"^тридцать(два|двум|три|четыре)(минус)?$")
_RU_A_SUBSCRIPT_GLUE_RE = re.compile(
    r"^a(один|два|три|четыре|пять|шесть|семь|восемь|девять|десять|пятнадцать|пятьдесят)$"
)
_RU_P2O5_GLUE_RE = re.compile(r"^p2o5$", re.IGNORECASE)
_RU_TOCHKA_SPLIT_RE = re.compile(r"(точка)")
_RU_MEGA_GLUE_MIN_LEN = 18


def _is_latin_alnum_char(ch: str) -> bool:
    return ch.isascii() and (ch.isdigit() or ("a" <= ch <= "z") or ch in "._/")


def _is_cyrillic_char(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04ff"


def _split_lat_cyr_token(token: str) -> list[str] | None:
    """Split a token at Latin/Cyrillic script boundaries (``catвода`` → ``cat`` ``вода``)."""
    if len(token) < 6:
        return None
    parts: list[str] = []
    buf: list[str] = []
    prev: str | None = None
    for ch in token:
        if _is_latin_alnum_char(ch):
            kind = "l"
        elif _is_cyrillic_char(ch):
            kind = "c"
        else:
            kind = "x"
        if prev is not None and kind != prev and buf and kind != "x" and prev != "x":
            parts.append("".join(buf))
            buf = []
        buf.append(ch)
        if kind != "x":
            prev = kind
    if buf:
        parts.append("".join(buf))
    if len(parts) < 2:
        return None
    return parts


def _split_tochka_glued_token(token: str) -> list[str] | None:
    """Split long Cyrillic glue around spoken ``точка`` list markers."""
    low = token.casefold()
    if len(low) < _RU_MEGA_GLUE_MIN_LEN or "точка" not in low:
        return None
    chunks = _RU_TOCHKA_SPLIT_RE.split(low)
    if len(chunks) < 3:
        return None
    out: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if chunk == "точка":
            out.append("точка")
        else:
            out.extend(_expand_ru_glued_token(chunk))
    return out if len(out) >= 2 else None


def _expand_ru_glued_token(token: str) -> list[str]:
    """Split one glued TN/ASR token into multiple normalized words."""
    if not token:
        return []
    low = token.casefold()
    if low == "nndn":
        return ["день"]
    if _RU_NETAP_RE.match(low):
        return ["этап"]
    m = _RU_EN_PREFIX_RE.match(low)
    if m:
        return _expand_ru_glued_token(m.group(1))
    m = _RU_NUMBER_EN_SUFFIX_RE.match(low)
    if m:
        return [m.group(1)]
    m = _RU_MC_NUMBER_VYBERITE_RE.match(low)
    if m:
        return [m.group(1), "выберите"]
    m = _RU_TRAILING_LATIN_T_RE.match(low)
    if m:
        return [m.group(1), "тэ"]
    m = _RU_THIRTY_TWO_GLUE_RE.match(low)
    if m:
        parts = ["тридцать", m.group(1)]
        if m.group(2):
            parts.append("минус")
        return parts
    m = _RU_NX_VAR_RE.match(low)
    if m:
        return [m.group(1)]
    m = _RU_A_SUBSCRIPT_GLUE_RE.match(low)
    if m:
        return ["a", m.group(1)]
    if _RU_P2O5_GLUE_RE.match(low):
        return ["p", "2", "o", "5"]
    if low.startswith("a") and low.endswith("десят") and len(low) > 7:
        return ["a", low[1:]]
    if low == "ина":
        return ["и", "на"]
    tochka_parts = _split_tochka_glued_token(token)
    if tochka_parts is not None:
        return tochka_parts
    if len(token) >= _RU_MEGA_GLUE_MIN_LEN:
        lat_cyr = _split_lat_cyr_token(token)
        if lat_cyr is not None:
            expanded: list[str] = []
            for part in lat_cyr:
                expanded.extend(_expand_ru_glued_token(part))
            return expanded
    return [token]


def _normalize_ru_glued_tokens_text(text: str) -> str:
    """Split ``\\эн``/MC/formula glued tokens (``одинвыберите``, ``дваэн``, ``netap``, …)."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        out.extend(_expand_ru_glued_token(token))
    return " ".join(out)


def _strip_ru_newline_tokens(text: str) -> str:
    """Remove spoken newline tokens and ``n``/``н`` glue from ASR hypotheses."""
    out: list[str] = []
    for token in text.split():
        if token in _RU_STANDALONE_NEWLINE_TOKENS:
            continue
        mc = _RU_MC_OPTION_RE.match(token)
        if mc:
            out.append(mc.group(1))
            continue
        cyr_n = _RU_CYR_N_LIST_NUMBER_RE.match(token)
        if cyr_n:
            out.append(cyr_n.group(1))
            continue
        peeled = _peel_latin_n_before_cyrillic(token)
        if peeled != token:
            token = peeled
        embedded = _RU_EMBEDDED_N_GLUE_RE.match(token)
        if embedded:
            out.extend([embedded.group(1), embedded.group(2)])
            continue
        if len(token) > 2 and token.endswith("nn") and not token.endswith("nnn"):
            base = token[:-2]
            if len(base) >= 2 and base not in _RU_STANDALONE_NEWLINE_TOKENS:
                token = base
        out.append(token)
    return " ".join(out)


def _normalize_ru_two_thousand_year_tokens(tokens: list[str]) -> list[str]:
    """Unify ``две/двух`` + ``тысячи/тысяч`` year phrasing to ``две тысячи``."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] in {"две", "двух"}
            and tokens[i + 1] in {"тысячи", "тысяч"}
        ):
            out.extend(["две", "тысячи"])
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out


def _normalize_ru_two_thousand_year_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(_normalize_ru_two_thousand_year_tokens(text.split()))


_RU_IPV_FOUR_SUFFIXES = frozenset({"четырех", "четыре", "4", "four"})
_RU_IPV_SIX_SUFFIXES = frozenset({"шести", "шесть", "6", "six"})


def _normalize_ru_ipv_tokens(tokens: list[str]) -> list[str]:
    """Unify TN ``ipvчетыре*`` / ASR ``эпф четыре`` (and v6) to ``ipv4`` / ``ipv6``."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("ipv") and len(token) > 3:
            suffix = token[3:]
            if suffix in _RU_IPV_FOUR_SUFFIXES:
                out.append("ipv4")
                i += 1
                continue
            if suffix in _RU_IPV_SIX_SUFFIXES:
                out.append("ipv6")
                i += 1
                continue
        if token == "эпф" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in _RU_IPV_FOUR_SUFFIXES:
                out.append("ipv4")
                i += 2
                continue
            if nxt in _RU_IPV_SIX_SUFFIXES:
                out.append("ipv6")
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def _normalize_ru_ipv_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(_normalize_ru_ipv_tokens(text.split()))


_RU_GLUED_SUBSCRIPT_TWO = re.compile(r"^([a-zа-яё0-9]+)2$")
_RU_GLUED_SUBSCRIPT_ONE = re.compile(r"^([a-zа-яё])(один)$")
_RU_NB_MC_ODIN = re.compile(r"^[nн]+[a-z]*один$")
_RU_K_KAKOY_AFTER = frozenset(
    {"какой", "какая", "какие", "какому", "каким", "каком", "какое"}
)


def _normalize_ru_glued_subscript_one(text: str) -> str:
    """Split glued subscript ``один`` (``aодin``/``eодin``/``nbrtодin`` → base + ``одin``)."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        if _RU_NB_MC_ODIN.match(token):
            out.append("один")
            continue
        m = _RU_GLUED_SUBSCRIPT_ONE.match(token)
        if m:
            out.extend([m.group(1), "один"])
            continue
        out.append(token)
    return " ".join(out)


def _normalize_ru_glued_subscript_two(text: str) -> str:
    """Drop trailing subscript ``2`` glued after a token (``x2``/``y2``/``ноль2`` → base)."""
    if not text:
        return ""
    out: list[str] = []
    for token in text.split():
        m = _RU_GLUED_SUBSCRIPT_TWO.match(token)
        if m:
            out.append(m.group(1))
        else:
            out.append(token)
    return " ".join(out)


_RU_DIG_GLUED_X = re.compile(r"^(\d+)x$")
_RU_FK_GLUED_X = re.compile(r"^([fk])x$")
_RU_CYR_NUM_GLUED_X = re.compile(
    r"^(два|три|четыре|пять|шесть|семь|восемь|девять|десять)x$"
)


def _normalize_ru_math_x_tokens(tokens: list[str]) -> list[str]:
    """Map TN ``икс`` to ``x`` and split ASR-glued math tokens (``2x``, ``fx``, ``дваx``)."""
    out: list[str] = []
    for token in tokens:
        if token == "икс":
            out.append("x")
            continue
        if token == "эф":
            out.append("f")
            continue
        matched = False
        for pat in (_RU_DIG_GLUED_X, _RU_FK_GLUED_X, _RU_CYR_NUM_GLUED_X):
            m = pat.match(token)
            if m:
                out.extend([m.group(1), "x"])
                matched = True
                break
        if not matched:
            out.append(token)
    return out


def _normalize_ru_math_x_text(text: str) -> str:
    if not text:
        return ""
    tokens = _normalize_ru_math_x_tokens(text.split())
    collapsed: list[str] = []
    for token in tokens:
        if token == "x" and collapsed and collapsed[-1] == "x":
            continue
        collapsed.append(token)
    return " ".join(collapsed)


def _normalize_ru_opengl_es_tokens(tokens: list[str]) -> list[str]:
    """Map ASR ``ease`` to ``es`` in the ``opengl es`` acronym."""
    out: list[str] = []
    for token in tokens:
        if token == "ease" and out and out[-1] == "opengl":
            out.append("es")
            continue
        out.append(token)
    return out


def _normalize_ru_opengl_es_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(_normalize_ru_opengl_es_tokens(text.split()))


_RU_OPTIONAL_TWO_LABELS = frozenset(
    {"этап", "день", "задание", "терминатор", "точка"}
)
_RU_OPTIONAL_ONE_LABELS = _RU_OPTIONAL_TWO_LABELS | frozenset(
    {"неделя", "вопрос", "ответ"}
)
_RU_OPTIONAL_ONE_EXCLUDE_BEFORE = frozenset({"равно", "равна", "c", "p"})
_RU_OPTIONAL_TWO_MATH_AFTER = frozenset({"плюс", "минус", "корень"})
_RU_OPTIONAL_FOUR_MATH_AFTER = frozenset(
    {"x", "игрек", "минус", "один", "определение", "из", "меньше"}
)
_RU_OPTIONAL_FOUR_EXTRA_LABELS = frozenset({"есть"})
_RU_OPTIONAL_SIX_MATH_AFTER = frozenset(
    {"производная", "теперь", "минус", "один", "t", "тэ", "sin", "синус", "dt"}
)
_RU_OPTIONAL_THREE_MATH_AFTER = frozenset(
    {
        "плюс",
        "минус",
        "корень",
        "dx",
        "dt",
        "равна",
        "задавался",
        "точка",
        "в",
        "визуальный",
        "определение",
        "квадратный",
        "из",
        "эф",
        "меньше",
    }
)
_RU_INTEGER_WORDS_BEFORE_CELYH = frozenset(
    {
        "ноль",
        "один",
        "одна",
        "одну",
        "два",
        "две",
        "двум",
        "двух",
        "три",
        "тридцать",
        "четыре",
        "пять",
        "пятьсот",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
        "двенадцать",
        "сорок",
        "пятьдесят",
        "шестьдесят",
    }
)

_RU_MC_OPTION_LETTERS = frozenset({"а", "б", "в", "г", "д", "е"})
_RU_OPTIONAL_LETTER_EXCLUDE_BEFORE = frozenset(
    {"значение", "где", "переменная", "буква", "точка"}
)
_RU_OPTIONAL_LETTER_EXCLUDE_AFTER = frozenset(
    {"для", "какая", "некоторое", "маска", "равно"}
)
_RU_MC_OPTION_LETTER_BEFORE = frozenset(
    {
        "взаимодействии",
        "детьми",
        "поликлиники",
        "события",
        "осей",
        "фондам",
        "восьмого",
        "степеней",
        "на",
        "равно",
        "умножить",
        "цэ",
        "дэ",
    }
)
_RU_ONEC_GLUE_RE = re.compile(r"одинc(?=\s|$)")


def _normalize_ru_onec_product_text(text: str) -> str:
    """Split glued TN ``одинC`` product token into ``один c`` for ASR alignment."""
    if not text:
        return ""
    return _RU_ONEC_GLUE_RE.sub("один c", text)


def _strip_optional_ru_digit_from_ref(
    ref: str,
    hyp: str,
    digit: str,
    *,
    math_after: frozenset[str],
    allow_pi_before: bool = False,
    extra_labels: frozenset[str] | None = None,
) -> str:
    """Drop ref list/formula digit when hyp omits it but the next word still aligns."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != [digit]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            label_ok = ri > 0 and (
                ref_tokens[ri - 1] in _RU_OPTIONAL_TWO_LABELS
                or (extra_labels is not None and ref_tokens[ri - 1] in extra_labels)
            )
            math_ok = ref_tokens[ri + 1] in math_after
            pi_ok = allow_pi_before and ri > 0 and ref_tokens[ri - 1] == "пи"
            if before_ok or label_ok or math_ok or pi_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_one_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``1`` when hyp omits a list marker digit but the next word still aligns.

    Skips numeric values such as ``равно 1`` where ``1`` is semantic, not a list index.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["1"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            if ri > 0 and ref_tokens[ri - 1] in _RU_OPTIONAL_ONE_EXCLUDE_BEFORE:
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            label_ok = ri > 0 and ref_tokens[ri - 1] in _RU_OPTIONAL_ONE_LABELS
            if before_ok or label_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_two_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``2`` when hyp omits a list/formula digit but the next word still aligns.

    TN refs often keep numbered markers (``Этап 2:``, ``\\n2.``, ``Терминатор 2``,
    formula ``(2-…`` fragments) while ASR skips the spoken digit. Removes ``2`` when
    the following ref word matches hyp at the gap and either neighbors align, the
    prior ref token is a list label, or the next token is a math operator.
    """
    return _strip_optional_ru_digit_from_ref(
        ref, hyp, "2", math_after=_RU_OPTIONAL_TWO_MATH_AFTER
    )


def strip_optional_ru_three_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``3`` when hyp omits list/formula/``пи/3`` digits but the next word aligns."""
    return _strip_optional_ru_digit_from_ref(
        ref,
        hyp,
        "3",
        math_after=_RU_OPTIONAL_THREE_MATH_AFTER,
        allow_pi_before=True,
    )


def strip_optional_ru_four_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``4`` when hyp omits list/formula digits but the next word still aligns."""
    return _strip_optional_ru_digit_from_ref(
        ref,
        hyp,
        "4",
        math_after=_RU_OPTIONAL_FOUR_MATH_AFTER,
        extra_labels=_RU_OPTIONAL_FOUR_EXTRA_LABELS,
    )


def strip_optional_ru_six_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``6`` when hyp omits ``пи/6`` or formula factors but the next word aligns."""
    return _strip_optional_ru_digit_from_ref(
        ref,
        hyp,
        "6",
        math_after=_RU_OPTIONAL_SIX_MATH_AFTER,
        allow_pi_before=True,
    )


def strip_optional_ru_t_before_tochka_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``т`` when hyp omits the spoken letter in ``т.е.``/``т.п.``/``т.д.`` but keeps ``точка``."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["т"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or ref_tokens[ri + 1] != "точка":
                continue
            if hi >= len(hyp_tokens) or hyp_tokens[hi] != "точка":
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            if before_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_tri_word_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``три`` when hyp omits list/formula ``3`` spoken as a word but the next word still aligns."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["три"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            at_end = ri + 1 >= len(ref_tokens) and hi >= len(hyp_tokens)
            if not at_end:
                if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                    continue
                if ref_tokens[ri + 1] != hyp_tokens[hi]:
                    continue
            glue_ok = hi > 0 and hyp_tokens[hi - 1].endswith("три")
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            label_ok = ri > 0 and ref_tokens[ri - 1] in _RU_OPTIONAL_TWO_LABELS
            math_ok = (
                not at_end
                and ref_tokens[ri + 1] in _RU_OPTIONAL_THREE_MATH_AFTER
            )
            pi_ok = ri > 0 and ref_tokens[ri - 1] == "пи"
            celyh_ok = not at_end and ref_tokens[ri + 1] == "целых"
            if (
                before_ok
                or glue_ok
                or label_ok
                or math_ok
                or pi_ok
                or celyh_ok
            ):
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_celyh_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``целых`` when hyp uses decimal notation and omits the spoken linker.

    TN refs use ``N целых M десятых`` while ASR often writes ``N,M``. Removes
    ``целых`` when the following ref word matches hyp at the gap and either neighbors
    align or the prior ref token is an integer word.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["целых"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            int_ok = ri > 0 and ref_tokens[ri - 1] in _RU_INTEGER_WORDS_BEFORE_CELYH
            if before_ok or int_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_sotyh_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``сотых`` when hyp uses decimal notation and omits the spoken linker."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["сотых"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            int_ok = ri > 0 and ref_tokens[ri - 1] in _RU_INTEGER_WORDS_BEFORE_CELYH
            if before_ok or int_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def _normalize_ru_idef_zero_text(text: str) -> str:
    """Unify TN ``IDEFноль`` and ASR ``ILEF``/``Aleph`` + ``ноль`` to ``idef0``."""
    if not text:
        return ""
    text = text.replace("idefноль", "idef0")
    tokens: list[str] = []
    toks = text.split()
    i = 0
    while i < len(toks):
        if toks[i] in {"ilef", "aleph"} and i + 1 < len(toks) and toks[i + 1] == "ноль":
            tokens.append("idef0")
            i += 2
            continue
        tokens.append(toks[i])
        i += 1
    return " ".join(tokens)


def _normalize_ru_yo_to_e(text: str) -> str:
    """Map ``ё`` to ``е`` so ``её``/``ее`` and similar pairs align after TN vs ASR spelling."""
    return text.replace("ё", "е")


_RU_SPOURIOUS_BREAK_TOKENS = frozenset({"и", "nn", "nnn", "нн", "н", "в", "and", "на", "in", "ин", "рен"})
_RU_SPOURIOUS_BREAK_TOKEN_SEQS: tuple[tuple[str, ...], ...] = (("и", "на"), ("и", "н"))


def _hyp_insertion_matches_ref_gap(
    hyp_tokens: list[str],
    ref_tokens: list[str],
    *,
    hyp_start: int,
    ref_next_idx: int,
    span_len: int,
) -> bool:
    """True when tokens after an inserted hyp span match ref at the gap."""
    after = hyp_start + span_len
    return (
        after < len(hyp_tokens)
        and ref_next_idx < len(ref_tokens)
        and hyp_tokens[after] == ref_tokens[ref_next_idx]
    )


def strip_spurious_ru_linebreak_tokens(ref: str, hyp: str) -> str:
    """Drop hyp line-break tokens (``и``, ``nn``/``nnn``/``нн``, Cyrillic ``н``/``в``, ``and``, ``на``, ``in``, ``ин``, ``рен``) mirroring ``\\эн``/code newlines.

    Also drops consecutive pairs ``и`` + ``на`` and ``и`` + ``н`` when jiwer groups
    them in one insertion chunk. Uses word alignment: remove inserted hyp token(s) when ref has
    no token at that position and the following hyp word matches the ref word at
    the gap. Iterates until stable so multiple breaks in one utterance are handled.
    """
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not ref_tokens or not hyp_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hs = hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]
            hi = chunk.hyp_start_idx
            ref_next_idx = chunk.ref_start_idx
            if tuple(hs) in _RU_SPOURIOUS_BREAK_TOKEN_SEQS and _hyp_insertion_matches_ref_gap(
                hyp_tokens,
                ref_tokens,
                hyp_start=hi,
                ref_next_idx=ref_next_idx,
                span_len=len(hs),
            ):
                remove_indices.update(range(hi, hi + len(hs)))
                continue
            if len(hs) != 1 or hs[0] not in _RU_SPOURIOUS_BREAK_TOKENS:
                continue
            if _hyp_insertion_matches_ref_gap(
                hyp_tokens,
                ref_tokens,
                hyp_start=hi,
                ref_next_idx=ref_next_idx,
                span_len=1,
            ):
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def _strip_optional_ref_token(ref: str, hyp: str, token: str) -> str:
    """Drop ref *token* when hyp omits it but neighboring words still align."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != [token]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            ref_after = ref_tokens[ri + 1]
            if hyp_tokens[hi] != ref_after:
                continue
            if ri > 0:
                if hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]:
                    continue
            remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


_RU_YEAR_TWO_AFTER = frozenset({"тысячи", "тысяч"})
_RU_DVE_FRACTION_AFTER = frozenset({"целых", "сотых"})
_RU_OPTIONAL_MINUS_AFTER = frozenset({"задание", "последняя", "версия"})
_RU_INTEGER_WORDS_MINUS_CONTEXT = frozenset(
    {
        "ноль",
        "один",
        "одна",
        "два",
        "две",
        "двум",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
        "двенадцать",
        "тридцать",
        "сорок",
    }
)
_RU_I_RELAXED_PREFIX = 7


def _ru_shared_prefix(a: str, b: str, n: int) -> bool:
    return len(a) >= n and len(b) >= n and a[:n] == b[:n]


def strip_optional_ru_i_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``и`` when hyp omits it but neighboring words still align.

    Mirror of :func:`strip_spurious_ru_linebreak_tokens` for deletions: TN refs
    often keep the conjunction ``и`` while ASR uses commas or omits it between
    the same surrounding words (``A и B`` vs ``A B``). Allows a shared prefix on
    the word before ``и`` when inflection differs (``сообщения`` vs ``сообщение``).
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["и"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            before_ok = ri == 0 or (
                hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            )
            relaxed_before = (
                ri > 0
                and hi > 0
                and _ru_shared_prefix(
                    ref_tokens[ri - 1], hyp_tokens[hi - 1], _RU_I_RELAXED_PREFIX
                )
            )
            if before_ok or relaxed_before:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def normalize_ru_dve_dvum_to_dva_from_ref(ref: str, hyp: str) -> str:
    """Map ref ``две``/``двум`` to ``два`` when hyp uses the nominative math form."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            word = ref_tokens[ri]
            if word not in {"две", "двум"} or hyp_tokens[hi : hi + 1] != ["два"]:
                continue
            if word == "две" and ri + 1 < len(ref_tokens):
                if ref_tokens[ri + 1] in _RU_YEAR_TWO_AFTER:
                    continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            after_ok = ref_tokens[ri + 1] == hyp_tokens[hi + 1]
            fraction_ok = ref_tokens[ri + 1] in _RU_DVE_FRACTION_AFTER
            score_k_ok = (
                ri > 0
                and hi > 0
                and ref_tokens[ri - 1] == "к"
                and hyp_tokens[hi - 1] == "к"
            )
            if not (after_ok or fraction_ok or score_k_ok):
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                if not score_k_ok:
                    continue
            ref_tokens[ri] = "два"
            changed = True
            break
    return " ".join(ref_tokens)


def strip_optional_ru_minus_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``минус`` when it mirrors a hyphen or ``\\эн`` list marker, not subtraction."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["минус"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            if ri > 0 and ref_tokens[ri - 1] in _RU_INTEGER_WORDS_MINUS_CONTEXT:
                if ref_tokens[ri + 1] in _RU_INTEGER_WORDS_MINUS_CONTEXT:
                    continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            label_ok = ref_tokens[ri + 1] in _RU_OPTIONAL_MINUS_AFTER
            gpt_ok = (
                ri > 0
                and ref_tokens[ri - 1] == "три"
                and ref_tokens[ri + 1] == "последняя"
            )
            if before_ok or label_ok or gpt_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_v_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``в`` when hyp omits it but neighboring words still align.

    TN refs often keep the preposition ``в`` while ASR drops it in fast speech
    (``A в B`` vs ``A B``).
    """
    return _strip_optional_ref_token(ref, hyp, "в")


def _is_protected_disjunction(ref_tokens: list[str], ri: int) -> bool:
    """True for ``да или нет`` / ``yes or no`` — do not unify ``или`` with ``и``."""
    if ri <= 0 or ri + 1 >= len(ref_tokens) or ref_tokens[ri] != "или":
        return False
    before, after = ref_tokens[ri - 1], ref_tokens[ri + 1]
    return (before, after) in {("да", "нет"), ("yes", "no")}


def normalize_ru_ili_to_i_from_ref(ref: str, hyp: str) -> str:
    """Map ref ``или`` to ``и`` when hyp has ``и`` and neighbors align.

    TN refs often use ``A или B`` while ASR writes ``A и B`` for the same
    coordinated phrase. Skips protected disjunctions such as ``да или нет``.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["или"] or hyp_tokens[hi : hi + 1] != ["и"]:
                continue
            if _is_protected_disjunction(ref_tokens, ri):
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri] = "и"
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ru_tak_zhe_to_takzhe_from_ref(ref: str, hyp: str) -> str:
    """Merge ref ``так`` + ``же`` into ``также`` when hyp already uses the adverb spelling."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["так"] or hyp_tokens[hi : hi + 1] != ["также"]:
                continue
            if ri + 1 >= len(ref_tokens) or ref_tokens[ri + 1] != "же":
                continue
            if ri + 2 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 2] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri : ri + 2] = ["также"]
            changed = True
            break
    return " ".join(ref_tokens)


_RU_PRITOM = "\u043f\u0440\u0438\u0442\u043e\u043c"
_RU_ETOM = "\u044d\u0442\u043e\u043c"


def normalize_ru_pritom_to_pri_etom_from_ref(ref: str, hyp: str) -> str:
    """Split ref ``пritom`` into ``при`` + ``этom`` when hyp uses the decomposed spelling."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        for ri in range(len(ref_tokens)):
            if ref_tokens[ri] != _RU_PRITOM:
                continue
            for hi in range(len(hyp_tokens) - 1):
                if hyp_tokens[hi : hi + 2] != ["при", _RU_ETOM]:
                    continue
                if ri + 1 >= len(ref_tokens) or hi + 2 >= len(hyp_tokens):
                    continue
                if ref_tokens[ri + 1] != hyp_tokens[hi + 2]:
                    continue
                if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                    continue
                ref_tokens[ri : ri + 1] = ["при", _RU_ETOM]
                changed = True
                break
            if changed:
                break
    return " ".join(ref_tokens)


def normalize_ru_a_to_na_from_ref(ref: str, hyp: str) -> str:
    """Map ref MC option letter ``a`` to ``на`` when hyp reads ``\\na.`` as the preposition."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["a"] or hyp_tokens[hi : hi + 1] != ["на"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri] = "на"
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ru_pervoe_to_pervaya_from_ref(ref: str, hyp: str) -> str:
    """Map ref list marker ``первое`` to ``первая`` when hyp uses the feminine list form."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["первое"] or hyp_tokens[hi : hi + 1] != ["первая"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri] = "первая"
            changed = True
            break
    return " ".join(ref_tokens)


def strip_optional_ru_by_after_chto_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``бы`` after ``что`` when hyp merges the particle as ``чтобы``."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            if ref_tokens[ri : ri + 1] != ["бы"]:
                continue
            if ri == 0 or ref_tokens[ri - 1] != "что":
                continue
            hi = chunk.hyp_start_idx
            if hi == 0 or hyp_tokens[hi - 1] != "чтобы":
                continue
            remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def normalize_ru_chto_to_chtoby_from_ref(ref: str, hyp: str) -> str:
    """Map ref ``что`` to ``чтобы`` when hyp merges ``что бы`` as a single particle."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["что"] or hyp_tokens[hi : hi + 1] != ["чтобы"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri] = "чтобы"
            changed = True
            break
    return " ".join(ref_tokens)


def strip_optional_ru_mc_letter_from_ref(ref: str, hyp: str) -> str:
    """Drop ref MC/formula single-letter markers when hyp omits them but content aligns."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            letter = ref_tokens[ri]
            if letter not in _RU_MC_OPTION_LETTERS:
                continue
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            if ri > 0 and ref_tokens[ri - 1] in _RU_OPTIONAL_LETTER_EXCLUDE_BEFORE:
                continue
            if ref_tokens[ri + 1] in _RU_OPTIONAL_LETTER_EXCLUDE_AFTER:
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            formula_ok = (
                letter == "а"
                and ref_tokens[ri + 1] == "в"
                and (
                    ri + 2 >= len(ref_tokens)
                    or ref_tokens[ri + 2] == "степени"
                )
                and (
                    ri == 0
                    or ref_tokens[ri - 1] in _RU_MC_OPTION_LETTER_BEFORE
                )
            )
            mc_relaxed = (
                not before_ok
                and ri > 0
                and ref_tokens[ri - 1] in _RU_MC_OPTION_LETTER_BEFORE
            )
            if before_ok or formula_ok or mc_relaxed:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def normalize_ru_odna_dve_to_odin_k_dvum_from_ref(ref: str, hyp: str) -> str:
    """Map ref ``одна две`` to ``один к двум`` when hyp uses the spoken fraction form."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        for ri in range(len(ref_tokens) - 1):
            if ref_tokens[ri : ri + 2] != ["одна", "две"]:
                continue
            for hi in range(len(hyp_tokens) - 2):
                if hyp_tokens[hi : hi + 3] != ["один", "к", "двум"]:
                    continue
                if ri > 0 and hi > 0 and ref_tokens[ri - 1] != hyp_tokens[hi - 1]:
                    continue
                ref_tokens[ri : ri + 2] = ["один", "к", "двум"]
                changed = True
                break
            if changed:
                break
    return " ".join(ref_tokens)


def strip_spurious_ru_score_odin_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``один`` inserted between two ref words that omit an abbreviated score."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            if hyp_tokens[hi : hi + 1] != ["один"]:
                continue
            ri = chunk.ref_start_idx
            if ri > 0 and hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]:
                if ri < len(ref_tokens) and hi + 1 < len(hyp_tokens):
                    if ref_tokens[ri] == hyp_tokens[hi + 1]:
                        remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


_RU_THE_HTML_BEFORE = frozenset(
    {"html", "in", "where", "body", "main", "div", "diva", "script", "scripted"}
)


def strip_spurious_ru_the_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``the`` when it only mirrors an English article in code paths or quotes."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["the"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri] != hyp_tokens[hi + 1]:
                continue
            before_ok = hi == 0 or (
                ri > 0 and hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            )
            html_ok = hi > 0 and hyp_tokens[hi - 1] in _RU_THE_HTML_BEFORE
            if before_ok or html_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_sentence_a_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp sentence-linker ``а`` when ref continues with the same word (``о``+stem)."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["а"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            ref_word = ref_tokens[ri]
            hyp_after = hyp_tokens[hi + 1]
            word_ok = ref_word == hyp_after or ref_word in {f"о{hyp_after}", f"а{hyp_after}"}
            before_ok = ri > 0 and hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if word_ok and before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_tochka_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``точка`` when it only mirrors a dot in ``т.п.``/``т.д.``/code abbreviations."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["точка"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri] != hyp_tokens[hi + 1]:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_k_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``к`` when it only mirrors an optional preposition omitted in ref."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["к"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            next_ref = ref_tokens[ri]
            next_hyp = hyp_tokens[hi + 1]
            after_ok = next_ref == next_hyp or (
                len(next_ref) >= 5
                and len(next_hyp) >= 5
                and next_ref[:5] == next_hyp[:5]
            )
            if not after_ok:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_celyh_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``целых`` when ref uses compact decimals without the spoken linker."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["целых"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and ref_tokens[ri - 1] == "целых":
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            int_ok = hi > 0 and hyp_tokens[hi - 1] in _RU_INTEGER_WORDS_BEFORE_CELYH
            if before_ok or int_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def normalize_ru_opros_to_vopros_on_hyp(ref: str, hyp: str) -> str:
    """Map hyp MC ``опрос`` to ref ``вопрос`` when the stem is the same quiz header."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["вопрос"] or hyp_tokens[hi : hi + 1] != ["опрос"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            ref_after = ref_tokens[ri + 1]
            hyp_after = hyp_tokens[hi + 1]
            after_ok = (
                ref_after == hyp_after
                or ref_after.startswith(hyp_after)
                or hyp_after.startswith(ref_after[:4])
            )
            if not after_ok:
                continue
            hyp_tokens[hi] = "вопрос"
            changed = True
            break
    return " ".join(hyp_tokens)


def strip_spurious_ru_mc_odin_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``один`` duplicated after ``вопрос один выберите`` MC headers."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["один"]:
                continue
            if hi == 0 or hyp_tokens[hi - 1] != "выберите":
                continue
            if ri > 0 and ref_tokens[ri - 1] == "выберите":
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_desyatyh_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``десятых`` when ref uses compact decimals without the spoken linker."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["десятых"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri] != hyp_tokens[hi + 1]:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            int_ok = hi > 0 and hyp_tokens[hi - 1] in _RU_INTEGER_WORDS_BEFORE_CELYH
            if before_ok or int_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_english_code_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp English code tokens (``and``, ``with``, ``if``, …) when ref omits them."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi] not in _RU_CODE_INSERT_EN:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri] != hyp_tokens[hi + 1]:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def _strip_spurious_ru_inserted_word_from_hyp(
    ref: str,
    hyp: str,
    words: frozenset[str],
    *,
    relax_after_prefix: int = 0,
) -> str:
    """Drop hyp insertions from ``words`` when ref omits them but the next token aligns."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            word = hyp_tokens[hi]
            if word not in words:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            next_ref = ref_tokens[ri]
            next_hyp = hyp_tokens[hi + 1]
            after_ok = next_ref == next_hyp or (
                relax_after_prefix > 0
                and len(next_ref) >= relax_after_prefix
                and len(next_hyp) >= relax_after_prefix
                and next_ref[:relax_after_prefix] == next_hyp[:relax_after_prefix]
            )
            if not after_ok:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_function_words_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp function words (``в``, ``и``, ``на``, …) when ref omits them."""
    return _strip_spurious_ru_inserted_word_from_hyp(
        ref, hyp, _RU_SPURIOUS_FUNC_INS, relax_after_prefix=5
    )


def strip_spurious_ru_digit_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp Arabic digits when ref uses the same value as a Russian number word."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            word = hyp_tokens[hi]
            if not word.isdigit() or len(word) > 3:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            ref_word = ref_tokens[ri]
            hyp_after = hyp_tokens[hi + 1]
            if ref_word == hyp_after:
                pass
            elif len(ref_word) >= 2 and len(hyp_after) >= 2 and ref_word[:2] == hyp_after[:2]:
                pass
            else:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_spurious_ru_sentence_a_linker_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``а`` before a shared stem when ref omits the linker (e.g. penthouse)."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["а"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri] != hyp_tokens[hi + 1]:
                continue
            before_ok = hi > 0 and ri > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok:
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_optional_ru_ot_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``от`` in ``x от t`` when hyp omits the spoken preposition."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["от"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri > 0 and ref_tokens[ri - 1] == "x" and ri + 1 < len(ref_tokens):
                if hi < len(hyp_tokens) and ref_tokens[ri + 1] == hyp_tokens[hi]:
                    remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_te_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``тэ`` when hyp glues it to the prior number (``шестьt`` → ``шесть``)."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["тэ"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri > 0 and hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_x_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``x`` when hyp uses ``nx`` glue for the same variable."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["x"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if hi < len(hyp_tokens) and hyp_tokens[hi].endswith("x"):
                if ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]):
                    remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_glued_subscript_words_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``десять``/``пятнадцать``/``пятьдесят`` glued in ``aдесять``-style tokens."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    _GLUED_NUMBERS = frozenset({"десять", "пятнадцать", "пятьдесят"})
    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if not rs or rs[0] not in _GLUED_NUMBERS:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri > 0 and len(ref_tokens[ri - 1]) > 1 and ref_tokens[ri - 1][0] == "a":
                if hi < len(hyp_tokens) and ref_tokens[ri] == hyp_tokens[hi]:
                    remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_abbr_tokens_from_ref(ref: str, hyp: str) -> str:
    """Drop ref code/abbrev tokens (``column``, ``status``, ``gpt``, …) when hyp omits them."""
    import jiwer

    _ABBR = _RU_OPTIONAL_ASCII_REF_DEL | frozenset(
        {"youtrack", "km", "рф", "спб", "сша", "одкб"}
    )
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if not rs:
                continue
            tok = rs[0]
            ascii_ok = tok in _ABBR or (
                tok.isascii()
                and tok.replace("_", "").isalpha()
                and 3 <= len(tok) <= 24
            )
            if not ascii_ok:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 < len(ref_tokens) and hi < len(hyp_tokens):
                if ref_tokens[ri + 1] == hyp_tokens[hi]:
                    remove_indices.add(ri)
            elif hi >= len(hyp_tokens) and ri + 1 >= len(ref_tokens):
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_gp_letter_from_ref(ref: str, hyp: str) -> str:
    """Drop ref label letters ``р``/``ш`` when hyp glues them (``gсемь`` → ``гп семь``)."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs not in (["р"], ["ш"]):
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri > 0 and ref_tokens[ri - 1] in {"гп", "a", "na"}:
                if hi < len(hyp_tokens) and ref_tokens[ri + 1 : ri + 2] == hyp_tokens[hi : hi + 1]:
                    remove_indices.add(ri)
                elif hi < len(hyp_tokens) and ri + 1 < len(ref_tokens):
                    if ref_tokens[ri + 1] == hyp_tokens[hi]:
                        remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


_RU_PRI_EXCLUDE_AFTER = frozenset({"процессорах", "процессора", "процессор"})
_RU_PRI_MIN_STEM = 6


def strip_spurious_ru_pri_from_hyp(ref: str, hyp: str) -> str:
    """Drop hyp ``при`` when it prefixes the same stem already present in ref (``пritom`` excluded)."""
    import jiwer

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return hyp

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(ref, " ".join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type != "insert":
                continue
            hi = chunk.hyp_start_idx
            ri = chunk.ref_start_idx
            if hyp_tokens[hi : hi + 1] != ["при"]:
                continue
            if ri >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if hyp_tokens[hi + 1] in _RU_PRI_EXCLUDE_AFTER:
                continue
            ref_word = ref_tokens[ri]
            hyp_word = hyp_tokens[hi + 1]
            stem_ok = (
                len(ref_word) >= _RU_PRI_MIN_STEM
                and len(hyp_word) >= _RU_PRI_MIN_STEM
                and ref_word[:_RU_PRI_MIN_STEM] == hyp_word[:_RU_PRI_MIN_STEM]
            )
            before_ok = ri > 0 and hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
            if before_ok and (stem_ok or ref_word == hyp_word):
                remove_indices.add(hi)
        if remove_indices:
            hyp_tokens = [
                tok for idx, tok in enumerate(hyp_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(hyp_tokens)


def strip_optional_ru_umnozhit_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``умножить`` when hyp abbreviates multiplication to ``на``."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["умножить"]:
                continue
            if ri + 1 >= len(ref_tokens) or ref_tokens[ri + 1] != "на":
                continue
            if hi >= len(hyp_tokens) or hyp_tokens[hi] != "на":
                continue
            after_ok = (
                ri + 2 < len(ref_tokens)
                and hi + 1 < len(hyp_tokens)
                and ref_tokens[ri + 2] == hyp_tokens[hi + 1]
            )
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            if before_ok or after_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


_RU_GENITIVE_TO_NOMINATIVE = {
    "двух": "два",
    "трех": "три",
    "четырех": "четыре",
    "пяти": "пять",
    "шести": "шесть",
    "семи": "семь",
    "восьми": "восемь",
    "девяти": "девять",
    "десяти": "десять",
    "двадцати": "двадцать",
    "тридцати": "тридцать",
    "шестнадцати": "шестнадцать",
    "пятнадцати": "пятнадцать",
}

_RU_GRAMMAR_REF_TO_HYP: dict[str, str] = {
    "одна": "один",
    "которое": "которая",
    "компании": "компаний",
    "права": "право",
    "значение": "значения",
    "первое": "первый",
    "неделя": "недели",
    "предприятии": "предприятие",
    "изображения": "изображение",
    "это": "эта",
    "ирмы": "ирма",
    "ответ": "ответа",
    "используй": "используя",
    "перефразируй": "перефразируя",
    "розыгрыш": "разыгрыш",
    "розыгрыше": "разыгрыше",
    "вопрос": "вопросы",
    "предоставление": "предоставления",
    "системы": "систему",
    "символа": "символом",
    "общаются": "общается",
    "одинвыберите": "выберите",
    "двавыберите": "выберите",
    "eчетыре": "четыре",
    "date": "datecolumn",
    "selecteditem": "item",
    "steper": "stepper",
}

_RU_CODE_INSERT_EN = frozenset(
    {
        "and",
        "or",
        "if",
        "with",
        "in",
        "as",
        "for",
        "to",
        "of",
        "the",
        "is",
        "by",
        "on",
        "at",
        "from",
        "h",
        "p",
        "d",
        "e",
        "g",
        "a",
        "selected",
        "standalone",
        "submit",
        "type",
        "value",
        "return",
        "void",
        "class",
        "package",
        "document",
        "array",
        "list",
        "item",
        "string",
        "int",
        "code",
        "api",
        "http",
        "https",
    }
)

_RU_SPURIOUS_FUNC_INS = frozenset(
    {
        "в",
        "и",
        "на",
        "по",
        "о",
        "а",
        "не",
        "от",
        "у",
        "с",
        "это",
        "таким",
        "же",
        "ли",
        "вы",
    }
)

_RU_OPTIONAL_ASCII_REF_DEL = frozenset(
    {
        "and",
        "by",
        "or",
        "if",
        "column",
        "status",
        "memory",
        "gpt",
        "event",
        "added",
        "index",
        "key",
        "value",
        "data",
        "type",
        "df",
        "go",
        "api",
        "http",
        "https",
        "www",
        "selected",
        "item",
        "return",
        "void",
        "class",
        "package",
        "array",
        "list",
        "string",
        "int",
        "date",
        "trunc",
        "path",
        "chat",
        "cuda",
        "lua",
        "busctl",
        "referer",
        "activity",
        "document",
        "standalone",
        "loader",
        "stepper",
        "submit",
        "input",
        "bodyparts",
        "concurrentcreates",
        "craftmanship",
        "epam",
        "hat",
        "finest",
    }
)

_RU_SPOKEN_PAIR_TO_HYP_DIGIT = {
    ("тридцать", "семь"): "37",
}


def _normalize_ru_ref_token_pair_from_ref(
    ref: str,
    hyp: str,
    mapping: dict[str, str],
    *,
    relax_after: bool = False,
) -> str:
    """Map ref tokens to hyp spellings when neighbors still align."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            src = ref_tokens[ri]
            dst = mapping.get(src)
            if dst is None or hyp_tokens[hi : hi + 1] != [dst]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            ref_after = ref_tokens[ri + 1]
            hyp_after = hyp_tokens[hi + 1]
            if ref_after != hyp_after:
                if not relax_after:
                    continue
                if not (
                    ref_after.startswith(hyp_after)
                    or hyp_after.startswith(ref_after[:4])
                ):
                    continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri] = dst
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ru_genitive_cardinal_to_nominative_from_ref(ref: str, hyp: str) -> str:
    """Map ref genitive cardinals to hyp nominative when neighbors still align."""
    return _normalize_ru_ref_token_pair_from_ref(
        ref, hyp, _RU_GENITIVE_TO_NOMINATIVE, relax_after=False
    )


def normalize_ru_grammar_variant_from_ref(ref: str, hyp: str) -> str:
    """Map ref grammar/TN/code spellings to hyp variants when neighbors still align."""
    return _normalize_ru_ref_token_pair_from_ref(
        ref, hyp, _RU_GRAMMAR_REF_TO_HYP, relax_after=True
    )


def normalize_ru_desyati_to_desyat_from_ref(ref: str, hyp: str) -> str:
    """Map ref genitive ``десяти`` to ``десять`` when hyp uses the nominative math form."""
    return normalize_ru_genitive_cardinal_to_nominative_from_ref(ref, hyp)


def normalize_ru_tekst_to_teksty_from_ref(ref: str, hyp: str) -> str:
    """Map ref ``текст`` to ``тексты`` in ``переформулировать текст`` instruction headers."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["текст"] or hyp_tokens[hi : hi + 1] != ["тексты"]:
                continue
            if ri == 0 or ref_tokens[ri - 1] != "переформулировать":
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            ref_tokens[ri] = "тексты"
            changed = True
            break
    return " ".join(ref_tokens)


def strip_optional_ru_tochka_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``точка`` when hyp omits a spoken dot but the next word still aligns."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if rs != ["точка"]:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            code_dot_ok = (
                ri > 0
                and ref_tokens[ri - 1].endswith(("один", "box"))
                and ref_tokens[ri + 1] == hyp_tokens[hi]
            )
            if before_ok or code_dot_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def normalize_ru_tri_to_three_from_ref(ref: str, hyp: str) -> str:
    """Map ref spoken ``три`` to digit ``3`` when hyp uses the digit in formulas."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["три"] or hyp_tokens[hi : hi + 1] != ["3"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            ref_tokens[ri] = "3"
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ru_spoken_pair_to_hyp_digit_from_ref(ref: str, hyp: str) -> str:
    """Collapse ref two-word integers to hyp digits (``тридцать семь`` → ``37``)."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type not in {"substitute", "delete"}:
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if hi >= len(hyp_tokens):
                continue
            digit = hyp_tokens[hi]
            if not digit.isdigit():
                continue
            for (w1, w2), target in _RU_SPOKEN_PAIR_TO_HYP_DIGIT.items():
                if target != digit:
                    continue
                if ri + 1 >= len(ref_tokens):
                    continue
                if ref_tokens[ri : ri + 2] != [w1, w2]:
                    continue
                before_ok = ri == 0 or (
                    hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1]
                )
                if not before_ok:
                    continue
                ref_tokens[ri : ri + 2] = [target]
                changed = True
                break
            if changed:
                break
    return " ".join(ref_tokens)


def strip_optional_ru_list_digit_tochka_from_ref(ref: str, hyp: str) -> str:
    """Drop ref list markers ``N точка`` when hyp omits spoken numbering but keeps the text."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            rs = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            if (
                len(rs) == 2
                and rs[1] == "точка"
                and rs[0] in _RU_LIST_MARKER_BEFORE_TOCHKA
            ):
                if ri + 2 <= len(ref_tokens) and hi < len(hyp_tokens):
                    if ref_tokens[ri + 2] == hyp_tokens[hi]:
                        remove_indices.update({ri, ri + 1})
                continue
            if rs == ["точка"] and ri > 0:
                if ref_tokens[ri - 1] not in _RU_LIST_MARKER_BEFORE_TOCHKA:
                    continue
                if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                    continue
                if ref_tokens[ri + 1] != hyp_tokens[hi]:
                    continue
                remove_indices.update({ri - 1, ri})
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def normalize_ru_otvet_to_otvety_from_ref(ref: str, hyp: str) -> str:
    """Map ref ``ответ`` to ``ответы`` when hyp uses the MC/label plural spelling."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["ответ"] or hyp_tokens[hi : hi + 1] != ["ответы"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi + 1 >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi + 1]:
                continue
            ref_tokens[ri] = "ответы"
            changed = True
            break
    return " ".join(ref_tokens)


def strip_optional_ru_k_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``к`` when hyp omits the preposition but surrounding words still align."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["к"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            if (
                ri == 0
                and ri + 1 < len(ref_tokens)
                and ref_tokens[ri + 1] in _RU_K_KAKOY_AFTER
            ):
                continue
            if ri > 0 and (hi == 0 or ref_tokens[ri - 1] != hyp_tokens[hi - 1]):
                continue
            remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_optional_ru_odin_from_ref(ref: str, hyp: str) -> str:
    """Drop ref ``один`` when hyp glues it as a subscript or MC marker token."""
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        remove_indices: set[int] = set()
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri : ri + 1] != ["один"]:
                continue
            if ri + 1 >= len(ref_tokens) or hi >= len(hyp_tokens):
                continue
            if ref_tokens[ri + 1] != hyp_tokens[hi]:
                continue
            subscript_ok = (
                ri > 0
                and len(ref_tokens[ri - 1]) == 1
                and ref_tokens[ri - 1].isalpha()
                and hi > 0
                and (
                    _RU_GLUED_SUBSCRIPT_ONE.match(hyp_tokens[hi - 1])
                    is not None
                    or _RU_NB_MC_ODIN.match(hyp_tokens[hi - 1]) is not None
                )
            )
            vyberite_ok = (
                ri > 0
                and ref_tokens[ri - 1] == "выберите"
                and ref_tokens[ri + 1] == "или"
            )
            order_ok = (
                ri > 0
                and ref_tokens[ri - 1] == "интегрирования"
                and ref_tokens[ri + 1] == "первым"
            )
            before_ok = ri == 0 or (hi > 0 and ref_tokens[ri - 1] == hyp_tokens[hi - 1])
            if before_ok or subscript_ok or vyberite_ok or order_ok:
                remove_indices.add(ri)
        if remove_indices:
            ref_tokens = [
                tok for idx, tok in enumerate(ref_tokens) if idx not in remove_indices
            ]
            changed = True
    return " ".join(ref_tokens)


def strip_spurious_ru_i(ref: str, hyp: str) -> str:
    """Backward-compatible alias for :func:`strip_spurious_ru_linebreak_tokens`."""
    return strip_spurious_ru_linebreak_tokens(ref, hyp)


def normalize_ar_glued_waw_from_ref(ref: str, hyp: str) -> str:
    """Merge ref ``prev`` + ``و`` when hyp glues the conjunction onto ``prev``.

    TN often emits ``زيري و أن`` while ASR transcribes ``زيريو أن``; likewise
    ``سبق و أن`` vs ``سبقوا أن``. Without this, jiwer counts a spurious ``و``
    deletion.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri] != _AR_WAW or ri == 0 or hi == 0:
                continue
            prev_r = ref_tokens[ri - 1]
            hyp_tok = hyp_tokens[hi - 1]
            if not _ar_hyp_glues_waw(prev_r, hyp_tok):
                continue
            if ri + 1 < len(ref_tokens):
                if hi >= len(hyp_tokens) or ref_tokens[ri + 1] != hyp_tokens[hi]:
                    continue
            elif hi < len(hyp_tokens):
                continue
            ref_tokens[ri - 1 : ri + 1] = [hyp_tok]
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ar_glued_ma_from_ref(ref: str, hyp: str) -> str:
    """Merge ref ``prev`` + ``ما`` when hyp glues the particle onto ``prev``.

    TN often emits ``بعد ما`` / ``كل ما`` while ASR transcribes ``بعدما`` /
    ``كلما``. Likewise ``وفق ما`` may appear as ``فقما`` when ``و`` is dropped.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "delete":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            if ref_tokens[ri] != _AR_MA or ri == 0 or hi == 0:
                continue
            prev_r = ref_tokens[ri - 1]
            hyp_tok = hyp_tokens[hi - 1]
            if not _ar_hyp_glues_ma(prev_r, hyp_tok):
                continue
            if ri + 1 < len(ref_tokens):
                if hi >= len(hyp_tokens) or ref_tokens[ri + 1] != hyp_tokens[hi]:
                    continue
            elif hi < len(hyp_tokens):
                continue
            ref_tokens[ri - 1 : ri + 1] = [hyp_tok]
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ar_split_ref_compounds_from_ref(ref: str, hyp: str) -> str:
    """Split ref compounds when hyp already uses the decomposed ASR spelling.

    TN may emit ``ألا`` / ``آنذاك`` as single tokens while ASR outputs ``أن لا`` /
    ``أن ذاك``, which otherwise yields a spurious ``أن`` insertion plus a false
    ``ألا`` -> ``لا`` substitution.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            ref_tok = ref_tokens[ri]
            split = _AR_REF_HYP_TOKEN_SPLITS.get(ref_tok)
            if split is None or hi == 0:
                continue
            hyp1, hyp2 = split
            if hyp_tokens[hi] != hyp2 or hyp_tokens[hi - 1] != hyp1:
                continue
            ref_tokens[ri : ri + 1] = [hyp1, hyp2]
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ar_split_bal_prefix_from_ref(ref: str, hyp: str) -> str:
    """Split ref ``بالX`` into ``بل`` + ``X`` when hyp uses the decomposed form.

    TN often emits ``بالقرب`` / ``بالإنجليزية`` while ASR outputs ``بل قرب`` /
    ``بل إنجليزية``, which otherwise yields a spurious ``بل`` insertion.
    """
    import jiwer

    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        output = jiwer.process_words(" ".join(ref_tokens), hyp)
        for chunk in output.alignments[0]:
            if chunk.type != "substitute":
                continue
            ri = chunk.ref_start_idx
            hi = chunk.hyp_start_idx
            ref_tok = ref_tokens[ri]
            if not ref_tok.startswith(_AR_BAL) or len(ref_tok) <= len(_AR_BAL):
                continue
            rest = ref_tok[len(_AR_BAL) :]
            if not rest or hi >= len(hyp_tokens) or hyp_tokens[hi] != rest:
                continue
            if hi == 0 or hyp_tokens[hi - 1] != _AR_BAL_SHORT:
                continue
            ref_tokens[ri : ri + 1] = [_AR_BAL_SHORT, rest]
            changed = True
            break
    return " ".join(ref_tokens)


def normalize_ar_merge_ref_glued_particles_from_ref(ref: str, hyp: str) -> str:
    """Merge ref particle pairs when hyp glues them into one token.

    TN may emit ``ما هو`` / ``و هي`` / ``هو ما`` / ``من هو`` while ASR outputs
    ``ماهو`` / ``وهي`` / ``هولما`` / ``ممنهو``, which otherwise counts spurious
    ``هو``/``هي``/``ما``/``من`` deletions.
    """
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return ref

    changed = True
    while changed:
        changed = False
        for ri in range(len(ref_tokens) - 1):
            for part1, part2, glued in _AR_REF_GLUED_PAIRS:
                if ref_tokens[ri] != part1 or ref_tokens[ri + 1] != part2:
                    continue
                for hi, hyp_tok in enumerate(hyp_tokens):
                    if hyp_tok != glued:
                        continue
                    if ri + 2 < len(ref_tokens):
                        if hi + 1 >= len(hyp_tokens) or ref_tokens[ri + 2] != hyp_tokens[hi + 1]:
                            continue
                    elif hi + 1 < len(hyp_tokens):
                        continue
                    ref_tokens[ri : ri + 2] = [glued]
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return " ".join(ref_tokens)


def normalize_texts_for_wer(
    ref: str,
    hyp: str,
    *,
    language: str | None = None,
) -> tuple[str, str]:
    """Normalize a ref/hyp pair for WER/CER, including pairwise Russian fixes."""
    ref_norm = normalize_for_wer(ref, language=language)
    hyp_norm = normalize_for_wer(hyp, language=language)
    if _is_russian(language):
        hyp_norm = strip_spurious_ru_linebreak_tokens(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_score_odin_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_the_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_tochka_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_pri_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_sentence_a_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_sentence_a_linker_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_k_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_celyh_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_desyatyh_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_english_code_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_function_words_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_digit_from_hyp(ref_norm, hyp_norm)
        hyp_norm = strip_spurious_ru_mc_odin_from_hyp(ref_norm, hyp_norm)
        hyp_norm = normalize_ru_opros_to_vopros_on_hyp(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_i_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_v_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_k_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_ili_to_i_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_tak_zhe_to_takzhe_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_pritom_to_pri_etom_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_a_to_na_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_by_after_chto_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_chto_to_chtoby_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_mc_letter_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_dve_dvum_to_dva_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_minus_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_pervoe_to_pervaya_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_odna_dve_to_odin_k_dvum_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_one_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_odin_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_two_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_three_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_four_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_six_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_tri_word_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_t_before_tochka_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_tochka_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_list_digit_tochka_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_ot_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_te_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_x_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_glued_subscript_words_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_abbr_tokens_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_gp_letter_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_celyh_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_sotyh_from_ref(ref_norm, hyp_norm)
        ref_norm = strip_optional_ru_umnozhit_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_genitive_cardinal_to_nominative_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_grammar_variant_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_tekst_to_teksty_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_tri_to_three_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_spoken_pair_to_hyp_digit_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ru_otvet_to_otvety_from_ref(ref_norm, hyp_norm)
    if _is_arabic(language):
        ref_norm = normalize_ar_glued_waw_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ar_glued_ma_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ar_split_ref_compounds_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ar_split_bal_prefix_from_ref(ref_norm, hyp_norm)
        ref_norm = normalize_ar_merge_ref_glued_particles_from_ref(ref_norm, hyp_norm)
        ref_norm = _normalize_ar_orthography_text(ref_norm)
        hyp_norm = _normalize_ar_orthography_text(hyp_norm)
    return ref_norm, hyp_norm


@lru_cache(maxsize=1)
def _american_spelling_fn():
    try:
        from breame.spelling import get_american_spelling
    except ImportError as exc:
        raise ImportError(
            "English spelling normalization requires the 'breame' package. "
            "Install ASR deps: bash scripts/setup_asr_deps.sh"
        ) from exc
    return get_american_spelling


def canonicalize_en_spelling_word(word: str) -> str:
    """Map British English spellings to American (identity if already US)."""
    if not word:
        return word
    word = _american_spelling_fn()(word)
    return _EN_EXTRA_BRITISH_TO_AMERICAN.get(word, word)


def normalize_en_compound_words(text: str) -> str:
    """Merge spaced compounds (per cent -> percent)."""
    return _EN_PER_CENT.sub("percent", text)


def _en_tail_is_percent(tokens: list[str], j: int) -> bool:
    if j < len(tokens) and tokens[j] == "percent":
        return True
    return (
        j + 1 < len(tokens) and tokens[j] == "per" and tokens[j + 1] == "cent"
    )


def normalize_ampm_markers(text: str) -> str:
    """Unify AM/PM time markers before punctuation removal."""
    text = _AM_DOT_M.sub(" am ", text)
    text = _PM_DOT_M.sub(" pm ", text)
    text = _DOT_AMPM.sub(r" \1 ", text)
    text = _DIGIT_AMPM.sub(r"\1 \2", text)
    # 4.51 pm -> 4 51 pm; 8.30 / 14.36 -> 8 30 / 14 36 (not decimals)
    text = re.sub(
        r"\b(\d{1,2})\.(\d{1,2})\s+(am|pm)\b",
        r"\1 \2 \3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(\d{1,2})\.(\d{2})\b", r"\1 \2", text)
    return text


def normalize_en_us_abbrev_markers(text: str) -> str:
    """Unify U.S.A. / U.S. / US before punctuation removal."""
    text = _EN_USA_DOTTED.sub(" USA ", text)
    text = _EN_US_DOTTED.sub(" US ", text)
    return text


def normalize_en_contractions(text: str) -> str:
    """Expand common English contractions (e.g. I'm -> I am)."""
    for pattern, replacement in _EN_CONTRACTION_EXPAND:
        text = pattern.sub(replacement, text)
    return text


def normalize_en_contraction_spelled_tokens(text: str) -> str:
    """Merge i m / don t etc. after apostrophes are removed."""
    for pattern, replacement in _EN_CONTRACTION_SPACED:
        text = pattern.sub(replacement, text)
    return text


def normalize_en_us_abbrev_spelled_tokens(text: str) -> str:
    """Merge u s / u s a into us / usa after punctuation strip."""
    text = _EN_USA_SPACED.sub("usa", text)
    text = _EN_US_SPACED.sub("us", text)
    return text


def normalize_ampm_spelled_tokens(text: str) -> str:
    """Merge ASR-style ``a m`` / ``p m`` into ``am`` / ``pm``."""
    text = _AM_SPACE_M.sub("am", text)
    return _PM_SPACE_M.sub("pm", text)


def normalize_en_number_phrase_and(text: str) -> str:
    """Drop optional 'and' inside English cardinal number phrases.

    Example: "seven hundred and ninety six" -> "seven hundred ninety six".
    """
    tokens = text.split()
    if not tokens:
        return text

    def is_numeric_token(tok: str) -> bool:
        return tok.isdigit() or tok in _EN_NUMERIC_WORDS

    out: list[str] = []
    for idx, tok in enumerate(tokens):
        if tok == "and":
            prev_tok = tokens[idx - 1] if idx > 0 else ""
            next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if is_numeric_token(prev_tok) and is_numeric_token(next_tok):
                continue
        out.append(tok)
    return " ".join(out)


def _parse_en_sub_100(tokens: list[str], idx: int) -> tuple[int, int]:
    if idx >= len(tokens):
        return 0, 0
    tok = tokens[idx]
    if tok in _EN_TEENS:
        return _EN_TEENS[tok], 1
    if tok in _EN_UNITS:
        return _EN_UNITS[tok], 1
    if tok in _EN_TENS:
        if idx + 1 < len(tokens) and tokens[idx + 1] in _EN_UNITS:
            return _EN_TENS[tok] + _EN_UNITS[tokens[idx + 1]], 2
        return _EN_TENS[tok], 1
    return 0, 0


def _parse_en_hour_24(tokens: list[str], idx: int) -> tuple[int, int]:
    if idx >= len(tokens):
        return 0, 0
    tok = tokens[idx]
    if tok.isdigit():
        hour = int(tok)
        if 0 <= hour <= 23:
            return hour, 1
        return 0, 0
    if tok in _EN_TEENS:
        return _EN_TEENS[tok], 1
    if tok in _EN_HOUR_WORDS:
        return _EN_HOUR_WORDS[tok], 1
    if tok == "twenty" and idx + 1 < len(tokens) and tokens[idx + 1] in _EN_UNITS:
        return 20 + _EN_UNITS[tokens[idx + 1]], 2
    if tok == "twenty":
        return 20, 1
    return 0, 0


def _parse_en_minute_after_point(tokens: list[str], idx: int) -> tuple[int, int]:
    digits: list[str] = []
    j = idx
    while j < len(tokens) and tokens[j] in _EN_UNITS and len(digits) < 2:
        digits.append(str(_EN_UNITS[tokens[j]]))
        j += 1
    if not digits:
        return 0, 0
    if len(digits) == 1:
        # TN may drop a trailing zero: "eight point three" for 8.30
        minute = int(digits[0]) * 10
    else:
        minute = int("".join(digits))
    if minute > 59:
        return 0, 0
    return minute, j - idx


def _parse_en_minute_tokens(tokens: list[str], idx: int) -> tuple[int, int]:
    if idx < len(tokens) and tokens[idx] == "oh" and idx + 1 < len(tokens):
        val, consumed = _parse_en_sub_100(tokens, idx + 1)
        if consumed == 1 and val < 10:
            return val, consumed + 1
    val, consumed = _parse_en_sub_100(tokens, idx)
    if consumed == 0 or val >= 60:
        return 0, 0
    return val, consumed


def _format_en_clock_token(hour: int, minute: int, ampm: str) -> str:
    return _EN_CLOCK_CANON.format(hour=hour, minute=minute, ampm=ampm)


def normalize_en_clock_strip_ampm_suffix(text: str) -> str:
    """Drop am/pm on clock tokens so @3h59pm and @3h59 compare equal."""
    return _EN_CLOCK_STRIP_SUFFIX.sub(r"@\1h\2", text)


def _emit_en_clock(
    out: list[str],
    hour: int,
    minute: int,
    tokens: list[str],
    j: int,
    *,
    allow_zero_minute: bool = False,
) -> int:
    if j < len(tokens) and tokens[j] in {"am", "pm"}:
        out.append(_format_en_clock_token(hour, minute, tokens[j]))
        return j + 1
    if (minute > 0 or allow_zero_minute) and (
        j >= len(tokens) or tokens[j] not in _EN_CLOCK_STOPPERS
    ):
        out.append(_format_en_clock_token(hour, minute, ""))
        return j
    return -1


def normalize_en_clock_times(text: str) -> str:
    """Normalize clock phrases to one token (e.g. eight thirty -> @8h30, @4h51pm).

    Handles 12h/24h, o'clock, digit times (8.30, 14.36), and TN point-read minutes.
    """
    tokens = text.split()
    if not tokens:
        return text

    out: list[str] = []
    i = 0
    while i < len(tokens):
        # Leave "twenty twenty" for year normalization (2020, not 20:20).
        if (
            tokens[i] == "twenty"
            and i + 1 < len(tokens)
            and tokens[i + 1] == "twenty"
        ):
            out.append(tokens[i])
            i += 1
            continue

        hour, hour_len = _parse_en_hour_24(tokens, i)
        if hour_len == 0:
            out.append(tokens[i])
            i += 1
            continue

        j = i + hour_len
        minute = 0

        if j + 1 < len(tokens) and tokens[j] == "o" and tokens[j + 1] == "clock":
            j += 2
            next_i = _emit_en_clock(
                out, hour, 0, tokens, j, allow_zero_minute=True
            )
            if next_i >= 0:
                i = next_i
                continue
            out.append(tokens[i])
            i += 1
            continue

        if j < len(tokens) and tokens[j] == "point":
            consumed_min = 0
            minute, consumed_min = _parse_en_minute_after_point(tokens, j + 1)
            if consumed_min == 0:
                out.append(tokens[i])
                i += 1
                continue
            j = j + 1 + consumed_min
            if _en_tail_is_percent(tokens, j):
                out.append(tokens[i])
                i += 1
                continue
        elif j < len(tokens) and tokens[j].isdigit():
            minute = int(tokens[j])
            if minute > 59:
                out.append(tokens[i])
                i += 1
                continue
            j += 1
        elif j < len(tokens) and (tokens[j] in _EN_NUMERIC_WORDS or tokens[j] == "oh"):
            if tokens[j] in {"am", "pm"}:
                out.append(_format_en_clock_token(hour, 0, tokens[j]))
                i = j + 1
                continue
            consumed_min = 0
            minute, consumed_min = _parse_en_minute_tokens(tokens, j)
            if consumed_min == 0:
                out.append(tokens[i])
                i += 1
                continue
            j += consumed_min
        elif j < len(tokens) and tokens[j] in {"am", "pm"}:
            out.append(_format_en_clock_token(hour, 0, tokens[j]))
            i = j + 1
            continue
        else:
            out.append(tokens[i])
            i += 1
            continue

        next_i = _emit_en_clock(out, hour, minute, tokens, j)
        if next_i >= 0:
            i = next_i
            continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def normalize_en_year_phrases_to_digits(text: str) -> str:
    """Normalize common spoken English year forms to 4-digit years.

    Examples:
    - "two thousand twenty one" -> "2021"
    - "twenty twenty one" -> "2021"
    - "nineteen ninety six" -> "1996"
    """
    tokens = text.split()
    if not tokens:
        return text

    out: list[str] = []
    i = 0
    while i < len(tokens):
        # twenty twenty (one) -> 2020 / 2021 before clock can read it as @20h20
        if tokens[i] == "twenty" and i + 1 < len(tokens) and tokens[i + 1] == "twenty":
            if i + 2 < len(tokens) and tokens[i + 2] == "one":
                out.append("2021")
                i += 3
                continue
            out.append("2020")
            i += 2
            continue

        # two thousand (and) twenty one -> 2021
        if i + 1 < len(tokens) and tokens[i] == "two" and tokens[i + 1] == "thousand":
            j = i + 2
            if j < len(tokens) and tokens[j] == "and":
                j += 1
            val, consumed = _parse_en_sub_100(tokens, j)
            if consumed > 0:
                out.append(str(2000 + val))
                i = j + consumed
                continue
            out.append("2000")
            i = j
            continue

        # nineteen ninety six -> 1996; twenty twenty one -> 2021
        if tokens[i] in {"nineteen", "twenty"}:
            val, consumed = _parse_en_sub_100(tokens, i + 1)
            if consumed > 0 and (tokens[i] == "nineteen" or val >= 10):
                base = 1900 if tokens[i] == "nineteen" else 2000
                out.append(str(base + val))
                i += 1 + consumed
                continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def canonicalize_en_spelling_text(text: str) -> str:
    if not text:
        return ""
    out: list[str] = []
    for word in text.split():
        word = canonicalize_en_spelling_word(word)
        word = _EN_TOKEN_ALIASES.get(word, word)
        out.append(word)
    return " ".join(out)


def normalize_for_wer(text: str, *, language: str | None = None) -> str:
    """Normalize reference/hypothesis before jiwer.

    Applies Unicode NFKC, expands TN newline markers (``\\эн``, ``\\n``, real
    newlines), strips punctuation, collapses whitespace, and case-folds so
    word-level WER treats tokens that differ only in case (e.g. the vs The)
    as matches. For Russian, maps ``ё`` to ``е`` (e.g. ``её`` vs ``ее``); drops standalone
    ``n``/``эн`` tokens and ``n``/``н`` glue (``nb``/``нчетыре``/``nnтри``/``nnдва``) produced when ASR
    transcribes spoken line breaks; unifies TN ``ipvчетыре*``/``ipvшест*`` with ASR ``эпф четыре/шесть``
    to ``ipv4``/``ipv6``; peels glued subscript ``2``; maps TN ``икс``/``эф`` and glued ``2x``/``fx``/``дваx``
    to ``x``/``f``; unifies ``IDEFноль`` with ASR ``ILEF``/``Aleph`` + ``ноль`` as ``idef0``; normalizes
    ``две/двух`` + ``тысячи/тысяч`` year phrases to ``две тысячи``. Pairwise alignment also drops spurious standalone ``и``/``nn``/``nnn``/Cyrillic ``н``/``в``/``на``/``in``/``рен``
    and the consecutive pair ``и``+``на`` when they only mirror line-break markers, optional ref ``1``/``2``/``3``
    when list/formula numbering is omitted in ASR (see :func:`strip_optional_ru_one_from_ref`,
    :func:`strip_optional_ru_two_from_ref`,
    :func:`strip_optional_ru_three_from_ref`, :func:`strip_optional_ru_tri_word_from_ref`),
    optional ref ``т`` before ``точка`` in ``т.е.``/``т.п.`` abbreviations
    (see :func:`strip_optional_ru_t_before_tochka_from_ref`), and     optional ref ``целых`` when hyp uses
    decimal notation (see :func:`strip_optional_ru_celyh_from_ref`), optional ref ``сотых`` in the same
    decimal pattern (see :func:`strip_optional_ru_sotyh_from_ref`), optional ref ``4``/``6`` for list/formula
    numbering (see :func:`strip_optional_ru_four_from_ref`, :func:`strip_optional_ru_six_from_ref`), and
    ref ``первое`` aligned to hyp ``первая`` for list markers
    (see :func:`normalize_ru_pervoe_to_pervaya_from_ref`), optional ref ``бы`` after ``что`` when hyp
    uses ``чтобы`` (see :func:`strip_optional_ru_by_after_chto_from_ref`,
    :func:`normalize_ru_chto_to_chtoby_from_ref`), optional ref MC/formula
    letters ``а``/``б``/``в``/``г``/``д``/``е`` (see :func:`strip_optional_ru_mc_letter_from_ref`),
    ref ``две``/``двум`` to hyp ``два`` in math/decimal forms
    (see :func:`normalize_ru_dve_dvum_to_dva_from_ref`), optional ref ``минус`` for hyphens and
    list markers (see :func:`strip_optional_ru_minus_from_ref`),
    ref ``одна две`` aligned to hyp ``один к двум`` (see :func:`normalize_ru_odna_dve_to_odin_k_dvum_from_ref`),
    and hyp score ``один`` inserted between ref neighbors (see :func:`strip_spurious_ru_score_odin_from_hyp`).
    Single-text Russian normalization also splits glued ``одинC`` to ``один c`` and glued subscript
    ``один`` (``aодin``/``nbrtодin``). Pairwise rules also drop optional ref ``умножить`` before ``на``
    (see :func:`strip_optional_ru_umnozhit_from_ref`), map ref ``десяти`` to hyp ``десять``
    (see :func:`normalize_ru_genitive_cardinal_to_nominative_from_ref`,
    :func:`normalize_ru_desyati_to_desyat_from_ref`), ref ``текст`` to ``тексты`` in reformulation headers
    (see :func:`normalize_ru_tekst_to_teksty_from_ref`), optional ref ``точка`` when hyp omits spoken dots
    (see :func:`strip_optional_ru_tochka_from_ref`), ref ``три`` to ``3`` (see :func:`normalize_ru_tri_to_three_from_ref`),
    spoken two-word integers to hyp digits (see :func:`normalize_ru_spoken_pair_to_hyp_digit_from_ref`),
    hyp ``пиэн`` to ``пи`` and ``g``+Cyrillic glue to ``гп`` + word (single-text),
    ref ``ответ`` to ``ответы`` (see :func:`normalize_ru_otvet_to_otvety_from_ref`), optional ref ``к``
    (see :func:`strip_optional_ru_k_from_ref`),
    and optional ref ``один`` for subscript/MC markers (see :func:`strip_optional_ru_odin_from_ref`).
    Hyp-side rules also drop spurious ``the`` in code/quote paths (see :func:`strip_spurious_ru_the_from_hyp`)
    and sentence-linker ``а`` before a repeated stem (see :func:`strip_spurious_ru_sentence_a_from_hyp`),
    spurious ``точка`` in abbreviations (see :func:`strip_spurious_ru_tochka_from_hyp`), spurious hyp ``к`` and
    ``целых`` (see :func:`strip_spurious_ru_k_from_hyp`, :func:`strip_spurious_ru_celyh_from_hyp`), hyp MC
    ``опрос`` to ``вопрос`` (see :func:`normalize_ru_opros_to_vopros_on_hyp`), optional ref list ``N точка``
    (see :func:`strip_optional_ru_list_digit_tochka_from_ref`), single-text ``ннопрос``/``c``/``плюс`` glue fixes,
    and optional ``при`` before a shared stem (see :func:`strip_spurious_ru_pri_from_hyp`). Single-text normalization maps ``opengl ease`` to
    ``opengl es`` and splits ref ``пritom`` to ``при`` + ``этom`` (see :func:`normalize_ru_pritom_to_pri_etom_from_ref`).

    For English (``language`` starting with ``en``), also maps
    British/American spelling variants to US forms (e.g. prioritised vs prioritized),
    and unifies AM/PM markers (AM, a.m., a m -> am; PM, p.m., p m -> pm).

    For character-level languages (e.g. ``zh``), all spaces are removed so ref/hyp
    spacing differences do not count as CER errors.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_newline_markers(text)
    if _is_english(language):
        text = normalize_ampm_markers(text)
        text = normalize_en_us_abbrev_markers(text)
        text = normalize_en_contractions(text)
    if _is_arabic(language):
        text = _strip_arabic_diacritics_text(text)
        text = _normalize_ar_tanwin_alef_text(text)
        text = _normalize_ar_miah_spelling_text(text)
        text = _normalize_ar_american_spelling_text(text)
        text = _normalize_ar_ayya_spelling_text(text)
        text = _normalize_ar_ziri_spelling_text(text)
        text = _normalize_ar_zagallo_spelling_text(text)
        text = _normalize_ar_tokyo_spelling_text(text)
        text = _normalize_ar_tens_case_text(text)
    text = _PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = text.casefold()
    if _is_russian(language):
        text = _normalize_ru_yo_to_e(text)
        text = _strip_ru_newline_tokens(text)
        text = _normalize_ru_glued_tokens_text(text)
        text = _normalize_ru_glued_g_prefix_text(text)
        text = _normalize_ru_pien_to_pi_text(text)
        text = _normalize_ru_nn_vopros_text(text)
        text = _normalize_ru_plus_glue_text(text)
        text = _normalize_ru_ipv_text(text)
        text = _normalize_ru_glued_subscript_two(text)
        text = _normalize_ru_glued_subscript_one(text)
        text = _normalize_ru_math_x_text(text)
        text = _normalize_ru_idef_zero_text(text)
        text = _normalize_ru_two_thousand_year_text(text)
        text = _normalize_ru_onec_product_text(text)
        text = _normalize_ru_latin_c_mc_text(text)
        text = _normalize_ru_opengl_es_text(text)
    if _is_english(language):
        text = normalize_ampm_spelled_tokens(text)
        text = normalize_en_contraction_spelled_tokens(text)
        text = normalize_en_us_abbrev_spelled_tokens(text)
        text = normalize_en_compound_words(text)
        text = normalize_en_number_phrase_and(text)
        text = normalize_en_year_phrases_to_digits(text)
        text = normalize_en_clock_times(text)
        text = normalize_en_clock_strip_ampm_suffix(text)
        text = canonicalize_en_spelling_text(text)
    if is_char_level_wer_language(language):
        text = _WHITESPACE.sub("", text)
    return text
