import re
import numpy as np
import textstat
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_nlp = None
_vader = None


def _load_models():
    global _nlp, _vader
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()


def extract_spacy_features(text: str) -> Dict[str, float]:
    _KEYS = [
        "spacy_pos_noun_ratio", "spacy_pos_verb_ratio", "spacy_pos_adj_ratio",
        "spacy_pos_adv_ratio", "spacy_pos_pron_ratio", "spacy_stopword_ratio",
        "spacy_entity_count", "spacy_entity_density",
        "spacy_avg_token_length", "spacy_sentence_count",
    ]
    if not text or not text.strip():
        return {k: 0.0 for k in _KEYS}

    _load_models()
    doc = _nlp(text[:100_000])

    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0}
    stopword_count = 0
    token_lengths = []

    for token in doc:
        if token.is_space:
            continue
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
        if token.is_stop:
            stopword_count += 1
        token_lengths.append(len(token.text))

    n = len(token_lengths) or 1
    sents = len(list(doc.sents))
    ents = len(doc.ents)

    return {
        "spacy_pos_noun_ratio":   pos_counts["NOUN"] / n,
        "spacy_pos_verb_ratio":   pos_counts["VERB"] / n,
        "spacy_pos_adj_ratio":    pos_counts["ADJ"] / n,
        "spacy_pos_adv_ratio":    pos_counts["ADV"] / n,
        "spacy_pos_pron_ratio":   pos_counts["PRON"] / n,
        "spacy_stopword_ratio":   stopword_count / n,
        "spacy_entity_count":     float(ents),
        "spacy_entity_density":   ents / n,
        "spacy_avg_token_length": float(np.mean(token_lengths)),
        "spacy_sentence_count":   float(sents),
    }


def extract_textstat_features(text: str) -> Dict[str, float]:
    _KEYS = [
        "textstat_flesch_reading_ease", "textstat_flesch_kincaid_grade",
        "textstat_gunning_fog", "textstat_smog_index", "textstat_coleman_liau",
        "textstat_automated_readability", "textstat_dale_chall",
        "textstat_difficult_words", "textstat_linsear_write",
    ]
    if not text or not text.strip():
        return {k: 0.0 for k in _KEYS}
    try:
        return {
            "textstat_flesch_reading_ease":   textstat.flesch_reading_ease(text),
            "textstat_flesch_kincaid_grade":  textstat.flesch_kincaid_grade(text),
            "textstat_gunning_fog":           textstat.gunning_fog(text),
            "textstat_smog_index":            textstat.smog_index(text),
            "textstat_coleman_liau":          textstat.coleman_liau_index(text),
            "textstat_automated_readability": textstat.automated_readability_index(text),
            "textstat_dale_chall":            textstat.dale_chall_readability_score(text),
            "textstat_difficult_words":       float(textstat.difficult_words(text)),
            "textstat_linsear_write":         textstat.linsear_write_formula(text),
        }
    except Exception:
        return {k: 0.0 for k in _KEYS}


def extract_vader_features(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {"vader_positive": 0.0, "vader_negative": 0.0,
                "vader_neutral": 0.0, "vader_compound": 0.0}
    _load_models()
    s = _vader.polarity_scores(text)
    return {
        "vader_positive": s["pos"],
        "vader_negative": s["neg"],
        "vader_neutral":  s["neu"],
        "vader_compound": s["compound"],
    }


def extract_regex_features(text: str) -> Dict[str, float]:
    _KEYS = [
        "regex_exclamation_count", "regex_question_count", "regex_quote_count",
        "regex_uppercase_ratio", "regex_digit_ratio", "regex_punct_ratio",
        "regex_url_count", "regex_hashtag_count", "regex_mention_count",
        "regex_ellipsis_count", "regex_word_count", "regex_char_count",
        "regex_avg_word_length", "regex_lexical_diversity",
    ]
    if not text or not text.strip():
        return {k: 0.0 for k in _KEYS}

    chars = len(text)
    words = re.findall(r"\b\w+\b", text)
    wc = len(words) or 1

    return {
        "regex_exclamation_count": float(text.count("!")),
        "regex_question_count":    float(text.count("?")),
        "regex_quote_count":       float(len(re.findall(r'["\']', text))),
        "regex_uppercase_ratio":   sum(c.isupper() for c in text) / chars if chars else 0,
        "regex_digit_ratio":       sum(c.isdigit() for c in text) / chars if chars else 0,
        "regex_punct_ratio":       len(re.findall(r'[.,!?;:"\'\-]', text)) / chars if chars else 0,
        "regex_url_count":         float(len(re.findall(r"https?://\S+", text))),
        "regex_hashtag_count":     float(len(re.findall(r"#\w+", text))),
        "regex_mention_count":     float(len(re.findall(r"@\w+", text))),
        "regex_ellipsis_count":    float(len(re.findall(r"\.{3,}", text))),
        "regex_word_count":        float(len(words)),
        "regex_char_count":        float(chars),
        "regex_avg_word_length":   float(np.mean([len(w) for w in words])) if words else 0.0,
        "regex_lexical_diversity": len(set(w.lower() for w in words)) / wc,
    }


def extract_all_features(text: str, prefix: str = "") -> Dict[str, float]:
    features: Dict[str, float] = {}
    for fn in [
        extract_spacy_features,
        extract_textstat_features,
        extract_vader_features,
        extract_regex_features,
    ]:
        for k, v in fn(text).items():
            features[f"{prefix}{k}"] = v
    return features
