import numpy as np
import spacy
import textstat
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict

# ------------------------------
# Lazy-loaded NLP models
# ------------------------------
_nlp = None
_vader = None

def load_nlp_models():
    global _nlp, _vader
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            ) from e
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()


# ------------------------------
# Feature extraction functions
# ------------------------------
def extract_spacy_features(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {f: 0.0 for f in [
            'spacy_pos_noun_ratio', 'spacy_pos_verb_ratio', 'spacy_pos_adj_ratio',
            'spacy_pos_adv_ratio', 'spacy_pos_pron_ratio', 'spacy_stopword_ratio',
            'spacy_entity_count', 'spacy_entity_density',
            'spacy_avg_token_length', 'spacy_sentence_count'
        ]}
    load_nlp_models()
    doc = _nlp(text[:100000])
    pos_counts = {'NOUN':0,'VERB':0,'ADJ':0,'ADV':0,'PRON':0}
    stopword_count = 0
    token_lengths = []

    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
        if token.is_stop:
            stopword_count += 1
        if not token.is_space:
            token_lengths.append(len(token.text))

    token_count = len([t for t in doc if not t.is_space])
    sent_count = len(list(doc.sents))
    entity_count = len(doc.ents)

    return {
        'spacy_pos_noun_ratio': pos_counts['NOUN']/token_count if token_count else 0,
        'spacy_pos_verb_ratio': pos_counts['VERB']/token_count if token_count else 0,
        'spacy_pos_adj_ratio': pos_counts['ADJ']/token_count if token_count else 0,
        'spacy_pos_adv_ratio': pos_counts['ADV']/token_count if token_count else 0,
        'spacy_pos_pron_ratio': pos_counts['PRON']/token_count if token_count else 0,
        'spacy_stopword_ratio': stopword_count/token_count if token_count else 0,
        'spacy_entity_count': entity_count,
        'spacy_entity_density': entity_count/token_count if token_count else 0,
        'spacy_avg_token_length': float(np.mean(token_lengths)) if token_lengths else 0,
        'spacy_sentence_count': sent_count
    }

def extract_textstat_features(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {k: 0.0 for k in [
            'textstat_flesch_reading_ease','textstat_flesch_kincaid_grade',
            'textstat_gunning_fog','textstat_smog_index','textstat_coleman_liau',
            'textstat_automated_readability','textstat_dale_chall',
            'textstat_difficult_words','textstat_linsear_write'
        ]}
    try:
        return {
            'textstat_flesch_reading_ease': textstat.flesch_reading_ease(text),
            'textstat_flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'textstat_gunning_fog': textstat.gunning_fog(text),
            'textstat_smog_index': textstat.smog_index(text),
            'textstat_coleman_liau': textstat.coleman_liau_index(text),
            'textstat_automated_readability': textstat.automated_readability_index(text),
            'textstat_dale_chall': textstat.dale_chall_readability_score(text),
            'textstat_difficult_words': textstat.difficult_words(text),
            'textstat_linsear_write': textstat.linsear_write_formula(text)
        }
    except:
        return {k:0.0 for k in [
            'textstat_flesch_reading_ease','textstat_flesch_kincaid_grade',
            'textstat_gunning_fog','textstat_smog_index','textstat_coleman_liau',
            'textstat_automated_readability','textstat_dale_chall',
            'textstat_difficult_words','textstat_linsear_write'
        ]}

def extract_vader_features(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {'vader_positive':0,'vader_negative':0,'vader_neutral':0,'vader_compound':0}
    load_nlp_models()
    sentiment = _vader.polarity_scores(text)
    return {
        'vader_positive': sentiment['pos'],
        'vader_negative': sentiment['neg'],
        'vader_neutral': sentiment['neu'],
        'vader_compound': sentiment['compound']
    }

def extract_regex_features(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {k:0.0 for k in [
            'regex_exclamation_count','regex_question_count','regex_quote_count',
            'regex_uppercase_ratio','regex_digit_ratio','regex_punct_ratio',
            'regex_url_count','regex_hashtag_count','regex_mention_count',
            'regex_ellipsis_count','regex_word_count','regex_char_count',
            'regex_avg_word_length','regex_lexical_diversity'
        ]}

    char_count = len(text)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    punct_count = len(re.findall(r'[.,!?;:"\'-]', text))
    unique_words = len(set(w.lower() for w in words))
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0

    return {
        'regex_exclamation_count': text.count('!'),
        'regex_question_count': text.count('?'),
        'regex_quote_count': len(re.findall(r'["\']', text)),
        'regex_uppercase_ratio': uppercase_count/char_count if char_count else 0,
        'regex_digit_ratio': digit_count/char_count if char_count else 0,
        'regex_punct_ratio': punct_count/char_count if char_count else 0,
        'regex_url_count': len(re.findall(r'http[s]?://\S+', text)),
        'regex_hashtag_count': len(re.findall(r'#\w+', text)),
        'regex_mention_count': len(re.findall(r'@\w+', text)),
        'regex_ellipsis_count': len(re.findall(r'\.{3,}', text)),
        'regex_word_count': word_count,
        'regex_char_count': char_count,
        'regex_avg_word_length': avg_word_length,
        'regex_lexical_diversity': unique_words/word_count if word_count else 0
    }

# ------------------------------
# MAIN API FUNCTION
# ------------------------------
def extract_all_features(text: str, prefix: str='') -> Dict[str,float]:
    """
    Combine all stylometric, readability, sentiment, and regex features.
    Returns a dict with optional prefix.
    """
    features = {}
    for func in [extract_spacy_features, extract_textstat_features, extract_vader_features, extract_regex_features]:
        result = func(text)
        for k,v in result.items():
            features[f"{prefix}{k}"] = v
    return features