import sys
import numpy as np
import spacy
import textstat
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ===================== LOAD NLP MODELS =====================
print("📦 Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("❌ spaCy model not found. Install with:")
    print("   python -m spacy download en_core_web_sm")
    sys.exit(1)

vader = SentimentIntensityAnalyzer()

# ===================== FEATURE FUNCTIONS =====================

def extract_spacy_features(text):
    if not text or not text.strip():
        return {
            'spacy_pos_noun_ratio': 0,
            'spacy_pos_verb_ratio': 0,
            'spacy_pos_adj_ratio': 0,
            'spacy_pos_adv_ratio': 0,
            'spacy_pos_pron_ratio': 0,
            'spacy_stopword_ratio': 0,
            'spacy_entity_count': 0,
            'spacy_entity_density': 0,
            'spacy_avg_token_length': 0,
            'spacy_sentence_count': 0
        }

    doc = nlp(text[:100000])

    pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0}
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
        'spacy_pos_noun_ratio': pos_counts['NOUN'] / token_count if token_count else 0,
        'spacy_pos_verb_ratio': pos_counts['VERB'] / token_count if token_count else 0,
        'spacy_pos_adj_ratio': pos_counts['ADJ'] / token_count if token_count else 0,
        'spacy_pos_adv_ratio': pos_counts['ADV'] / token_count if token_count else 0,
        'spacy_pos_pron_ratio': pos_counts['PRON'] / token_count if token_count else 0,
        'spacy_stopword_ratio': stopword_count / token_count if token_count else 0,
        'spacy_entity_count': entity_count,
        'spacy_entity_density': entity_count / token_count if token_count else 0,
        'spacy_avg_token_length': np.mean(token_lengths) if token_lengths else 0,
        'spacy_sentence_count': sent_count
    }

def extract_textstat_features(text):
    if not text or not text.strip():
        return {
            'textstat_flesch_reading_ease': 0,
            'textstat_flesch_kincaid_grade': 0,
            'textstat_gunning_fog': 0,
            'textstat_smog_index': 0,
            'textstat_coleman_liau': 0,
            'textstat_automated_readability': 0,
            'textstat_dale_chall': 0,
            'textstat_difficult_words': 0,
            'textstat_linsear_write': 0
        }

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
        return {k: 0 for k in [
            'textstat_flesch_reading_ease',
            'textstat_flesch_kincaid_grade',
            'textstat_gunning_fog',
            'textstat_smog_index',
            'textstat_coleman_liau',
            'textstat_automated_readability',
            'textstat_dale_chall',
            'textstat_difficult_words',
            'textstat_linsear_write'
        ]}

def extract_vader_features(text):
    if not text or not text.strip():
        return {
            'vader_positive': 0,
            'vader_negative': 0,
            'vader_neutral': 0,
            'vader_compound': 0
        }

    sentiment = vader.polarity_scores(text)
    return {
        'vader_positive': sentiment['pos'],
        'vader_negative': sentiment['neg'],
        'vader_neutral': sentiment['neu'],
        'vader_compound': sentiment['compound']
    }

def extract_regex_features(text):
    if not text or not text.strip():
        return {
            'regex_exclamation_count': 0,
            'regex_question_count': 0,
            'regex_quote_count': 0,
            'regex_uppercase_ratio': 0,
            'regex_digit_ratio': 0,
            'regex_punct_ratio': 0,
            'regex_url_count': 0,
            'regex_hashtag_count': 0,
            'regex_mention_count': 0,
            'regex_ellipsis_count': 0,
            'regex_word_count': 0,
            'regex_char_count': 0,
            'regex_avg_word_length': 0,
            'regex_lexical_diversity': 0
        }

    char_count = len(text)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)

    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    punct_count = len(re.findall(r'[.,!?;:"\'-]', text))

    unique_words = len(set(w.lower() for w in words))
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    return {
        'regex_exclamation_count': text.count('!'),
        'regex_question_count': text.count('?'),
        'regex_quote_count': len(re.findall(r'["\']', text)),
        'regex_uppercase_ratio': uppercase_count / char_count if char_count else 0,
        'regex_digit_ratio': digit_count / char_count if char_count else 0,
        'regex_punct_ratio': punct_count / char_count if char_count else 0,
        'regex_url_count': len(re.findall(r'http[s]?://\S+', text)),
        'regex_hashtag_count': len(re.findall(r'#\w+', text)),
        'regex_mention_count': len(re.findall(r'@\w+', text)),
        'regex_ellipsis_count': len(re.findall(r'\.{3,}', text)),
        'regex_word_count': word_count,
        'regex_char_count': char_count,
        'regex_avg_word_length': avg_word_length,
        'regex_lexical_diversity': unique_words / word_count if word_count else 0
    }

# ===================== MAIN API FUNCTION =====================

def extract_all_features(text, prefix=''):
    features = {}

    for func in [
        extract_spacy_features,
        extract_textstat_features,
        extract_vader_features,
        extract_regex_features
    ]:
        result = func(text)
        for k, v in result.items():
            features[f"{prefix}{k}"] = v

    return features
