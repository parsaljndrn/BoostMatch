import os
import sys
import pandas as pd
import numpy as np
import spacy
import textstat
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

print("📦 Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("❌ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    sys.exit(1)

vader = SentimentIntensityAnalyzer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "cleaneddata_part21.csv")

if not os.path.exists(DATASET_PATH):
    print(f"❌ File not found: {DATASET_PATH}")
    print("\nAvailable CSV files:")
    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.csv'):
            print(f"  - {file}")
    sys.exit(1)

print(f"📂 Loading data from: {DATASET_PATH}")
data = pd.read_csv(DATASET_PATH)
print(f"✅ Data loaded! Shape: {data.shape}")
print(f"Columns: {list(data.columns)}\n")

required_cols = ['title', 'text', 'label']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print(f"Available columns: {list(data.columns)}")
    sys.exit(1)

data['title'] = data['title'].fillna("")
data['text'] = data['text'].fillna("")

print("🔍 Extracting stylometric features...")

def extract_spacy_features(text):
    if not text or len(text.strip()) == 0:
        return {
            'spacy_pos_noun_ratio': 0, 'spacy_pos_verb_ratio': 0,
            'spacy_pos_adj_ratio': 0, 'spacy_pos_adv_ratio': 0,
            'spacy_pos_pron_ratio': 0, 'spacy_stopword_ratio': 0,
            'spacy_entity_count': 0, 'spacy_entity_density': 0,
            'spacy_avg_token_length': 0, 'spacy_sentence_count': 0
        }
    
    doc = nlp(text[:1000000])
    
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
        'spacy_pos_noun_ratio': pos_counts['NOUN'] / token_count if token_count > 0 else 0,
        'spacy_pos_verb_ratio': pos_counts['VERB'] / token_count if token_count > 0 else 0,
        'spacy_pos_adj_ratio': pos_counts['ADJ'] / token_count if token_count > 0 else 0,
        'spacy_pos_adv_ratio': pos_counts['ADV'] / token_count if token_count > 0 else 0,
        'spacy_pos_pron_ratio': pos_counts['PRON'] / token_count if token_count > 0 else 0,
        'spacy_stopword_ratio': stopword_count / token_count if token_count > 0 else 0,
        'spacy_entity_count': entity_count,
        'spacy_entity_density': entity_count / token_count if token_count > 0 else 0,
        'spacy_avg_token_length': np.mean(token_lengths) if token_lengths else 0,
        'spacy_sentence_count': sent_count
    }

def extract_textstat_features(text):
    if not text or len(text.strip()) == 0:
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

def extract_vader_features(text):
    if not text or len(text.strip()) == 0:
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
    if not text or len(text.strip()) == 0:
        return {
            'regex_exclamation_count': 0, 'regex_question_count': 0,
            'regex_quote_count': 0, 'regex_uppercase_ratio': 0,
            'regex_digit_ratio': 0, 'regex_punct_ratio': 0,
            'regex_url_count': 0, 'regex_hashtag_count': 0,
            'regex_mention_count': 0, 'regex_ellipsis_count': 0,
            'regex_word_count': 0, 'regex_char_count': 0,
            'regex_avg_word_length': 0, 'regex_lexical_diversity': 0
        }
    
    char_count = len(text)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    punct_count = len(re.findall(r'[.,!?;:"\'-]', text))
    
    exclamation_count = len(re.findall(r'!', text))
    question_count = len(re.findall(r'\?', text))
    quote_count = len(re.findall(r'["\']', text))
    url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    hashtag_count = len(re.findall(r'#\w+', text))
    mention_count = len(re.findall(r'@\w+', text))
    ellipsis_count = len(re.findall(r'\.{3,}', text))
    
    unique_words = len(set(word.lower() for word in words))
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    return {
        'regex_exclamation_count': exclamation_count,
        'regex_question_count': question_count,
        'regex_quote_count': quote_count,
        'regex_uppercase_ratio': uppercase_count / char_count if char_count > 0 else 0,
        'regex_digit_ratio': digit_count / char_count if char_count > 0 else 0,
        'regex_punct_ratio': punct_count / char_count if char_count > 0 else 0,
        'regex_url_count': url_count,
        'regex_hashtag_count': hashtag_count,
        'regex_mention_count': mention_count,
        'regex_ellipsis_count': ellipsis_count,
        'regex_word_count': word_count,
        'regex_char_count': char_count,
        'regex_avg_word_length': avg_word_length,
        'regex_lexical_diversity': lexical_diversity
    }

def extract_all_features(text, prefix=''):
    spacy_feat = extract_spacy_features(text)
    textstat_feat = extract_textstat_features(text)
    vader_feat = extract_vader_features(text)
    regex_feat = extract_regex_features(text)
    
    all_features = {}
    for feat_dict in [spacy_feat, textstat_feat, vader_feat, regex_feat]:
        for key, value in feat_dict.items():
            all_features[f"{prefix}{key}"] = value
    
    return all_features

print("📊 Processing data...")
output_rows = []

for idx, row in data.iterrows():
    if idx % 100 == 0:
        print(f"  Processed {idx}/{len(data)} rows...")
    
    caption = str(row['title'])
    content = str(row['text'])
    label = row['label']
    
    caption_features = extract_all_features(caption, prefix='caption_')
    #content_features = extract_all_features(content, prefix='content_')
    
    output_row = {
        'Caption': caption,
        'Content': content,
        'Label': label
    }
    output_row.update(caption_features)
    #output_row.update(content_features)
    
    output_rows.append(output_row)

output_df = pd.DataFrame(output_rows)

print(f"\n✅ Feature extraction complete! Shape: {output_df.shape}")
print(f"Columns: {len(output_df.columns)}")

output_file = 'extracteddatastylo.csv'
output_df.to_csv(output_file, index=False)
print(f"\n✅ Saved dataset: {output_file}")

numeric_cols = output_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"   Total Features: {len(numeric_cols)}")
print(f"   Total Rows: {len(output_df)}")