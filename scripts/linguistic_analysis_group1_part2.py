import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
INDEX_PATH = "data/corpus_index.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INDEX_PATH)
df['date'] = pd.to_datetime(df['date'])

def get_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def task_6_collocate_sentiment():
    print("Tracking Task 6: Inflation Collocates...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat', 'parser'])
    collocates = {'FED': Counter(), 'MEDIA': Counter()}
    
    # Sample if needed, but let's try full
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path']).lower()
        if 'inflation' not in text:
            continue
        doc = nlp(text)
        for i, token in enumerate(doc):
            if token.text == 'inflation':
                start = max(0, i - 5)
                end = min(len(doc), i + 6)
                window = doc[start:end]
                for t in window:
                    if t.pos_ == 'ADJ':
                        collocates[row['corpus']][t.lemma_] += 1
                        
    for corp, counts in collocates.items():
        top = pd.DataFrame(counts.most_common(20), columns=['adjective', 'count'])
        top.to_csv(f"{OUTPUT_DIR}/task6_inflation_collocates_{corp}.csv", index=False)

def task_7_quarterly_tfidf():
    print("Tracking Task 7: Quarterly TF-IDF...")
    df['quarter'] = df['date'].dt.to_period('Q')
    for corp in ['FED', 'MEDIA']:
        corp_df = df[df['corpus'] == corp]
        quarters = sorted(corp_df['quarter'].unique())
        quarterly_text = []
        valid_quarters = []
        for q in quarters:
            q_paths = corp_df[corp_df['quarter'] == q]['path']
            if len(q_paths) > 0:
                combined_text = " ".join([get_text(p) for p in q_paths])
                quarterly_text.append(combined_text)
                valid_quarters.append(q)
            
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(quarterly_text)
        feature_names = vectorizer.get_feature_names_out()
        
        results = []
        for i, quarter in enumerate(valid_quarters):
            row = tfidf_matrix.getrow(i).toarray()[0]
            top_indices = row.argsort()[-20:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            results.append({'quarter': quarter, 'top_words': ", ".join(top_words)})
        pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/task7_tfidf_{corp}.csv", index=False)

if __name__ == "__main__":
    task_6_collocate_sentiment()
    task_7_quarterly_tfidf()
