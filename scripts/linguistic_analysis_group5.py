import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from collections import Counter
import spacy
from visualizer import plot_with_smoothing

# --- Configuration ---
INDEX_PATH = "data/corpus_index.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {'FED': '#000080', 'MEDIA': '#8B0000'}

# Load Index
df = pd.read_csv(INDEX_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month_year'] = df['date'].dt.to_period('M')

def get_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Group V: Discourse Perimeters (Foucauldian / Bankspeak Analysis) ---

def run_group5_analysis():
    print("Initializing Group V Analysis (Discourse Perimeters)...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    # Task 23: Price vs. Inflation Perimeter
    # Task 24: Management Cluster vs. Reality Cluster
    # Task 25: The Perimeter of "Expectations"
    
    target_words = ['price', 'prices', 'inflation', 'expectations']
    management_cluster = ['framework', 'strategy', 'approach', 'priorities', 'objectives', 'goals', 'report', 'papers', 'methodology', 'toolkit']
    reality_cluster = ['employment', 'income', 'wages', 'production', 'supply', 'demand', 'cost', 'workers', 'consumers']
    
    collocates = {
        'price_inflation': {'FED': {'price': Counter(), 'inflation': Counter()}, 'MEDIA': {'price': Counter(), 'inflation': Counter()}},
        'expectations': {'FED': Counter(), 'MEDIA': Counter()}
    }
    
    management_results = []
    
    print("Processing documents for Group V...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path']).lower()
        doc = nlp(text)
        
        # Management vs Reality counts
        m_count = len([t for t in doc if t.lemma_ in management_cluster])
        r_count = len([t for t in doc if t.lemma_ in reality_cluster])
        management_results.append({
            'month_year': row['month_year'],
            'corpus': row['corpus'],
            'management': m_count,
            'reality': r_count,
            'word_count': row['word_count']
        })
        
        # Collocates
        for i, token in enumerate(doc):
            # Task 23: Price vs Inflation
            if token.lemma_ in ['price', 'inflation']:
                key = 'price' if token.lemma_ == 'price' else 'inflation'
                start = max(0, i - 5)
                end = min(len(doc), i + 6)
                for t in doc[start:end]:
                    if t.i != i and (t.pos_ in ['NOUN', 'ADJ']):
                        collocates['price_inflation'][row['corpus']][key][t.lemma_] += 1
            
            # Task 25: Expectations
            if token.lemma_ == 'expectation':
                start = max(0, i - 5)
                end = min(len(doc), i + 6)
                for t in doc[start:end]:
                    if t.i != i and (t.pos_ in ['NOUN', 'ADJ']):
                        collocates['expectations'][row['corpus']][t.lemma_] += 1
                        
    # --- Task 23: Visualize Price vs Inflation Perimeters ---
    for corp in ['FED', 'MEDIA']:
        for target in ['price', 'inflation']:
            top = pd.DataFrame(collocates['price_inflation'][corp][target].most_common(20), columns=['word', 'count'])
            top.to_csv(f"{OUTPUT_DIR}/task23_{target}_perimeter_{corp}.csv", index=False)
            
    # --- Task 24: Management vs Reality Time Series ---
    m_df = pd.DataFrame(management_results)
    baseline = pd.read_csv(f"{OUTPUT_DIR}/task1_normalization_baseline.csv", index_col=0)
    
    m_agg = m_df.groupby(['month_year', 'corpus'])['management'].sum().unstack().fillna(0)
    r_agg = m_df.groupby(['month_year', 'corpus'])['reality'].sum().unstack().fillna(0)
    
    m_density = (m_agg / baseline) * 10000
    r_density = (r_agg / baseline) * 10000
    
    # Plot Ratio: Management / Reality
    ratio = m_density / r_density.replace(0, 1) # Avoid div by zero
    plot_with_smoothing(ratio, "Task 24: Management/Reality Ratio (Paperwork vs Concrete Reality)", "Ratio Index", f"{OUTPUT_DIR}/task24_management_ratio_plot.png", COLORS)
    
    # --- Task 25: Expectations Perimeter ---
    for corp in ['FED', 'MEDIA']:
        top = pd.DataFrame(collocates['expectations'][corp].most_common(20), columns=['word', 'count'])
        top.to_csv(f"{OUTPUT_DIR}/task25_expectations_perimeter_{corp}.csv", index=False)

    print("Group V Analysis Complete.")

if __name__ == "__main__":
    run_group5_analysis()
