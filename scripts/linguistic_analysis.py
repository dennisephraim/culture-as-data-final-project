import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from collections import Counter
import spacy

# --- Configuration ---
INDEX_PATH = "data/corpus_index.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "figure.figsize": (10, 6)
})
COLORS = {'FED': '#000080', 'MEDIA': '#8B0000'}

# Load Index
df = pd.read_csv(INDEX_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month_year'] = df['date'].dt.to_period('M')

def get_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Group I: Semantic Façades & Focus ---

def task_1_normalization_baseline():
    """Calculate word counts for every month-year slice per corpus."""
    baseline = df.groupby(['month_year', 'corpus'])['word_count'].sum().unstack().fillna(0)
    baseline.to_csv(f"{OUTPUT_DIR}/task1_normalization_baseline.csv")
    return baseline

def track_keyword_cluster(name, keywords, baseline):
    """Generic tracker for keyword clusters (Tasks 2 & 3)."""
    print(f"Tracking cluster: {name}...")
    results = []
    
    # Pre-compile regex
    regex = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path'])
        matches = len(regex.findall(text))
        results.append({
            'month_year': row['month_year'],
            'corpus': row['corpus'],
            'count': matches
        })
    
    counts_df = pd.DataFrame(results).groupby(['month_year', 'corpus'])['count'].sum().unstack().fillna(0)
    
    # Normalize per 10k words
    density = (counts_df / baseline) * 10000
    density.to_csv(f"{OUTPUT_DIR}/task_{name}_density.csv")
    
    # Plot
    plt.figure()
    for col in density.columns:
        density[col].plot(label=col, color=COLORS.get(col, None), linewidth=2)
    plt.title(f"Linguistic Density: {name.replace('_', ' ').title()}")
    plt.ylabel("Count per 10k Words")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/task_{name}_plot.png")
    plt.close()

def task_4_and_multiplier():
    """Identify 'Competence Stacking' (3+ abstract nouns joined by 'and')."""
    print("Tracking Task 4: Competence Stacking...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    # Abstract noun pattern: NOUN and NOUN and NOUN
    # We'll use a simplified regex approach first or spaCy for better accuracy
    # But regex is faster for large corpus.
    # Let's use a combined approach.
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path'])
        # Simple regex for Word and Word and Word
        # \b\w+\s+and\s+\w+\s+and\s+\w+\b
        matches = re.findall(r'\b\w+\s+and\s+\w+\s+and\s+\w+\b', text, re.IGNORECASE)
        results.append({
            'month_year': row['month_year'],
            'corpus': row['corpus'],
            'count': len(matches)
        })
    
    # ... aggregation and plotting ...
    counts_df = pd.DataFrame(results).groupby(['month_year', 'corpus'])['count'].sum().unstack().fillna(0)
    counts_df.to_csv(f"{OUTPUT_DIR}/task4_stacking_counts.csv")

if __name__ == "__main__":
    baseline = task_1_normalization_baseline()
    
    # Task 2: Institutional Control
    track_keyword_cluster("institutional_control", 
                          ["mandate", "framework", "toolkit", "transparency", "methodology"], 
                          baseline)
    
    # Task 3: Volatility Regime
    track_keyword_cluster("volatility_regime", 
                          ["surge", "shock", "unprecedented", "crisis", "spike", "panic"], 
                          baseline)
    
    # Task 4
    task_4_and_multiplier()
