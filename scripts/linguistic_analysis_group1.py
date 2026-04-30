import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from visualizer import plot_with_smoothing

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
    print("Task 1: Normalization Baseline...")
    baseline = df.groupby(['month_year', 'corpus'])['word_count'].sum().unstack().fillna(0)
    baseline.to_csv(f"{OUTPUT_DIR}/task1_normalization_baseline.csv")
    return baseline

def track_keyword_cluster(name, keywords, baseline):
    """Generic tracker for keyword clusters (Tasks 2 & 3)."""
    print(f"Tracking cluster: {name}...")
    results = []
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
    density = (counts_df / baseline) * 10000
    csv_name = f"{name}_density.csv" if name.startswith('task') else f"task_{name}_density.csv"
    plot_name = f"{name}_plot.png" if name.startswith('task') else f"task_{name}_plot.png"
    
    density.to_csv(f"{OUTPUT_DIR}/{csv_name}")
    
    plot_with_smoothing(density, f"Linguistic Density: {name.replace('_', ' ').title()}", "Count per 10k Words", 
                        f"{OUTPUT_DIR}/{plot_name}", COLORS)

def task_4_and_multiplier(baseline):
    """Identify 'Competence Stacking' (3+ abstract nouns joined by 'and')."""
    print("Tracking Task 4: Competence Stacking...")
    results = []
    # Pattern: Word and Word and Word
    regex = re.compile(r'\b\w+\s+and\s+\w+\s+and\s+\w+\b', re.IGNORECASE)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path'])
        matches = len(regex.findall(text))
        results.append({
            'month_year': row['month_year'],
            'corpus': row['corpus'],
            'count': matches
        })
    
    counts_df = pd.DataFrame(results).groupby(['month_year', 'corpus'])['count'].sum().unstack().fillna(0)
    density = (counts_df / baseline) * 10000
    density.to_csv(f"{OUTPUT_DIR}/task4_stacking_density.csv")
    
    plot_with_smoothing(density, "Task 4: Competence Stacking Index", "Matches per 10k Words", 
                        f"{OUTPUT_DIR}/task4_stacking_plot.png", COLORS)

def task_5_narrative_lifecycle(baseline):
    """Track 'Narrative Mortality' of specific terms."""
    print("Tracking Task 5: Narrative Life-Cycle...")
    terms = ['transitory', 'soft landing']
    
    for term in terms:
        regex = re.compile(r'\b' + term + r'\b', re.IGNORECASE)
        results = []
        for _, row in df.iterrows():
            text = get_text(row['path'])
            matches = len(regex.findall(text))
            results.append({
                'month_year': row['month_year'],
                'corpus': row['corpus'],
                'count': matches
            })
        
        counts_df = pd.DataFrame(results).groupby(['month_year', 'corpus'])['count'].sum().unstack().fillna(0)
        density = (counts_df / baseline) * 10000
        density.to_csv(f"{OUTPUT_DIR}/task5_{term.replace(' ', '_')}_density.csv")
        
        plot_with_smoothing(density, f"Narrative Lifecycle: '{term.title()}'", "Count per 10k Words", 
                            f"{OUTPUT_DIR}/task5_{term.replace(' ', '_')}_plot.png", COLORS)

def task_6_collocate_sentiment():
    """Map the 'Environment' of Inflation (±5 words)."""
    print("Tracking Task 6: Inflation Collocates...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat', 'parser'])
    
    collocates = {'FED': Counter(), 'MEDIA': Counter()}
    
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
    """Find 'Defining Tokens' per quarter."""
    print("Tracking Task 7: Quarterly TF-IDF...")
    df['quarter'] = df['date'].dt.to_period('Q')
    
    for corp in ['FED', 'MEDIA']:
        print(f"Processing TF-IDF for {corp}...")
        corp_df = df[df['corpus'] == corp]
        
        # Filter for quarters with at least one document
        quarterly_text = []
        quarters = sorted(corp_df['quarter'].unique())
        for q in quarters:
            q_paths = corp_df[corp_df['quarter'] == q]['path']
            combined_text = " ".join([get_text(p) for p in q_paths])
            quarterly_text.append(combined_text)
            
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(quarterly_text)
        feature_names = vectorizer.get_feature_names_out()
        
        results = []
        for i, quarter in enumerate(quarters):
            row = tfidf_matrix.getrow(i).toarray()[0]
            top_indices = row.argsort()[-20:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            results.append({'quarter': quarter, 'top_words': ", ".join(top_words)})
            
        pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/task7_tfidf_{corp}.csv", index=False)

if __name__ == "__main__":
    baseline = task_1_normalization_baseline()
    
    track_keyword_cluster("task2_institutional_control", ["mandate", "framework", "toolkit", "transparency", "methodology"], baseline)
    track_keyword_cluster("task3_volatility_regime", ["surge", "shock", "unprecedented", "crisis", "spike", "panic"], baseline)
    
    task_4_and_multiplier(baseline)
    task_5_narrative_lifecycle(baseline)
    task_6_collocate_sentiment()
    task_7_quarterly_tfidf()
    
    print("\nGroup I Analysis Complete. Results saved in 'results/' folder.")
