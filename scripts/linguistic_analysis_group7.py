import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
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

# --- Group VII: The Politics of Attribution (Foucauldian Analysis) ---

def run_group7_analysis():
    print("Initializing Group VII Analysis (Attribution & Abstraction)...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    # Task 29: Attribution Clusters
    labor_cluster = ['wage', 'wages', 'earning', 'payroll', 'labor', 'worker', 'employment']
    supply_cluster = ['supply', 'chain', 'disruption', 'bottleneck', 'commodity', 'energy', 'oil', 'shipping']
    capital_cluster = ['profit', 'margin', 'markup', 'corporate', 'dividend', 'greed']
    
    # Task 30: Core vs Headline Clusters
    core_cluster = ['core', 'underlying', 'trimmed', 'mean', 'excluding', 'exclude', 'pce']
    headline_cluster = ['headline', 'total', 'food', 'energy', 'gas', 'gasoline', 'grocery', 'overall']
    
    results = []
    
    print("Processing documents for Group VII...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path']).lower()
        doc = nlp(text)
        
        # We look for 'inflation' and check its neighbors
        for i, token in enumerate(doc):
            if token.lemma_ == 'inflation':
                window = doc[max(0, i-10):min(len(doc), i+11)]
                window_lemmas = [t.lemma_ for t in window]
                
                # Attribution checks
                is_labor = any(w in labor_cluster for w in window_lemmas)
                is_supply = any(w in supply_cluster for w in window_lemmas)
                is_capital = any(w in capital_cluster for w in window_lemmas)
                
                # Abstraction checks
                is_core = any(w in core_cluster for w in window_lemmas)
                is_headline = any(w in headline_cluster for w in window_lemmas)
                
                results.append({
                    'month_year': row['month_year'],
                    'corpus': row['corpus'],
                    'labor': 1 if is_labor else 0,
                    'supply': 1 if is_supply else 0,
                    'capital': 1 if is_capital else 0,
                    'core': 1 if is_core else 0,
                    'headline': 1 if is_headline else 0
                })
                
    analysis_df = pd.DataFrame(results)
    baseline = pd.read_csv(f"{OUTPUT_DIR}/task1_normalization_baseline.csv", index_col=0)
    baseline.index = baseline.index.astype(str)
    
    # Task 29: Guilt Attribution Index (Labor vs. Supply vs. Capital)
    agg_29 = analysis_df.groupby(['month_year', 'corpus'])[['labor', 'supply', 'capital']].mean().unstack().fillna(0)
    agg_29.index = agg_29.index.astype(str)
    
    # We will generate three separate plots for clarity
    for attr_type, title in [('labor', 'Labor Focus (Wages/Workers)'), 
                             ('supply', 'Supply Focus (Chains/Bottlenecks)'), 
                             ('capital', 'Capital Focus (Profits/Margins)')]:
        
        plt.figure(figsize=(10, 6))
        for corp in ['FED', 'MEDIA']:
            baseline_slice = baseline[corp]
            valid_months = baseline_slice[baseline_slice > 0].index.astype(str)
            data = agg_29[attr_type][corp].loc[valid_months]
            
            if data.empty: continue
            idx = pd.to_datetime(data.index)
            
            # 6-month heavy smoothing
            plt.plot(idx, data.rolling(6, min_periods=1).mean(), color=COLORS[corp], linewidth=2.5, label=f"{corp} (6m MA)")
            
            # Polynomial Trendline
            x = np.arange(len(data))
            z = np.polyfit(x, data.values, 2)
            p = np.poly1d(z)
            plt.plot(idx, p(x), color=COLORS[corp], linestyle='--', linewidth=1.5, alpha=0.8, label=f"{corp} Trend")

        plt.title(f"Task 29: {title} in Inflation Discourse")
        plt.ylabel("% of Inflation Mentions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/task29_{attr_type}_attribution_plot.png")
        plt.close()
    
    # Task 30: Abstraction Gap (Core vs Headline)
    agg_30 = analysis_df.groupby(['month_year', 'corpus'])[['core', 'headline']].sum().unstack().fillna(0)
    agg_30.index = agg_30.index.astype(str)
    abs_ratio = agg_30['core'] / (agg_30['headline'] + 1)
    
    # Enhanced Plot for Task 30
    plt.figure(figsize=(10, 6))
    for corp in ['FED', 'MEDIA']:
        baseline_slice = baseline[corp]
        valid_months = baseline_slice[baseline_slice > 0].index.astype(str)
        data = abs_ratio[corp].loc[valid_months]
        
        if data.empty: continue
        idx = pd.to_datetime(data.index)
        
        # 6-month heavy smoothing ONLY
        plt.plot(idx, data.rolling(6, min_periods=1).mean(), color=COLORS[corp], linewidth=2.5, label=f"{corp} (6m MA)")
        
        # Trendline
        x = np.arange(len(data))
        z = np.polyfit(x, data.values, 2)
        p = np.poly1d(z)
        plt.plot(idx, p(x), color=COLORS[corp], linestyle='--', linewidth=1.5, alpha=0.8, label=f"{corp} Trend")

    plt.title("Task 30: The Abstraction Gap (Core vs. Headline Focus)")
    plt.ylabel("Core-to-Headline Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/task30_abstraction_gap_plot.png")
    plt.close()

    print("Group VII Analysis Complete.")

if __name__ == "__main__":
    run_group7_analysis()
