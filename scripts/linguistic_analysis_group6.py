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

# --- Group VI: Temporal Orientation ---

def run_group6_analysis():
    print("Initializing Group VI Analysis (Temporal Orientation)...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    # Task 26: Future-Oriented Language Cluster
    # Task 27: Present/Current-Oriented Language Cluster
    
    future_markers = [
        'will', 'shall', 'expect', 'expectation', 'anticipate', 'anticipation', 
        'project', 'projection', 'forecast', 'outlook', 'prospect', 'upcoming', 
        'ahead', 'intend', 'intent', 'plan', 'future', 'medium-term', 'long-term', 
        'guidance', 'forward'
    ]
    
    current_markers = [
        'currently', 'now', 'today', 'presently', 'moment', 'ongoing', 'latest', 
        'current', 'recent', 'situation', 'environment', 'context', 'remains', 
        'is', 'are', 'now', 'today'
    ]
    
    results = []
    inflation_temporal = []
    
    print("Processing documents for Group VI...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = get_text(row['path']).lower()
        
        # Task 26/27: General counts
        f_count = len(re.findall(r'\b(' + '|'.join(future_markers) + r')\b', text))
        c_count = len(re.findall(r'\b(' + '|'.join(current_markers) + r')\b', text))
        
        results.append({
            'month_year': row['month_year'],
            'corpus': row['corpus'],
            'future_count': f_count,
            'current_count': c_count,
            'word_count': row['word_count']
        })
        
        # Task 28: Inflation Temporal Delta
        # Find 'inflation' and check window for markers
        tokens = text.split()
        inf_future = 0
        inf_current = 0
        for i, word in enumerate(tokens):
            if 'inflation' in word:
                window = tokens[max(0, i-5):min(len(tokens), i+6)]
                window_str = ' '.join(window)
                if any(m in window_str for m in future_markers):
                    inf_future += 1
                if any(m in window_str for m in current_markers):
                    inf_current += 1
        
        inflation_temporal.append({
            'month_year': row['month_year'],
            'corpus': row['corpus'],
            'inf_future': inf_future,
            'inf_current': inf_current
        })
        
    analysis_df = pd.DataFrame(results)
    inf_df = pd.DataFrame(inflation_temporal)
    
    baseline = pd.read_csv(f"{OUTPUT_DIR}/task1_normalization_baseline.csv", index_col=0)
    
    # Task 26/27 Aggregate and Normalize
    f_agg = analysis_df.groupby(['month_year', 'corpus'])['future_count'].sum().unstack().fillna(0)
    c_agg = analysis_df.groupby(['month_year', 'corpus'])['current_count'].sum().unstack().fillna(0)
    
    # Task 28 Aggregate
    inf_f_agg = inf_df.groupby(['month_year', 'corpus'])['inf_future'].sum().unstack().fillna(0)
    inf_c_agg = inf_df.groupby(['month_year', 'corpus'])['inf_current'].sum().unstack().fillna(0)
    
    # Ensure indices are strings for alignment with baseline CSV
    f_agg.index = f_agg.index.astype(str)
    c_agg.index = c_agg.index.astype(str)
    inf_f_agg.index = inf_f_agg.index.astype(str)
    inf_c_agg.index = inf_c_agg.index.astype(str)
    baseline.index = baseline.index.astype(str)
    
    f_density = (f_agg / baseline) * 10000
    c_density = (c_agg / baseline) * 10000
    
    # Task 28: Temporal Delta Ratio (Future-to-Current for Inflation)
    # Higher = More focused on future inflation, Lower = More focused on current inflation
    inf_ratio = inf_f_agg / (inf_c_agg + 1) # Simple ratio of raw co-occurrences
    
    # Save CSVs
    f_density.to_csv(f"{OUTPUT_DIR}/task26_future_orientation_density.csv")
    c_density.to_csv(f"{OUTPUT_DIR}/task27_current_orientation_density.csv")
    inf_ratio.to_csv(f"{OUTPUT_DIR}/task28_temporal_inflation_delta.csv")
    
    # Plotting
    plot_with_smoothing(f_density, "Task 26: Future-Oriented Language Density", "Count per 10k Words", 
                        f"{OUTPUT_DIR}/task26_future_orientation_plot.png", COLORS)
    
    plot_with_smoothing(c_density, "Task 27: Current-Oriented Language Density", "Count per 10k Words", 
                        f"{OUTPUT_DIR}/task27_current_orientation_plot.png", COLORS)
    
    # Task 28 Special Plot: Extra Smoothing + Trendlines
    plt.figure(figsize=(10, 6))
    for corp in ['FED', 'MEDIA']:
        # Drop NaNs to ensure lines end where data ends
        data = inf_ratio[corp].dropna()
        if data.empty:
            continue
            
        idx = pd.to_datetime(data.index)
        
        # Raw faint
        plt.plot(idx, data, color=COLORS[corp], alpha=0.15, linewidth=1)
        # 6-month heavy smoothing
        plt.plot(idx, data.rolling(6, min_periods=1).mean(), color=COLORS[corp], linewidth=2.5, label=f"{corp} (6-mo MA)")
        
        # Polynomial Trendline (Degree 2)
        x = np.arange(len(data))
        y = data.values
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        plt.plot(idx, p(x), color=COLORS[corp], linestyle='--', linewidth=1.5, alpha=0.8, label=f"{corp} Trend")

    plt.title("Task 28: Temporal Inflation Delta (Heavily Smoothed with Trendlines)")
    plt.ylabel("Future-to-Current Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/task28_temporal_inflation_delta_plot.png")
    plt.close()

    print("Group VI Analysis Complete.")

if __name__ == "__main__":
    run_group6_analysis()
