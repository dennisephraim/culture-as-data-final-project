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

# --- Group III: Temporality & Accountability ---

def run_group3_analysis():
    print("Initializing Group III Analysis...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    results = []
    
    paths = df['path'].tolist()
    metadata = df[['month_year', 'corpus']].to_dict('records')
    
    # Specific time adverbs
    specific_time = ['now', 'today', 'yesterday', 'tomorrow', 'soon', 'currently', 'recently']
    # Add months and days
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    specific_time.extend(months)
    specific_time.extend(days)
    
    vague_time = ['periodically', 'eventually', 'gradually', 'potentially', 'possibly', 'typically', 'generally', 'regularly']
    
    assessment_verbs = ['monitor', 'evaluate', 'assess', 'review', 'consider', 'examine', 'analyze', 'observe']
    
    modals = ['may', 'might', 'could', 'should', 'would', 'can', 'must', 'ought']

    print("Processing documents for Group III...")
    texts = [get_text(p) for p in paths]
    
    for doc, meta in tqdm(zip(nlp.pipe(texts, batch_size=20), metadata), total=len(texts)):
        # Task 14: Gerund/Verb Ratio
        gerunds = len([t for t in doc if t.pos_ == 'VERB' and t.tag_ == 'VBG'])
        active_verbs = len([t for t in doc if t.pos_ == 'VERB' and t.tag_ in ['VBD', 'VBP', 'VBZ']])
        gerund_ratio = gerunds / active_verbs if active_verbs > 0 else 0
        
        # Task 15: Specificity Gap
        spec_count = len([t for t in doc if t.lower_ in specific_time])
        vague_count = len([t for t in doc if t.lower_ in vague_time])
        spec_gap = spec_count / (vague_count + 1) # Avoid div by zero
        
        # Task 16: Process Framing
        process_count = len([t for t in doc if t.lemma_ in assessment_verbs])
        process_density = (process_count / len(doc)) * 10000 if len(doc) > 0 else 0
        
        # Task 17: Hedging & Modals
        modal_count = len([t for t in doc if t.lower_ in modals])
        total_verbs = len([t for t in doc if t.pos_ == 'VERB'])
        modal_ratio = modal_count / total_verbs if total_verbs > 0 else 0
        
        results.append({
            'month_year': meta['month_year'],
            'corpus': meta['corpus'],
            'gerund_ratio': gerund_ratio,
            'specificity_gap': spec_gap,
            'process_density': process_density,
            'modal_ratio': modal_ratio
        })
        
    analysis_df = pd.DataFrame(results)
    monthly_stats = analysis_df.groupby(['month_year', 'corpus']).mean().unstack()
    monthly_stats.to_csv(f"{OUTPUT_DIR}/group3_temporality.csv")
    # Plotting each metric
    metrics = [
        ('gerund_ratio', 'task14_gerund_ratio', 'Gerund Ratio (Blurred Temporality)'),
        ('specificity_gap', 'task15_specificity_gap', 'Specificity Gap (Time Markers)'),
        ('process_density', 'task16_process_density', 'Process Framing Density'),
        ('modal_ratio', 'task17_modal_ratio', 'Hedging/Modal Ratio')
    ]
    
    for metric_key, file_name, title in metrics:
        data = monthly_stats[metric_key]
        plot_with_smoothing(data, f"Group III: {title}", "Value", 
                            f"{OUTPUT_DIR}/{file_name}_plot.png", COLORS)

if __name__ == "__main__":
    run_group3_analysis()
