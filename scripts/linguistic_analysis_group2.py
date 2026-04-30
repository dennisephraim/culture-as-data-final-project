import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import spacy
import syllapy
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

# --- Group II: Structural Abstraction (Bankspeak Techniques) ---

def run_group2_analysis():
    print("Initializing Group II Analysis...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    
    results = []
    
    # Process in batches for speed
    paths = df['path'].tolist()
    metadata = df[['month_year', 'corpus']].to_dict('records')
    
    print("Processing documents...")
    # Read texts
    texts = [get_text(p) for p in paths]
    
    for doc, meta in tqdm(zip(nlp.pipe(texts, batch_size=20, n_process=1), metadata), total=len(texts)):
        text = doc.text
        
        # Task 8: Nominalization
        nominalizations = [t for t in doc if t.pos_ == 'NOUN' and re.search(r'(tion|sion|ment|ity|ance)$', t.text, re.I)]
        nom_ratio = len(nominalizations) / len(doc) if len(doc) > 0 else 0
        
        # Task 9: Noun-on-Noun Density
        noun_sequences = 0
        for i in range(len(doc)-1):
            if doc[i].pos_ == 'NOUN' and doc[i+1].pos_ == 'NOUN':
                noun_sequences += 1
        noun_noun_density = noun_sequences / len(doc) if len(doc) > 0 else 0
        
        # Task 10: Verb-to-Noun Ratio
        nouns = len([t for t in doc if t.pos_ == 'NOUN'])
        verbs = len([t for t in doc if t.pos_ == 'VERB'])
        v_n_ratio = nouns / verbs if verbs > 0 else 0
        
        # Task 11: Subjectivity Index
        first_person = len([t for t in doc if t.lower_ in ['i', 'we', 'me', 'us', 'my', 'our']])
        impersonal = len(re.findall(r'\b(Committee|Board|System|Fed|FOMC|Federal Reserve)\b', text, re.I))
        subjectivity_index = first_person / impersonal if impersonal > 0 else 0
        
        # Task 12: Passive Voice Density
        passive_count = len([t for t in doc if t.dep_ == 'nsubjpass'])
        passive_density = passive_count / len(doc) if len(doc) > 0 else 0
        
        # Task 13: Sentence Complexity
        num_sentences = len(list(doc.sents))
        avg_sent_length = len(doc) / num_sentences if num_sentences > 0 else 0
        syllables = sum([syllapy.count(t.text) for t in doc if t.is_alpha])
        syll_per_word = syllables / len(doc) if len(doc) > 0 else 0
        
        results.append({
            'month_year': meta['month_year'],
            'corpus': meta['corpus'],
            'nominalization_ratio': nom_ratio,
            'noun_noun_density': noun_noun_density,
            'verb_noun_ratio': v_n_ratio,
            'subjectivity_index': subjectivity_index,
            'passive_density': passive_density,
            'avg_sent_length': avg_sent_length,
            'syll_per_word': syll_per_word
        })
        
    analysis_df = pd.DataFrame(results)
    # Aggregate by month and corpus
    monthly_stats = analysis_df.groupby(['month_year', 'corpus']).mean().unstack()
    
    # Save results
    monthly_stats.to_csv(f"{OUTPUT_DIR}/group2_structural_abstraction.csv")
    
    # Plotting each metric
    metrics = [
        ('nominalization_ratio', 'task8_nominalization_ratio', 'Nominalization Ratio'),
        ('noun_noun_density', 'task9_noun_noun_density', 'Noun-on-Noun Density'),
        ('verb_noun_ratio', 'task10_verb_noun_ratio', 'Noun-to-Verb Ratio'),
        ('subjectivity_index', 'task11_subjectivity_index', 'Subjectivity Index'),
        ('passive_density', 'task12_passive_density', 'Passive Voice Density'),
        ('avg_sent_length', 'task13_avg_sent_length', 'Avg Sentence Length'),
        ('syll_per_word', 'task13_syll_per_word', 'Syllables Per Word')
    ]
    
    for metric_key, file_name, title in metrics:
        data = monthly_stats[metric_key]
        plot_with_smoothing(data, f"Group II: {title}", "Value", 
                            f"{OUTPUT_DIR}/{file_name}_plot.png", COLORS)

if __name__ == "__main__":
    run_group2_analysis()
