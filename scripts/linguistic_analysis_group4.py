import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import pearsonr
from visualizer import plot_with_smoothing

# --- Configuration ---
INDEX_PATH = "data/corpus_index.csv"
CPI_PATH = "data/cpi_data.csv"
GROUP2_STATS = "results/group2_structural_abstraction.csv"
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

# --- Group IV: Relational Metrics (The "Linguistic Delta") ---

def run_group4_analysis():
    print("Initializing Group IV Analysis...")
    
    # Task 18: CPI/Nominalization Correlation
    if os.path.exists(GROUP2_STATS) and os.path.exists(CPI_PATH):
        g2_df = pd.read_csv(GROUP2_STATS, header=[0, 1], index_col=0)
        
        # Multi-index adjustment
        fed_nom = g2_df['nominalization_ratio']['FED']
        fed_nom.index = pd.PeriodIndex(fed_nom.index, freq='M')
        
        cpi_df = pd.read_csv(CPI_PATH)
        cpi_df['Date'] = pd.to_datetime(cpi_df['Date']).dt.to_period('M')
        cpi_df.set_index('Date', inplace=True)
        
        merged = pd.concat([fed_nom, cpi_df['CPI_MoM_Percent_Change']], axis=1).dropna()
        if len(merged) < 2:
            print(f"Warning: Not enough data for correlation. Merged size: {len(merged)}")
        else:
            merged.columns = ['FED', 'CPI']
            merged['FED'] = pd.to_numeric(merged['FED'])
            corr, _ = pearsonr(merged['FED'], merged['CPI'])
            print(f"Task 18 Correlation (Fed Nom vs CPI): {corr:.4f}")
            
            # Dual axis plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            idx = merged.index.to_timestamp()
            ax1.plot(idx, merged['FED'], color=COLORS['FED'], label='Fed Nominalization', alpha=0.3)
            ax1.plot(idx, merged['FED'].rolling(3).mean(), color=COLORS['FED'], linewidth=2.5, label='Fed Nom (MA)')
            ax2.plot(idx, merged['CPI'], color='green', label='CPI MoM %', linestyle='--', alpha=0.3)
            ax2.plot(idx, merged['CPI'].rolling(3).mean(), color='green', linewidth=2.5, label='CPI (MA)')
            ax1.set_ylabel('Nominalization Ratio', color=COLORS['FED'])
            ax2.set_ylabel('CPI MoM % Change', color='green')
            plt.title(f"Task 18: Fed Nominalization vs. CPI (Corr: {corr:.2f})")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/task18_cpi_correlation_plot.png")
            plt.close()

    # Task 19: Narrative Lag
    vol_path = f"{OUTPUT_DIR}/task3_volatility_regime_density.csv"
    if os.path.exists(vol_path):
        vol_df = pd.read_csv(vol_path, index_col=0)
        lags = range(-6, 7)
        corrs = [vol_df['MEDIA'].shift(lag).corr(vol_df['FED']) for lag in lags]
        
        # Apply paper format styling
        plt.rcParams.update({
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "figure.figsize": (10, 6)
        })
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(lags, corrs, color=COLORS['FED'], alpha=0.8, edgecolor='black', width=0.6)
        
        # Highlight the peak
        max_corr_idx = np.argmax(corrs)
        bars[max_corr_idx].set_color(COLORS['MEDIA'])
        bars[max_corr_idx].set_edgecolor('black')
        
        plt.title("Task 19: Narrative Lag (Volatility Regime)")
        plt.xlabel("Lag in Months (Negative = Fed Leads, Positive = Media Leads)")
        plt.ylabel("Cross-Correlation")
        plt.xticks(lags)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/task19_narrative_lag_plot.png")
        plt.close()

    # Task 20 & 21
    baseline = pd.read_csv(f"{OUTPUT_DIR}/task1_normalization_baseline.csv", index_col=0)
    regex20 = re.compile(r'\b(framework|methodology)\b', re.I)
    results20 = []
    regex21 = re.compile(r'\b(dissent|alternative|disagree|however|nevertheless)\b', re.I)
    results21 = []
    
    for _, row in df.iterrows():
        text = get_text(row['path'])
        results20.append({'month_year': row['month_year'], 'corpus': row['corpus'], 'count': len(regex20.findall(text))})
        results21.append({'month_year': row['month_year'], 'corpus': row['corpus'], 'count': len(regex21.findall(text))})
    
    c20 = pd.DataFrame(results20).groupby(['month_year', 'corpus'])['count'].sum().unstack().fillna(0)
    d20 = (c20 / baseline) * 10000
    plot_with_smoothing(d20, "Task 20: Self-Referentiality Index", "Count per 10k Words", f"{OUTPUT_DIR}/task20_self_referentiality_plot.png", COLORS)
    
    c21 = pd.DataFrame(results21).groupby(['month_year', 'corpus'])['count'].sum().unstack().fillna(0)
    d21 = (c21 / baseline) * 10000
    plot_with_smoothing(d21, "Task 21: Dissent/Complexity Index", "Count per 10k Words", f"{OUTPUT_DIR}/task21_dissent_scarcity_plot.png", COLORS)

    # Task 22: Readability Delta
    if os.path.exists(GROUP2_STATS):
        g2_df = pd.read_csv(GROUP2_STATS, header=[0, 1], index_col=0)
        fk_fed = 0.39 * g2_df['avg_sent_length']['FED'] + 11.8 * g2_df['syll_per_word']['FED'] - 15.59
        fk_media = 0.39 * g2_df['avg_sent_length']['MEDIA'] + 11.8 * g2_df['syll_per_word']['MEDIA'] - 15.59
        fk_df = pd.DataFrame({'FED': fk_fed, 'MEDIA': fk_media})
        plot_with_smoothing(fk_df, "Task 22: Flesch-Kincaid Grade Level Comparison", "Grade Level", f"{OUTPUT_DIR}/task22_readability_delta_plot.png", COLORS)

if __name__ == "__main__":
    run_group4_analysis()
