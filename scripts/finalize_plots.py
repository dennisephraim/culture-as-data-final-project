import pandas as pd
import os
import matplotlib.pyplot as plt
from visualizer import plot_with_smoothing

OUTPUT_DIR = "results"
COLORS = {'FED': '#000080', 'MEDIA': '#8B0000'}

def finalize():
    print("Finalizing plots with task-numbered filenames...")
    
    # Task 2: Institutional Control
    if os.path.exists(f"{OUTPUT_DIR}/task_institutional_control_density.csv"):
        df = pd.read_csv(f"{OUTPUT_DIR}/task_institutional_control_density.csv", index_col=0)
        plot_with_smoothing(df, "Institutional Control Cluster Density", "Count per 10k Words", 
                            f"{OUTPUT_DIR}/task2_institutional_control_plot.png", COLORS)

    # Task 3: Volatility Regime
    if os.path.exists(f"{OUTPUT_DIR}/task_volatility_regime_density.csv"):
        df = pd.read_csv(f"{OUTPUT_DIR}/task_volatility_regime_density.csv", index_col=0)
        plot_with_smoothing(df, "Volatility Regime Cluster Density", "Count per 10k Words", 
                            f"{OUTPUT_DIR}/task3_volatility_regime_plot.png", COLORS)

    # Task 4: Competence Stacking
    if os.path.exists(f"{OUTPUT_DIR}/task4_stacking_density.csv"):
        df = pd.read_csv(f"{OUTPUT_DIR}/task4_stacking_density.csv", index_col=0)
        plot_with_smoothing(df, "Task 4: Competence Stacking Index", "Matches per 10k Words", 
                            f"{OUTPUT_DIR}/task4_stacking_plot.png", COLORS)

    # Task 5: Narrative Lifecycle
    for term in ['transitory', 'soft_landing']:
        csv_path = f"{OUTPUT_DIR}/task5_{term}_density.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            plot_with_smoothing(df, f"Narrative Lifecycle: '{term.replace('_', ' ').title()}'", "Count per 10k Words", 
                                f"{OUTPUT_DIR}/task5_{term}_plot.png", COLORS)

    # Group 2: Structural Abstraction (Tasks 8-13)
    if os.path.exists(f"{OUTPUT_DIR}/group2_structural_abstraction.csv"):
        df = pd.read_csv(f"{OUTPUT_DIR}/group2_structural_abstraction.csv", header=[0, 1], index_col=0)
        metrics = [
            ('nominalization_ratio', 'task8_nominalization_ratio', 'Nominalization Ratio'),
            ('noun_noun_density', 'task9_noun_noun_density', 'Noun-on-Noun Density'),
            ('verb_noun_ratio', 'task10_verb_noun_ratio', 'Noun-to-Verb Ratio'),
            ('subjectivity_index', 'task11_subjectivity_index', 'Subjectivity Index'),
            ('passive_density', 'task12_passive_density', 'Passive Voice Density'),
            ('avg_sent_length', 'task13_avg_sent_length', 'Avg Sentence Length'),
            ('syll_per_word', 'task13_syll_per_word', 'Syllables Per Word')
        ]
        for metric_key, file_base, title in metrics:
            data = df[metric_key]
            plot_with_smoothing(data, f"Group II: {title}", "Value", 
                                f"{OUTPUT_DIR}/{file_base}_plot.png", COLORS)
    
    # Group 3: Temporality (Tasks 14-17)
    if os.path.exists(f"{OUTPUT_DIR}/group3_temporality.csv"):
        df = pd.read_csv(f"{OUTPUT_DIR}/group3_temporality.csv", header=[0, 1], index_col=0)
        metrics = [
            ('gerund_ratio', 'task14_gerund_ratio', 'Gerund Ratio'),
            ('specificity_gap', 'task15_specificity_gap', 'Specificity Gap'),
            ('process_density', 'task16_process_density', 'Process Framing Density'),
            ('modal_ratio', 'task17_modal_ratio', 'Hedging/Modal Ratio')
        ]
        for metric_key, file_base, title in metrics:
            data = df[metric_key]
            plot_with_smoothing(data, f"Group III: {title}", "Value", 
                                f"{OUTPUT_DIR}/{file_base}_plot.png", COLORS)

if __name__ == "__main__":
    finalize()
