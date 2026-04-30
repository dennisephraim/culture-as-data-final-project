import matplotlib.pyplot as plt
import pandas as pd

def plot_with_smoothing(df_metrics, title, ylabel, filename, colors, window=3):
    """
    Plots metrics with a faint monthly line and a bold moving average.
    df_metrics: DataFrame with months as index and corpus as columns.
    """
    plt.figure(figsize=(10, 6))
    
    # Ensure index is datetime for better plotting
    if not isinstance(df_metrics.index, pd.DatetimeIndex):
        try:
            df_metrics.index = pd.to_datetime(df_metrics.index.astype(str))
        except:
            pass
    
    for corpus in df_metrics.columns:
        color = colors.get(corpus, 'gray')
        
        # Original monthly data (faint)
        plt.plot(df_metrics.index, df_metrics[corpus], 
                 color=color, alpha=0.2, linewidth=1, label=f"{corpus} (Monthly)")
        
        # Moving Average (bold)
        ma = df_metrics[corpus].rolling(window=window, min_periods=1).mean()
        plt.plot(df_metrics.index, ma, 
                 color=color, alpha=1.0, linewidth=2.5, label=f"{corpus} ({window}m MA)")
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
