import os
import pandas as pd
from tqdm import tqdm

def build_index(extracts_path, output_csv):
    data = []
    
    # Sources to check
    sources = ['BBG', 'FT', 'FED']
    
    for source in sources:
        source_dir = os.path.join(extracts_path, source)
        if not os.path.exists(source_dir):
            continue
            
        print(f"Indexing {source}...")
        files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
        
        for filename in tqdm(files):
            file_path = os.path.join(source_dir, filename)
            
            # Filename format: [Source]_[Number]-[Date].txt
            # Example: BBG_0001-2025-10-28.txt
            try:
                parts = filename.replace('.txt', '').split('-')
                unique_id = parts[0]
                date_str = '-'.join(parts[1:])
                
                if date_str == 'unknown-date':
                    # Fallback to a neutral date or skip? Let's use None
                    date = None
                else:
                    date = pd.to_datetime(date_str, errors='coerce')
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    word_count = len(text.split())
                
                # Combine BBG and FT as MEDIA
                corpus_type = 'MEDIA' if source in ['BBG', 'FT'] else 'FED'
                
                data.append({
                    'unique_id': unique_id,
                    'filename': filename,
                    'source': source,
                    'corpus': corpus_type,
                    'date': date,
                    'word_count': word_count,
                    'path': file_path
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(data)
    # Filter for January 2021 – December 2025
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2025-12-31')
    
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # Save index
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"\nIndexed {len(df_filtered)} documents within timeframe.")
    print(df_filtered.groupby('corpus')['word_count'].sum())
    return df_filtered

if __name__ == "__main__":
    extracts_path = "/Users/pranav/Documents/culture-as-data-final-project/Extracts"
    output_csv = "/Users/pranav/Documents/culture-as-data-final-project/data/corpus_index.csv"
    build_index(extracts_path, output_csv)
