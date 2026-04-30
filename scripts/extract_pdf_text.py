import os
import re
import subprocess
from datetime import datetime
from dateutil import parser
import hashlib

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF using pdftotext command line tool."""
    try:
        result = subprocess.run(['pdftotext', pdf_path, '-'], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def find_date_in_text(text, filename):
    """Attempts to find a date in the text or filename."""
    # Strategy 1: Check filename for YYYY-MM-DD (Common in FED)
    file_date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if file_date_match:
        return file_date_match.group(1)

    # Strategy 2: Look for common patterns in first 2000 chars of text
    snippet = text[:2000]
    
    # BBG Pattern: "November 10, 2022 at 9:58 AM EST"
    bbg_match = re.search(r'([A-Z][a-z]+ \d{1,2}, \d{4})', snippet)
    if bbg_match:
        try:
            dt = parser.parse(bbg_match.group(1))
            return dt.strftime('%Y-%m-%d')
        except:
            pass

    # FT Pattern: "Published AUG 7 2024"
    ft_match = re.search(r'Published ([A-Z]{3} \d{1,2} \d{4})', snippet, re.IGNORECASE)
    if ft_match:
        try:
            dt = parser.parse(ft_match.group(1))
            return dt.strftime('%Y-%m-%d')
        except:
            pass
            
    # Generic date patterns
    generic_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', snippet)
    if generic_match:
        try:
            dt = parser.parse(generic_match.group(1))
            return dt.strftime('%Y-%m-%d')
        except:
            pass

    # Last resort: Try to find any date-like string in the first few lines
    for line in snippet.split('\n')[:20]:
        try:
            # Avoid very short strings or obvious non-dates
            if len(line.strip()) > 5:
                dt = parser.parse(line, fuzzy=True)
                # Ensure it's a reasonable date (between 2000 and 2030)
                if 2000 <= dt.year <= 2030:
                    return dt.strftime('%Y-%m-%d')
        except:
            continue

    return "unknown-date"

def process_folders(base_path, source_folders):
    extracts_dir = os.path.join(base_path, 'Extracts')
    os.makedirs(extracts_dir, exist_ok=True)
    
    stats = {}

    for source in source_folders:
        source_path = os.path.join(base_path, source)
        if not os.path.exists(source_path):
            print(f"Source folder {source} not found.")
            continue
            
        target_dir = os.path.join(extracts_dir, source)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Processing {source}...")
        word_count = 0
        file_counter = 0
        
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_counter += 1
                    pdf_path = os.path.join(root, file)
                    text = extract_text_from_pdf(pdf_path)
                    
                    if not text:
                        continue
                        
                    date_str = find_date_in_text(text, file)
                    
                    # Generate unique ID (hash of filename + counter to ensure uniqueness)
                    unique_id = f"{source}_{file_counter:04d}"
                    output_filename = f"{unique_id}-{date_str}.txt"
                    output_path = os.path.join(target_dir, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # Word count (simple split)
                    word_count += len(text.split())
        
        stats[source] = {
            'files': file_counter,
            'word_count': word_count
        }
        
    return stats

if __name__ == "__main__":
    base_dir = "/Users/pranav/Documents/culture-as-data-final-project"
    folders = ['BBG', 'FT', 'FED']
    results = process_folders(base_dir, folders)
    
    print("\n--- Summary ---")
    for source, data in results.items():
        print(f"{source}: {data['files']} files, {data['word_count']} words total.")
