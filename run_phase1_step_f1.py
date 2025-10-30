# run_phase1_step_f1.py (Corrected)

import pandas as pd
import numpy as np
import re
from collections import Counter

print("--- Phase I, Step F1: Numerical Feature Extraction (Corrected) ---")

# --- 1. Load Data from Previous Step ---
try:
    df = pd.read_csv('train_t1.csv')
    print(f"✅ Successfully loaded 'train_t1.csv'. Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: 'train_t1.csv' not found. Please run the previous step first.")
    exit()

# --- 2. Define Feature Extraction Functions (with corrections) ---

def extract_ipq(text):
    """Extracts the Item Pack Quantity (IPQ) from text."""
    text = str(text)
    match = re.search(r'(?:pack of|pack of|pack|count)\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1

def extract_base_quantity(text):
    """Extracts and standardizes the quantity of a single item to grams (more robustly)."""
    text = str(text).lower()
    
    conversion_map = {
        'fl oz': 29.57, 'fluid ounce': 29.57,
        'oz': 28.35, 'ounce': 28.35,
        'lb': 453.59, 'pound': 453.59,
        'g': 1, 'gram': 1,
        'kg': 1000,
        'ml': 1,
        'l': 1000
    }

    # A more specific regex that looks for a number with at least one digit.
    value_regex = r'value:\s*(\d*\.?\d+)'
    unit_regex = r'unit:\s*([a-z\s]+)'
    
    value_match = re.search(value_regex, text)
    unit_match = re.search(unit_regex, text)
    
    if value_match and unit_match:
        try:
            value = float(value_match.group(1))
            unit = unit_match.group(1).strip()
            for key, factor in conversion_map.items():
                if key in unit:
                    return value * factor
        except (ValueError, IndexError):
            # If conversion fails for any reason, pass and try the next method
            pass

    # A more robust pattern for general text search
    pattern = r'(\d+\.?\d*|\.\d+)\s*(' + '|'.join(conversion_map.keys()) + r')'
    match = re.search(pattern, text)
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2)
            return value * conversion_map[unit]
        except (ValueError, IndexError):
            pass
            
    return np.nan

def extract_brand(df):
    """Generates a brand list from the data and extracts brand names."""
    def get_title(text):
        match = re.search(r'Item Name:(.*?)\n', str(text), re.DOTALL)
        return match.group(1).strip().lower() if match else ""
    
    titles = df['catalog_content'].apply(get_title)
    starts_2 = Counter(" ".join(t.split()[:2]) for t in titles if len(t.split()) > 1)
    starts_3 = Counter(" ".join(t.split()[:3]) for t in titles if len(t.split()) > 2)
    brands_from_data = {s for s, c in starts_3.items() if c > 1}
    brands_from_data.update({s for s, c in starts_2.items() if c > 1})

    def find_brand(title):
        words = title.split()
        if len(words) > 2 and " ".join(words[:3]) in brands_from_data:
            return " ".join(words[:3])
        if len(words) > 1 and " ".join(words[:2]) in brands_from_data:
            return " ".join(words[:2])
        return words[0] if words else "unknown"
        
    return titles.apply(find_brand)

# --- 3. Apply Functions to Create New Columns ---
print("⏳ Extracting features: IPQ, Base Quantity, Brand...")
df['ipq'] = df['catalog_content'].apply(extract_ipq)
df['base_quantity_g'] = df['catalog_content'].apply(extract_base_quantity)
df['brand'] = extract_brand(df)
df['total_quantity_g'] = df['base_quantity_g'] * df['ipq']
print("✅ Feature extraction complete.")

# --- 4. Save the Result ---
output_filename = 'train_f1.csv'
df.to_csv(output_filename, index=False)
print(f"✅ Step F1 complete. Output saved to '{output_filename}'.")

# --- 5. Display a preview ---
print("\nPreview of the new features:")
preview_cols = ['brand', 'ipq', 'base_quantity_g', 'total_quantity_g', 'price']
print(df[preview_cols].head())