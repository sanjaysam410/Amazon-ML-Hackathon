# run_phase1_step_f3.py (Corrected)

import pandas as pd
import numpy as np
import re

print("--- Phase I, Step F3: Text Cleaning (Corrected) ---")

# --- 1. Load Data from Previous Step ---
try:
    df = pd.read_csv('train_f2.csv')
    print(f"✅ Successfully loaded 'train_f2.csv'. Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: 'train_f2.csv' not found. Please run the previous step first.")
    exit()

# --- 2. Define the Improved Text Cleaning Function ---
def clean_catalog_text_corrected(text):
    """
    Correctly cleans the raw catalog_content text by removing only the
    boilerplate tags, not the entire lines.
    """
    text = str(text)
    
    # Create a single regex pattern to find all boilerplate tags.
    # The `|` symbol means "OR". This is much more efficient.
    boilerplate_pattern = re.compile(
        r'item name:|bullet point \d?:|product description:|value:|unit:',
        re.IGNORECASE
    )
    
    # Replace the found tags with a space, leaving the content.
    text = boilerplate_pattern.sub(' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any remaining special characters (non-letters, non-numbers, non-spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Replace multiple whitespace characters (including newlines) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 3. Apply the Corrected Cleaning Function ---
print("⏳ Applying corrected text cleaning function...")
df['clean_text'] = df['catalog_content'].apply(clean_catalog_text_corrected)
print("✅ Text cleaning complete.")

# --- 4. Save the Result ---
output_filename = 'train_f3.csv'
df.to_csv(output_filename, index=False)
print(f"✅ Step F3 complete. Output saved to '{output_filename}'.")

# --- 5. Display a preview ---
print("\nPreview of original vs. corrected cleaned text for the first entry:")
print("\n--- ORIGINAL ---")
print(df['catalog_content'].iloc[0])
print("\n--- CLEANED (Corrected) ---")
print(df['clean_text'].iloc[0])