# run_phase1_step_f2.py

import pandas as pd
import numpy as np

print("--- Phase I, Step F2: Imputation & Binning ---")

# --- 1. Load Data from Previous Step ---
try:
    df = pd.read_csv('train_f1.csv')
    print(f"✅ Successfully loaded 'train_f1.csv'. Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: 'train_f1.csv' not found. Please run the previous step first.")
    exit()

# --- 2. Create 'quantity_missing' Indicator ---
# This column will be 1 if total_quantity_g is NaN, and 0 otherwise.
df['quantity_missing'] = df['total_quantity_g'].isnull().astype(int)
missing_count = df['quantity_missing'].sum()
print(f"✅ Created 'quantity_missing' indicator. Found {missing_count} missing quantity values.")

# --- 3. Impute Missing 'total_quantity_g' with Median ---
median_quantity = df['total_quantity_g'].median()
df['total_quantity_g'].fillna(median_quantity, inplace=True)
print(f"✅ Imputed missing 'total_quantity_g' values with median ({median_quantity:.2f}g).")

# --- 4. Bin Rare Brands into 'OTHER' ---
brand_counts = df['brand'].value_counts()
# Identify brands that appear 10 times or fewer. You can adjust this threshold.
rare_brands = brand_counts[brand_counts <= 5].index

# Replace these rare brands with the 'OTHER' category
df['brand'] = df['brand'].replace(rare_brands, 'OTHER')
rare_brands_count = len(rare_brands)
print(f"✅ Binned {rare_brands_count} rare brands into 'OTHER' category.")

# --- 5. Save the Result ---
output_filename = 'train_f2.csv'
df.to_csv(output_filename, index=False)
print(f"✅ Step F2 complete. Output saved to '{output_filename}'.")

# --- 6. Display a preview ---
print("\nPreview of the cleaned features:")
preview_cols = ['brand', 'total_quantity_g', 'quantity_missing', 'price']
print(df[preview_cols].head())
print("\nTop 10 most common brands after binning:")
print(df['brand'].value_counts().head(10))
