# run_phase1_step_t1.py

import pandas as pd
import numpy as np

print("--- Phase I, Step T1: Target Transformation ---")

# --- 1. Load Data ---
try:
    df_train = pd.read_csv('train1.csv')
    print(f"✅ Successfully loaded 'train1.csv'. Shape: {df_train.shape}")
except FileNotFoundError:
    print("❌ Error: 'train1.csv' not found. Please ensure it is in the correct folder.")
    exit()

# --- 2. Perform Log Transformation ---
# We use np.log1p which calculates log(1 + x). This is safer than np.log
# as it gracefully handles any potential prices of 0.
df_train['log_price'] = np.log1p(df_train['price'])
print("✅ Created 'log_price' column successfully.")

# --- 3. Save the Result ---
# We save the output to a new file to mark the completion of this step.
output_filename = 'train_t1.csv'
df_train.to_csv(output_filename, index=False)
print(f"✅ Step T1 complete. Output saved to '{output_filename}'.")

# --- 4. Display a preview ---
print("\nPreview of the transformation:")
print(df_train[['price', 'log_price']].head())