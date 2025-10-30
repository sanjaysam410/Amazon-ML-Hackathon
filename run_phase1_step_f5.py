# run_phase1_step_f5.py (Corrected for Memory)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

print("--- Phase I, Step F5: Text Embedding (Memory-Safe Mode) ---")

# --- 1. Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device.upper()}")
model_name = 'all-MiniLM-L6-v2'

# --- 2. Load Data ---
try:
    df = pd.read_csv('train_f3.csv')
    print(f"✅ Successfully loaded 'train_f3.csv'. Found {len(df)} texts to embed.")
except FileNotFoundError:
    print("❌ Error: 'train_f3.csv' not found. Please run the previous step first.")
    exit()

# --- 3. Load Model ---
print(f"⏳ Loading the '{model_name}' model...")
model = SentenceTransformer(model_name, device=device)
print("✅ Model loaded successfully.")

# --- 4. Generate Text Embeddings in Smaller Batches ---
texts_to_embed = df['clean_text'].fillna('').tolist()

# --- KEY CHANGE IS HERE ---
# We explicitly set a smaller batch_size. The default is 32.
# A smaller batch size uses less RAM at the cost of being slightly slower.
safe_batch_size = 16 
print(f"⏳ Generating embeddings with a memory-safe batch size of {safe_batch_size}...")

text_embeddings = model.encode(
    texts_to_embed, 
    show_progress_bar=True, 
    batch_size=safe_batch_size
)

# --- 5. Save the Embeddings ---
output_filename = 'text_embeddings.npy'
np.save(output_filename, text_embeddings)
print(f"✅ Step F5 complete. Embeddings saved to '{output_filename}'.")
print(f"✅ Shape of the saved embeddings array: {text_embeddings.shape}")