# run_phase1_step_f4.py

import pandas as pd
import requests
import os
from tqdm import tqdm

print("--- Phase I, Step F4: Image Download ---")

# --- 1. Setup ---
# Define the directory where images will be saved
IMG_DIR = "product_images"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
    print(f"✅ Created directory: '{IMG_DIR}'")

# --- 2. Load Data from Previous Step ---
try:
    df = pd.read_csv('train_f3.csv')
    print(f"✅ Successfully loaded 'train_f3.csv'. Found {len(df)} image links to process.")
except FileNotFoundError:
    print("❌ Error: 'train_f3.csv' not found. Please run the previous step first.")
    exit()

# --- 3. Download Loop ---
print(f"⏳ Starting download... Images will be saved in the '{IMG_DIR}' folder.")

# We will iterate with a progress bar from tqdm
for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading Images"):
    sample_id = row['sample_id']
    url = row['image_link']
    
    # Define the output filename for the image
    image_path = os.path.join(IMG_DIR, f"{sample_id}.jpg")
    
    # --- IMPORTANT ---
    # Skip downloading if the file already exists.
    # This allows you to stop and restart the script without losing progress.
    if os.path.exists(image_path):
        continue
        
    try:
        # Send a request to the URL to get the image
        response = requests.get(url, stream=True, timeout=10)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Write the image content to a file
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
    except Exception as e:
        # This will catch timeouts, connection errors, etc., and print a warning
        # but will not stop the script from continuing with the next image.
        # print(f"\n⚠️ Warning: Could not download image for sample_id {sample_id}. Error: {e}")
        pass

print(f"\n✅ Step F4 complete. All available images have been downloaded.")