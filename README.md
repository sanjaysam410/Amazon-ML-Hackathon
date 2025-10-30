# Amazon Product Catalog ML Feature Engineering Pipeline

This repository provides a full-featured pipeline for preparing and engineering features from Amazon product catalog data, ready for use in machine learning and deep learning models. The workflow is divided into logical, well-documented Python scripts that handle each transformation step in sequence.

## Features

- **Target Transformation:** Applies log transformation to product prices for better regression performance.
- **Feature Extraction:** Extracts useful features such as item pack quantity, normalized base quantity (in grams), and brand information from complex catalog text using regular expressions and heuristics.
- **Imputation & Binning:** Detects missing values, flags them, imputes numerics with the median, and bins rare brands into an "OTHER" category.
- **Text Cleaning:** Cleans product description text by removing boilerplate keywords and special characters for robust NLP processing.
- **Image Download:** Bulk downloads product images linked in the dataset, skipping already-downloaded files for efficiency.
- **Text Embedding:** Converts cleaned product text into high-dimensional vector embeddings with a Sentence Transformer model compatible with PyTorch or CUDA.

## Pipeline Execution

Run each step sequentially for full data preparation:

```bash
python run_phase1_step_t1.py     # Target variable transformation (log of price)
python run_phase1_step_f1.py     # Feature extraction (quantity, brand, etc.)
python run_phase1_step_f2.py     # Imputation and rare brand binning
python run_phase1_step_f3.py     # Text cleaning
python run_phase1_step_f4.py     # Image download (if using image features)
python run_phase1_step_f5.py     # Text embedding (generates .npy file)
````

Each step produces a new CSV or output file, ensuring data lineage can be tracked or rolled back.

## Model Training Process

Once your features are prepared, you can train a wide range of machine learning or deep learning models. For example:

1.  **Data Preparation:** - Use `train_f3.csv` as your main training feature file.

      - Use `text_embeddings.npy` as additional model inputs if modeling with neural networks.

2.  **Train/Test Split:** - Split your dataset into training and validation sets.

      - Standardize or normalize features as needed (outside the pipeline).

3.  **Model Selection:** - For tabular features: Try regression models such as XGBoost, LightGBM, or scikit-learn’s RandomForest.

      - For embedding+tabular features: Use neural network regressor (e.g., PyTorch or TensorFlow), concatenating tabular data and embeddings.

4.  **Training:** - Define your model architecture.

      - Fit it using the log-transformed price as your target.
      - Use MSE or RMSE as your loss metric.

5.  **Evaluation:** - Evaluate predictions on the log scale, or invert the transformation (exp(`log_price`) - 1) for real-world price interpretation.

      - Visualize feature importance or embedding clustering for insight.

## Output

  - Intermediate processed CSVs per stage
  - Cleaned and normalized training data
  - Downloaded product images (for multi-modal models)
  - High-quality vector embeddings (`text_embeddings.npy`)

## Requirements

  - Python 3.8+
  - pandas, numpy, scikit-learn
  - sentence-transformers
  - tqdm, requests, torch
