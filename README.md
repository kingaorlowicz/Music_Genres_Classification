# Music Genre Classification from Audio Features
- **Goal:** classify music tracks into genres and analyze which audio features are most discriminative
- **Data:** Free Music Archive dataset (official repository): https://github.com/mdeff/fma

Note: This project was initially started in collaboration with user Jakub-Karczewski, but is currently being developed independently.


## Overview
This is an academic project developed as part of the Data Mining course.
The project focuses on music genre classification using the Free Music Archive (FMA) dataset. It combines audio signal processing, feature engineering, and machine learning techniques to build and evaluate predictive models, as well as to understand which audio features contribute most to genre discrimination.

## Progress Tracking: What’s Done So Far? (FMA SMALL)
1. Data Preparation & Exploration
   - Metadata Integration: Successfully mapped audio file paths to the FMA metadata database.
   - Audio Processing: Extracted audio-based features using multiple libraries.
2. Feature Engineering
   - Extracted features using librosa.
   - Extended feature set using Essentia.
   - Dimensionality Reduction: Implemented PCA to reduce feature space while preserving variance
   - Scaling: Standardized features to ensure optimal performance for distance-based and gradient-based algorithms.
3. Machine Learning Modeling
   - Implemented multiple classification algorithms:
      - Logistic Regression
      - SVM
      - Random Forest
      - LightGBM
   - Model Comparison: Evaluated and compared different algorithms on the fma_small subset
   - Performance Evaluation: Monitored training vs. test accuracy to detect and mitigate overfitting



Hyperparameters testing table: https://docs.google.com/spreadsheets/d/1zVdp_KwDR07jjUFOIKxyNs_meEN2bUFtcPnCqbhq_AA/edit?gid=0#gid=0

## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, Librosa, Essentia, Matplotlib, Seaborn
