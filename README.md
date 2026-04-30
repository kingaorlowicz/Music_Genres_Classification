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
4. Deep Learning & Transfer Learning: Advanced classification approach using a pre-trained Transformer-based model from Hugging Face.
   - Model Architecture: DistilHubert (ntu-spml/distillhubert), a distilled version of the HuBERT model that uses knowledge distillation to achieve ~95% of the original performance while being 3x faster and significantly lighter.
   - Transfer Learning: The model was fine-tuned on the FMA Small dataset for 3 epochs.
   - Preprocessing Pipeline: Resampling audio to 16kHz (required by the model), processing 10-second audio segments, AutoFeatureExtractor for robust input normalization
   - Hardware Acceleration: Training was optimized using NVIDIA CUDA on a GTX 1070 Ti, reducing training time from ~7 hours (CPU) to approximately 1 hour.

## Current Results (FMA SMALL)
### Classical ML

The best results among classical algorithms were achieved using LightGBM trained on the full feature set extracted with Librosa.

Test Accuracy: 0.4719

Classification Report:
```
+----------------+-----------+--------+----------+---------+
| Genre          | Precision | Recall | F1-Score | Support |
+----------------+-----------+--------+----------+---------+
| Electronic     |    0.51   |  0.56  |   0.53   |   200   |
| Experimental   |    0.40   |  0.38  |   0.39   |   200   |
| Folk           |    0.44   |  0.48  |   0.46   |   200   |
| Hip-Hop        |    0.57   |  0.69  |   0.63   |   200   |
| Instrumental   |    0.39   |  0.38  |   0.38   |   200   |
| International  |    0.52   |  0.46  |   0.49   |   200   |
| Pop            |    0.28   |  0.21  |   0.24   |   200   |
| Rock           |    0.59   |  0.60  |   0.60   |   200   |
+----------------+-----------+--------+----------+---------+
| Overall Acc    |           |        |   0.47   |  1600   |
+----------------+-----------+--------+----------+---------+
```
Classical models performed most reliably on genres with distinct rhythmic and instrumental structures, such as Hip-Hop, Rock, and Electronic. Conversely, Pop was the most difficult genre to classify across all tested ML models, likely due to its stylistic diversity and overlapping characteristics with other genres.

---

### Deep Learning: DistilHubert

Implementing a Transfer Learning approach with a pre-trained Transformer model (DistilHubert) led to a significant leap in performance.

Test Accuracy: 0.57

Classification Report:
```
+----------------+-----------+--------+----------+---------+
| Genre          | Precision | Recall | F1-Score | Support |
+----------------+-----------+--------+----------+---------+
| Electronic     |    0.60   |  0.64  |   0.62   |   101   |
| Experimental   |    0.47   |  0.41  |   0.44   |    90   |
| Folk           |    0.56   |  0.52  |   0.54   |    95   |
| Hip-Hop        |    0.73   |  0.72  |   0.72   |    96   |
| Instrumental   |    0.49   |  0.61  |   0.54   |    94   |
| International  |    0.66   |  0.68  |   0.67   |    96   |
| Pop            |    0.31   |  0.25  |   0.28   |   100   |
| Rock           |    0.65   |  0.70  |   0.67   |   128   |
+----------------+-----------+--------+----------+---------+
| Overall Acc    |           |        |   0.57   |   800   |
+----------------+-----------+--------+----------+---------+
```
The neural network was far more effective at capturing subtle audio nuances. While the general difficulty trends remained the same (Pop remained the hardest to classify), scores for every single genre improved by approximately 10 percentage points.

--- 

Hyperparameters testing table: https://docs.google.com/spreadsheets/d/1zVdp_KwDR07jjUFOIKxyNs_meEN2bUFtcPnCqbhq_AA/edit?gid=0#gid=0

## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, Librosa, Essentia, Matplotlib, Seaborn, PyTorch
