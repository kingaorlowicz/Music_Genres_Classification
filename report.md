# Music Genre Classification on the FMA Dataset 
*Music Genre Classification using Classical ML and Transformer-based Transfer Learning on the FMA Dataset*

**Course:** Data Mining  
**Institution:** AGH Kraków  
**Date:** March - May 2026  

## Abstract
<div align="justify">

Music genre classification is a long‑standing challenge, due to the high variability of musical styles, overlapping acoustic characteristics, and the subjective nature of genre boundaries. This project investigates the effectiveness of both classical machine learning methods and modern deep learning approaches for automatic genre prediction using the Free Music Archive (FMA) dataset, focusing on the balanced FMA Small subset.  

The study begins with extensive audio feature extraction using Librosa and Essentia, followed by preprocessing steps such as scaling and dimensionality reduction with PCA. Several classical models—including Logistic Regression, Support Vector Machines, Random Forests, and LightGBM—were trained and evaluated. Among these, LightGBM achieved the strongest performance with an accuracy of 0.47, showing particular strength in genres with distinctive rhythmic or instrumental patterns (Hip‑Hop, Rock, Electronic).

To explore the benefits of representation learning, a Transformer‑based model (DistilHuBERT) was fine‑tuned on raw audio waveforms. This transfer learning approach significantly outperformed all classical baselines, reaching 0.57 accuracy and improving F1‑scores across nearly all genres. Despite these gains, certain categories—especially Pop—remained difficult to classify due to their stylistic diversity and overlap with other genres.

Overall, the results highlight the limitations of hand‑crafted audio features and demonstrate the superior ability of self‑supervised Transformer models to capture complex musical structure.

## Introduction

Music genre classification is an interesting and challenging task because music is complex, diverse, and often hard to describe using simple rules. Many genres share similar sounds, instruments, or rhythms, and some tracks mix elements from several styles. Because of this, even humans sometimes disagree about the correct genre. For machine learning models, the problem becomes even harder: they must learn patterns from audio signals that are noisy, varied, and full of subtle details.

The Free Music Archive (FMA) dataset is a popular open dataset used in music research. It contains thousands of tracks from many genres, with a wide range of styles and recording qualities. This makes it a good benchmark, but also a difficult one. Some genres, like Hip‑Hop or Electronic, have strong and recognizable patterns. Others, like Pop or Experimental, are more mixed and harder to classify.

There are two main ways to approach this task. The first uses hand‑crafted audio features, such as MFCCs, chroma features, spectral statistics, and rhythm descriptors. These features can be used with classical machine learning models like Logistic Regression, SVM, Random Forest, or LightGBM. The second approach uses end‑to‑end deep learning, where a neural network learns directly from raw audio. Modern Transformer‑based models, such as DistilHuBERT, can capture much deeper and more abstract information from sound.

The goal of this project is to compare these two approaches. I wanted to see how classical ML models perform when using engineered audio features, and how much improvement can be achieved with a pre‑trained Transformer model fine‑tuned on the FMA dataset. I also wanted to understand which genres are easier or harder to classify, and which types of features seem most helpful for the task.


## Dataset Description

The dataset used in this project comes from the Free Music Archive (FMA), an open collection of music tracks released under Creative Commons licenses. The specific subset used here is FMA Small, which contains 8,000 tracks evenly distributed across 8 genres. Each track is a 30‑second audio clip.

The official repository is available at: https://github.com/mdeff/fma

The dataset includes two main components:
- Metadata files (contain information such as genre, artist, album)
- Audio files (30s clips, stored in a structured folder hierarchy based on track IDs)

Genres in FMA SMALL:
- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

## Data preparation 
- Resampling: All audio files were loaded with a fixed sampling rate of 22050 Hz. This allowed the models to keep the most important acoustic information while reducing the computational load.
- Mono Conversion: Multi-channel (stereo) audio files were forced to a single mono channel during loading. This step removed spatial differences and standardized the input signal for the feature extraction libraries.

## Feature Engineering
### Librosa
Feature engineering was an important part of this project, especially for the classical machine learning models. I used the Librosa library to extract several groups of audio features that describe different aspects of the signal, such as timbre, energy, pitch content, and spectral shape.

I created two versions of the feature set:

- a smaller set (~70 features) – using only the mean and standard deviation of each feature,
- a larger set (~519 features) – where I originally included many more statistics (skewness, kurtosis, median, min, max), but later removed them to simplify the feature space.

Both versions were tested in the experiments. The smaller set was easier to train and less prone to overfitting, while the larger set captured more detailed information about the audio signal.

Description of all calculated Librosa features:
- **Zero Crossing Rate (ZCR):** counts how often the audio signal changes sign (from positive to negative or the opposite).
- **Chroma STFT:** describe the distribution of energy across the 12 pitch classes (C, C#, D, …, B).
- **RMS Energy:** represents the loudness or energy of the signal over time.
- **Spectral Centroid:** is often described as the “center of mass” of the spectrum. It tells how bright or dark the sound is.
- **MFCCs (Mel-Frequency Cepstral Coefficients):** MFCCs are one of the most widely used features in audio analysis.
They describe the timbre of the sound — the overall “color” or texture.

The larger set originally included many additional statistics, but these were later removed because they increased dimensionality without improving model performance. The smaller set turned out to be more stable and easier to use with classical ML models.

---

### Essentia
I also created a second feature set using the Essentia library. Essentia is widely used in Music Information Retrieval because it provides high‑quality audio descriptors and a more detailed signal analysis pipeline. It processes audio frame‑by‑frame and gives very stable results for spectral and harmonic features.

Just like with Librosa, I extracted the same five groups of features:

- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Chroma (HPCP)
- MFCC

This allowed me to compare both libraries directly and check whether Essentia’s implementations improve classification performance.

The final Essentia feature set has the same structure as the Librosa one: each feature group is summarized using mean and standard deviation, which keeps the dimensionality manageable and consistent across both pipelines.

---

### Dimensionality Reduction with PCA
To better understand the structure of the feature space and to reduce dimensionality, I applied Principal Component Analysis (PCA) to both Librosa feature sets. PCA is often used to compress high‑dimensional data while keeping most of the variance, and it can  help classical ML models work faster or generalize better.

For both feature sets, I generated a cumulative explained variance plot. Surprisingly, the first two principal components explained almost 100% of the total variance. At first glance, this might suggest that the data lies mostly in a low‑dimensional space and that PCA should work very well. However, this turned out to be misleading.

The reason is that many audio features are highly correlated. For example:

- MFCC coefficients often move together,
- chroma bins are related harmonically,
- RMS, spectral centroid, and ZCR all depend on energy and brightness.

Because of this, PCA captures a lot of variance very quickly — but this variance does not correspond to meaningful structure for classification.

I also plotted the projection of all samples onto the first two components (PCA1 vs PCA2). The result was very clear - all points were clustered tightly in one place, with no visible separation between genres. There were no clusters, no patterns, and no boundaries.

This showed that even though PCA explains almost all variance, the variance it captures is not related to genre differences.


</div>