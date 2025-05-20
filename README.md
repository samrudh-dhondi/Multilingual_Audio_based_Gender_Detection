
# Multi-lingual Audio-based Gender Detection

Multilingual Audio-Based Gender Detection is an AI-driven project that identifies a speaker's gender from audio recordings across multiple languages (English, German, and Hindi). By extracting key acoustic features like MFCC, Chroma, Mel Spectrogram, and VGGish embeddings, and using both machine learning (DT, RF, XGB, SVM, etc.) and deep learning models (CNN, LSTM), we built and evaluated gender classification systems in monolingual and cross-lingual contexts. This project addresses the challenges of language diversity and aims to support applications in security, voice assistants, call centers, and inclusive digital experiences.

## Dataset

This project uses three diverse audio datasets—RAVDESS, Berlin EmoDB, and IITKGP-SEHSC—to build a multilingual gender detection model. RAVDESS includes high-quality English speech recordings from 24 actors expressing eight emotions, providing rich emotional and acoustic variety. Berlin EmoDB is a German emotional speech dataset with 535 utterances from 10 speakers, covering seven emotion classes, recorded at 16 kHz. IITKGP-SEHSC is a Hindi dataset featuring recordings from 10 professional radio artists simulating eight emotions using 15 neutral Hindi prompts. All datasets are organized by gender to enable effective feature extraction and model training across English, German, and Hindi speech.

## Methodology 
### Feature Extraction

To prepare audio data for gender classification, we extracted both traditional and deep audio features. Extracted features include 40 MFCCs, 12 Chroma, 128 Mel Spectrogram and some other features including deep audio embeddings from the pretrained VGGish model. Audio was resampled to 16 kHz mono to ensure consistency and compact representation.

Extracted features were saved as .npy files with gender labels and reshaped into arrays of size (20, 450) to prepare them for training machine learning and deep learning models.
These features provide both low-level and high-level insights, helping our models effectively learn gender-related vocal patterns.

![mfcc](https://github.com/user-attachments/assets/85b83883-9807-410a-bce5-dc7f0f8a968d)
![mel](https://github.com/user-attachments/assets/7217c4c1-7695-42de-a142-d4b46d73686e)

## Proposed Models

### Machine Learning Models

We experimented with various machine learning algorithms for gender classification using extracted audio features. Decision Tree (DT), Random Forest (RF), XGBoost, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Logistic Regression, Ensemble Classifier.

A soft-voting ensemble combining all ML models was used to improve accuracy and reliability in cross-lingual settings. It outperformed most individual models in generalizing across languages.

### Deep Learning Models

We also implemented deep learning approaches to better capture complex patterns in voice data.

- CNN Model: Captured spatial patterns in audio feature maps and generalized well across different languages. Regularization techniques ensured better generalization and reduced overfitting.

- LSTM-GRU Model: Combined sequential modeling power of LSTMs and GRUs to learn time-dependent features. Achieved strong performance across language boundaries.

## Results

Models were trained on combined multilingual datasets, deep learning approaches significantly outperformed traditional machine learning models in gender classification accuracy.

- XGBoost was the best among ML models with an accuracy of 96.90%.
- The Ensemble Soft Voting Classifier and Random Forest also performed well, achieving 96.57% and 96.12%, respectively.
- CNN emerged as the top performer overall, achieving an impressive 98.11% accuracy.
- LSTM model achieved 96.22%, showcasing the strength of sequential deep architectures in multilingual voice-based gender recognition.

<img src="https://github.com/user-attachments/assets/376f068e-104b-4c83-9b1b-9f2cb43496a8" alt="combined" width="400"/>
<img src="https://github.com/user-attachments/assets/39de732e-dc80-4d6e-bd0c-60ba0185e3cd" alt="image" width="300"/>

