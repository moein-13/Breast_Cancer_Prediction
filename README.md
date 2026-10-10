# Breast Cancer Prediction: Comparative Analysis (SVM vs. PyTorch ANN)

## Overview
This project focuses on the classification of breast cancer tumors as malignant or benign using the UCI Breast Cancer Wisconsin (Diagnostic) dataset. It provides a comparative analysis between traditional Machine Learning baseline models and a Deep Learning approach, specifically contrasting a Support Vector Machine (SVM) with an Artificial Neural Network (ANN) built in PyTorch.

## Dataset
The dataset utilized is the `wdbc.data` from the UC Irvine Machine Learning Repository. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image.

**Key Feature Categories:**
- Radius, Texture, Perimeter, Area, Smoothness
- Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
*(Each feature includes mean, standard error, and "worst" or largest values).*

## Project Workflow
1. **Data Preprocessing:** 
   - Loaded raw `.data` files using Pandas with explicitly defined column headers.
   - Applied `LabelEncoder` to transform the categorical 'diagnosis' target variable.
   - Scaled feature matrices using `StandardScaler` to normalize distributions for optimal gradient descent.
2. **Exploratory Data Analysis (EDA):**
   - Conducted feature correlation analysis using Seaborn/Matplotlib heatmaps.
   - Dropped highly correlated and redundant columns to reduce multicollinearity and improve model generalization.
3. **Model Training & Evaluation:**
   - Split data into training and testing sets using `train_test_split`.
   - Trained traditional ML models (SVM) establishing a performance baseline.
   - Designed, trained, and evaluated a PyTorch-based neural network.

## Architectures & Algorithms
### 1. Artificial Neural Network (Deep Learning)
- **Framework:** PyTorch
- **Architecture:** Feedforward neural network built using `nn.Sequential`.
- **Loss Function:** Binary Cross Entropy (`BCELoss`) optimized for binary classification.
- **Optimizer:** Adam optimizer for adaptive learning rate adjustments.

### 2. Support Vector Machine (Machine Learning Baseline)
- **Framework:** Scikit-Learn
- **Role:** Served as the primary baseline to evaluate whether deep learning complexity yields performance gains on structured tabular data.

## Key Findings
Comparative analysis demonstrated that the **Support Vector Machine (SVM) achieved slightly higher accuracy** than the Artificial Neural Network on this specific dataset. This highlights an important principle in data science: traditional, well-tuned machine learning algorithms often outperform deep learning models on smaller, highly structured tabular medical datasets where complex feature extraction (the primary strength of ANNs) is less critical.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/moein-13/Breast_Cancer_Prediction.git](https://github.com/moein-13/Breast_Cancer_Prediction.git)
