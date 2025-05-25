# Breast-Cancer-Classification-using-GA-Based-Feature-Selection

## Project Overview
This project focuses on feature selection using a Genetic Algorithm (GA) for the Breast Cancer Wisconsin (Diagnostic) dataset, followed by classification using K-Nearest Neighbors (KNN) and Artificial Neural Network (ANN) algorithms. The goal is to improve the precision, recall, and accuracy of classifiers for breast cancer diagnosis by selecting the most relevant features.

## Key Features
- **Genetic Algorithm (GA)**: Used to select optimal features from the dataset.
- **K-Nearest Neighbors (KNN)**: A supervised learning classifier for breast cancer prediction.
- **Artificial Neural Network (ANN)**: A deep learning model for classification.
- **Performance Metrics**: Accuracy, precision, recall, and confusion matrices are evaluated for both classifiers.

## Dataset
![BreastCancerDataSet.csv](https://github.com/AponGhosh/Breast-Cancer-Classification-using-GA-Based-Feature-Selection/blob/main/BreastCancerDataSet.csv)

## Methodology
1. **Data Preprocessing**: The dataset is loaded, split into training and testing sets (80-20), and standardized using `StandardScaler`.
2. **Feature Selection**: A Genetic Algorithm is implemented to select the most relevant features.
3. **Classification**: 
   - KNN and ANN models are trained on the selected features.
   - Performance metrics (accuracy, precision, recall) are calculated.
4. **Evaluation**: Confusion matrices and classification reports are generated to compare the models.

## Results
| Metrics    | KNN    | ANN    |
|------------|--------|--------|
| Accuracy   | 96%    | 97%    |
| Precision  | 96%    | 96%    |
| Recall     | 99%    | 100%   |

### Fitness score and Confusion Matrices
![Fitness score and matrices](https://github.com/AponGhosh/Breast-Cancer-Classification-using-GA-Based-Feature-Selection/blob/main/Fitness-score-and-Matrices.png)

## Dependencies
- **Python 3.11.12
- **Libraries**:
  - `numpy` (for numerical operations)
  - `pandas` (for data manipulation)
  - `scikit-learn` (for ML models and metrics)
  - `matplotlib` (for visualizations)
  - `seaborn` (optional, for enhanced plots)
- **Google Colab Tools**:
  - Built-in dataset loading (from sklearn.datasets import load_breast_cancer)
  - GPU/TPU acceleration (optional for ANN training)

## Clone Repository
   ```bash
   git clone https://github.com/AponGhosh/Breast-Cancer-Classification-using-GA-Based-Feature-Selection.git
