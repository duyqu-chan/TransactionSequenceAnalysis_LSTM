# Customer Transaction Sequence Classification with LSTM

## Project Overview
This project aims to predict which customers are likely to purchase a targeted product based on their last ten transactions. Given the highly imbalanced nature of the dataset, various architectures, regularization techniques, and class balancing methods were explored to address the imbalance problem and improve model performance.

The primary model used for classification is **LSTM (Long Short-Term Memory)**, a type of recurrent neural network (RNN) ideal for sequence-based tasks. The project investigates different approaches for improving the model, including **Auto-encoder LSTM** for anomaly detection, **CNN + LSTM**, and the use of **Graph Embeddings** and **GNN** (Graph Neural Networks).

## Objective
The objective is to accurately classify customers based on their transaction sequences and predict whether they will apply for a targeted product (class 1) or not (class 0). The project specifically tackles:
- Class imbalance in the dataset
- Evaluation of model performance using metrics like F1-Score, Precision, Recall, and MCC
- Exploration of different neural network architectures for sequence classification

## To-Do List
- **Auto-encoder LSTM:** Treating the minority class as outliers for anomaly detection.
- **CNN + LSTM:** Exploring convolutional layers alongside LSTM for feature extraction.
- **Graph Embeddings:** Embedding transaction and customer data as graphs to improve predictions.
- **Graph Neural Networks (GNN):** Modeling with GNN to capture complex relationships in the data.

## Libraries and Dependencies
The following libraries are essential for the project:
- **TensorFlow & Keras:** For building and training the LSTM model.
- **imblearn:** For handling class imbalance through techniques like SMOTE (Synthetic Minority Over-sampling Technique) and undersampling.
- **keras-tuner:** For hyperparameter tuning of Keras models.
- **sklearn:** For model training, evaluation, and metrics calculation.
- **matplotlib, seaborn:** For visualization and analysis of results.

## Data Preprocessing and Model Training
The dataset consists of customer transaction sequences with class 0 (negative) and class 1 (positive) labels. Several preprocessing steps were applied:
1. **Handling Class Imbalance:** Using SMOTE and undersampling to balance the dataset.
2. **Feature Engineering:** One-hot encoding of transaction types and other relevant features.
3. **Model Evaluation:** The model performance was evaluated based on accuracy, precision, recall, F1-Score, MCC, and confusion matrix.

## Model Architecture
The core model uses an LSTM architecture, which is well-suited for sequential data. The model consists of:
- Input layer: Representing customer transactions.
- LSTM layers: For capturing sequential dependencies.
- Dense layers: For classification output.
- Dropout: For regularization and preventing overfitting.

## Results and Discussion
- The model achieved an accuracy of approximately 91.30% on the test set.
- Precision was moderate at 67.15%, indicating room for improvement in correctly identifying positive cases.
- Recall was lower at 36.52%, suggesting that many positive instances were missed by the model.
- The F1 score, combining precision and recall, was 47.31%, showing a balance between the two but highlighting the need for better recall performance.

## Future Work
- **Hyperparameter Optimization:** Using techniques like RandomSearch to find optimal hyperparameters for the model.
- **Graph Neural Networks:** Implementing GNN for better sequence classification performance by capturing complex relationships.
- **Anomaly Detection:** Using Auto-encoder LSTM for detecting rare and anomalous behaviors in customer transaction sequences.

## How to Use
1. Clone the repository:
