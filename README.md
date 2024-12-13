# LendingClub Loan Prediction Using Deep Neural Networks (DNN)

![Deep Learning](https://img.shields.io/badge/Machine%20Learning-Deep%20Neural%20Networks-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📄 Project Overview

This project explores the application of **Deep Neural Networks (DNNs)** for predicting loan defaults using the LendingClub dataset. By leveraging advanced machine learning techniques and addressing class imbalance with **SMOTE**, this project aims to enhance credit risk modeling and support financial decision-making.

Key highlights:
- Development of multiple DNN architectures to identify patterns in high-dimensional, imbalanced financial datasets.
- Implementation of hyperparameter tuning to optimize model performance.
- Final tuned architecture achieves a **98.17% test accuracy**, outperforming traditional methods and previous research.

## 🔑 Key Features

- **Multiple DNN Architectures:** Experiments with architectures of varying complexity, including dropout layers to prevent overfitting.
- **Class Imbalance Handling:** Use of Synthetic Minority Over-sampling Technique (**SMOTE**) to balance the dataset.
- **Comprehensive Evaluation:** Metrics such as Accuracy, Precision, Recall, F1-Score, and detailed classification reports.
- **Visualization:** Correlation heatmaps, FICO score analysis, and learning curves for insights into data and model performance.

## 📊 Results

The project demonstrates the power of **Deep Neural Networks** in handling complex, non-linear relationships in financial datasets. Below is the performance summary of the best architecture:

| Architecture | Optimizer | Epochs | Test Accuracy | Precision | Recall | F1-Score |
|--------------|-----------|--------|---------------|-----------|--------|----------|
| Tuned DNN 3  | Adam      | 100    | **98.17%**    | 0.98      | 0.98   | 0.98     |

## 🗂 Dataset

The project uses the **LendingClub dataset**, a publicly available dataset containing detailed loan and borrower information. Key features include:
- Borrower FICO scores
- Loan amounts and interest rates
- Credit policies and repayment statuses

### Preprocessing Steps:
1. **Feature Engineering:** One-hot encoding of categorical variables such as loan purposes.
2. **Normalization:** Standardizing continuous features for better model training.
3. **Balancing Classes:** Using SMOTE to handle imbalances in loan repayment statuses.

## 📀 Model Architectures

### Architecture 3 (Best Performing Model):
- **Input Layer**: Standardized feature set.
- **Hidden Layers**:
  - 1024 neurons, ReLU activation, Dropout (0.3)
  - 512 neurons, ReLU activation, Dropout (0.3)
  - 256 neurons, ReLU activation, Dropout (0.3)
  - 128 neurons, ReLU activation, Dropout (0.3)
- **Output Layer**: 2 neurons with SoftMax activation.
- **Optimizer**: Adam with hyperparameter tuning.
- **Loss Function**: Categorical Crossentropy.

## 🔧 Tools and Libraries

- **Deep Learning Frameworks:** TensorFlow, Keras
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Class Imbalance Handling:** imbalanced-learn (SMOTE)
- **Evaluation:** Scikit-learn

## 🎯 Objectives

1. Build and evaluate DNN architectures for loan default prediction.
2. Address class imbalance effectively using SMOTE.
3. Optimize hyperparameters to maximize model performance.

## 📈 Usage

### Step 1: Clone the Repository
Clone the GitHub repository to your local machine.
```bash
git clone https://github.com/ADNAN-BAVA/loan-prediction-dnn.git
cd loan-prediction-dnn
```

### Step 2: Install Required Libraries
Ensure you have Python 3.8+ installed. Then, install all required dependencies using:
```bash
pip install -r requirements.txt
```

### Step 3: Perform Exploratory Data Analysis (EDA)
Run the `eda.ipynb` notebook in the `notebooks/` directory to visualize and understand the dataset. 
```bash
jupyter notebook Loan_preddiction.ipynb
```

### Step 4: Preprocess the Dataset
Run the preprocessing script to prepare the data for modeling. This script performs:
- Handling class imbalances using SMOTE.
- Normalizing continuous features.
- Encoding categorical variables.

### Step 5: Train the Model
Execute the training script to train the selected DNN architecture. Use the command below to specify the configuration.

### Step 6: Evaluate the Model
Run the evaluation script to generate a classification report and plots for accuracy, precision, recall, and F1-score.

### Step 7: Hyperparameter Tuning
Use the hyperparameter tuning script to find the best model configuration. Adjust the parameters in the script file before running.

### Step 8: Visualize Results
Access saved results like training-validation plots and classification reports in the `results/` directory.

## 🏆 Contributions

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and make improvements.

## 📧 Contact

- **Author**: Mohammad Adnan
- **Email**: adnan.bava123@gmail.com

---

**Keywords**: Loan Default Prediction, Deep Neural Networks, Credit Risk Analysis, SMOTE, Financial Data Modeling
