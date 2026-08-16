# Iris Flower Classification — PRD

## 1. Project Overview

**Project Name:** Iris Flower Classification  
**Project Type:** Machine Learning Classification Application  
**Domain:** Artificial Intelligence / Machine Learning  
**Target Users:** Students, beginners learning machine learning, educators, and users interested in flower species prediction.

### Objective

Build a machine learning system that predicts the species of an iris flower based on its physical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model should classify an iris flower into one of three species:

- Setosa
- Versicolor
- Virginica

The project should include complete data analysis, visualization, model training, evaluation, comparison, and identification of the best-performing classifier.

---

## 2. Problem Statement

Manually identifying iris flower species from physical measurements can be difficult for beginners. This project uses machine learning classification algorithms to automatically predict the species based on four measurable features.

---

## 3. Goals

- Load the Iris dataset using `sklearn.datasets.load_iris()`.
- Understand and analyze the dataset using EDA.
- Visualize relationships between features and species.
- Identify the most important/discriminative features.
- Train multiple classification algorithms.
- Compare model performance.
- Select the best-performing model.
- Provide accurate predictions for new flower measurements.
- Maintain clean, well-commented code.

---

## 4. Dataset

The Iris dataset is available directly through scikit-learn.

**Source:** `sklearn.datasets.load_iris()`

### Dataset Details

- Total samples: 150
- Features: 4
- Classes: 3
- Samples per class: 50

### Features

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal |
| Sepal Width | Width of the sepal |
| Petal Length | Length of the petal |
| Petal Width | Width of the petal |

### Target Classes

| Class | Species |
|---|---|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

---

## 5. Technology Stack

- **Programming Language:** Python
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook
- **Optional UI:** Streamlit
- **Version Control:** Git and GitHub

---

## 6. Functional Requirements

### FR-01: Dataset Loading

The system must load the Iris dataset using:

`sklearn.datasets.load_iris()`

No external dataset download should be required.

### FR-02: Data Exploration

The notebook must display:

- Dataset shape
- Column names
- Data types
- Null-value information
- Descriptive statistics
- Class distribution

### FR-03: Data Visualization

The project must include:

- Pairplot/scatter matrix
- Feature distribution plots
- Box plots
- Species-based visualizations

### FR-04: Feature Analysis

The project should discuss which features provide the strongest separation between species.

Special attention should be given to petal length and petal width because they generally provide strong class separation in this dataset.

### FR-05: Data Splitting

The dataset must be divided into training and testing sets.

Recommended split:

- Training: 80%
- Testing: 20%

`train_test_split()` should be used.

### FR-06: Model Training

At least two classification algorithms must be trained.

Recommended models:

1. Logistic Regression
2. K-Nearest Neighbours
3. Decision Tree
4. Random Forest

The project may train all four for a stronger comparison.

### FR-07: Model Evaluation

Each model must be evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report

### FR-08: Model Comparison

A comparison table should be created containing the performance of each model.

Example:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | — | — | — | — |
| KNN | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |

### FR-09: Best Model Selection

The model with the strongest overall test performance should be declared the best-performing model.

The decision must be supported using evaluation metrics rather than assumptions.

### FR-10: Prediction

The final model should be capable of predicting the species of a new iris flower from four measurements.

Example input:

```text
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

Expected output:

```text
Predicted Species: Setosa
```

---

## 7. Optional Professional Application

To make the project look like a real application instead of only a notebook, an optional Streamlit interface can be added.

### Application Features

- Professional dashboard
- Input fields/sliders for four measurements
- Predict button
- Predicted species display
- Model accuracy display
- Dataset statistics
- Visualizations
- Model comparison
- Responsive layout

### Application Flow

```text
User
  ↓
Enter Flower Measurements
  ↓
Validate Input
  ↓
Load Trained Model
  ↓
Make Prediction
  ↓
Display Iris Species
```

---

## 8. Non-Functional Requirements

### Performance

Prediction should be generated quickly after the user submits measurements.

### Usability

The interface should be simple and understandable for beginners.

### Maintainability

Code should be divided into logical sections/files with comments and meaningful variable names.

### Reliability

Invalid or missing inputs should be handled properly.

### Accessibility

The application should be usable on common desktop and mobile browsers if deployed as a web application.

### Security

No sensitive user information should be collected or stored.

---

## 9. Notebook Structure

```text
01. Project Introduction
02. Import Libraries
03. Load Dataset
04. Dataset Information
05. Exploratory Data Analysis
06. Data Cleaning / Validation
07. Data Visualization
08. Feature Analysis
09. Train/Test Split
10. Model Training
11. Model Evaluation
12. Model Comparison
13. Best Model Selection
14. Sample Predictions
15. Conclusion
```

---

## 10. Project Deliverables

- `Iris_Flower_Classification.ipynb`
- Clean Python source files if applicable
- Trained model file
- Visualization outputs
- Model comparison results
- README documentation
- Optional Streamlit application
- `requirements.txt`
- GitHub repository

---

## 11. Success Criteria

The project is considered complete when:

- [ ] Iris dataset is successfully loaded.
- [ ] EDA is completed.
- [ ] Null values and data types are checked.
- [ ] Required visualizations are included.
- [ ] Feature importance/discrimination is discussed.
- [ ] Train/test split is implemented.
- [ ] At least two classifiers are trained.
- [ ] Accuracy is calculated for every model.
- [ ] Confusion matrices are generated.
- [ ] Classification reports are generated.
- [ ] Models are compared.
- [ ] Best-performing model is clearly identified.
- [ ] New flower measurements can be classified.
- [ ] Notebook is clean and commented.
- [ ] Optional web application works correctly.
- [ ] Project can be shared through GitHub and, if deployed, through a public URL.