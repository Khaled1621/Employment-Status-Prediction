
# Employment Status Prediction Using Machine Learning

## Project Overview

This project aims to predict an individual's employment status (Employed or Unemployed) based on their personality traits, demographic information, and lifestyle factors. The dataset includes various features like age, exercise frequency, social media usage, and responses to personality assessment questions.

We implemented and evaluated several machine learning models to achieve this goal, including Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and Neural Networks.

## Dataset

- **Source**: `2024_PersonalityTraits_SurveyData.csv`
- **Target Variable**: `Employment Status`  
  - Encoded as:
    - **1**: Employed
    - **0**: Unemployed

### Features

- **Personality Traits** (e.g., extraversion, conscientiousness, etc.)
- **Demographic Information** (e.g., age, gender, marital status)
- **Lifestyle Factors** (e.g., exercise frequency, social media usage)

## Project Workflow

1. **Data Preprocessing**:
    - Handling missing values by dropping rows and columns with more than 40% missing data.
    - Encoding categorical features using `LabelEncoder`.
    - Standardizing numerical features with `StandardScaler`.

2. **Model Selection**:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Machine (SVM)
    - Neural Network (MLP Classifier)

3. **Data Splitting**:
    - **Training Set**: 60%
    - **Validation Set**: 20%
    - **Test Set**: 20%

4. **Hyperparameter Tuning**:
    - Applied `GridSearchCV` for Gradient Boosting to find optimal parameters.

5. **Evaluation Metrics**:
    - Accuracy on training, validation, and test sets.
    - Analyzed overfitting and underfitting for each model.

## Results Summary

| Model                | Train Accuracy | Validation Accuracy | Test Accuracy |
|----------------------|----------------|---------------------|---------------|
| **Logistic Regression** | 85.04%         | 64.29%              | 74.42%        |
| **Random Forest**       | 100%           | 85.71%              | 88.37%        |
| **Gradient Boosting**   | 100%           | 97.62%              | 97.67%        |
| **SVM**                 | 62.20%         | 45.24%              | 55.81%        |
| **Neural Network**      | 100%           | 59.52%              | 72.09%        |

### Observations:
- **Gradient Boosting** achieved the highest accuracy with 97.67% on the test set.
- **Random Forest** performed well but showed signs of overfitting.
- **SVM** and **Neural Network** struggled with underfitting or overfitting.

## Dependencies

Install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run the Project

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/employment-status-prediction.git
    cd employment-status-prediction
    ```

2. **Run the Code**:

    Execute the Python script to train models and visualize results:

    ```bash
    python employment_status_prediction.py
    ```

3. **Modify Hyperparameters**:

    Edit the script to adjust model hyperparameters for further tuning.

## Directory Structure

```
employment-status-prediction/
│-- 2024_PersonalityTraits_SurveyData.csv
│-- README.md
│-- employment_status_prediction.ipynb
│-- employment_status_prediction.py
```

## Future Improvements

- **Feature Engineering**: Create new features to improve model performance.
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation.
- **Reduce Overfitting**: Apply techniques like regularization, dropout (for Neural Networks), and early stopping.
- **Additional Models**: Experiment with other models like XGBoost and LightGBM.


## Contact

- **Name**: Hamza Atout and Khaled Ammoura  
- **Email**: hsa60@mail.aub.edu, kaa74@mail.aub.edu 
