import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load the dataset
file_path = '2024_PersonalityTraits_SurveyData.csv'
df = pd.read_csv(file_path)

# Drop rows and columns with excessive missing values
df.dropna(axis=0, thresh=len(df.columns) * 0.6, inplace=True)  # Drop rows with >40% missing values
df.dropna(axis=1, thresh=len(df) * 0.6, inplace=True)          # Drop columns with >40% missing values

# Reset index after dropping
df.reset_index(drop=True, inplace=True)


# Set 'Employment Status' as the target variable
target = 'Employment Status'

# Encode the target variable: 'Employed' -> 1, 'Unemployed' -> 0
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Display the target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df)
plt.title('Target Distribution: Employment Status')
plt.xlabel('Employment Status (1: Employed, 0: Unemployed)')
plt.ylabel('Count')
plt.show()


# Drop unnecessary columns
drop_columns = ['Unnamed: 0', 'Last page']
df.drop(columns=drop_columns, inplace=True)

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Encode categorical columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Scale numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data: 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "Neural Network": MLPClassifier(max_iter=1000)
}

# Train and evaluate models on train, val, and test sets
model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Accuracy on training set
    y_train_pred = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    
    # Accuracy on validation set
    y_val_pred = model.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    
    # Accuracy on testing set
    y_test_pred = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    model_results[name] = {'Train Accuracy': acc_train, 'Validation Accuracy': acc_val, 'Test Accuracy': acc_test}
    print(f"{name}: Train Accuracy = {acc_train:.4f}, Validation Accuracy = {acc_val:.4f}, Test Accuracy = {acc_test:.4f}")

# Convert model_results to a DataFrame for better visualization
results_df = pd.DataFrame(model_results).T

# Display the results DataFrame
print("\nModel Performance Summary:")
print(results_df)

# Visualize model performance
results_df.plot(kind='bar', figsize=(12, 7))
plt.title("Model Comparison: Train, Validation, and Test Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()


# Hyperparameter tuning for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model and accuracy
best_gb_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred_best = best_gb_model.predict(X_test)
best_gb_accuracy = accuracy_score(y_test, y_pred_best)

print(f"Best Parameters: {best_params}")
print(f"Best Gradient Boosting Accuracy: {best_gb_accuracy:.4f}")