# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the employee dataset.
 
 2.Preprocess the data by handling null values, encoding categorical features, and separating feures and target variable.
 
3.Split the dataset into training and testing sets and train a Decision Tree Classifier using entropy criterion

4.Predict employee churn and evaluate the model accuracy using performance metrics.
Program:


## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harish N
RegisterNumber: 212225220037 
*/
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)

try:
    df = pd.read_csv(r"C:\Users\acer\Downloads\Employee.csv")
    print("✅ Dataset loaded successfully\n")
except:
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(22, 60, 300),
        "Department": np.random.choice(["Sales","HR","IT","Finance"],300),
        "Salary": np.random.choice(["Low","Medium","High"],300),
        "Tenure": np.random.randint(1,15,300),
        "Satisfaction": np.round(np.random.rand(300),2),
        "Churn": np.random.choice([0,1],300,p=[0.7,0.3])
    })
    print("⚠ Dataset not found. Using sample dataset.\n")

print(df.head())

df.columns = df.columns.str.strip()
print("\nColumns:", df.columns.tolist())

possible_targets = ["Churn","Attrition","LeaveOrNot","left","Exited","Status"]

target = None
for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    raise ValueError("❌ Target column not found in dataset")

print("✅ Target Column:", target)

X = df.drop(target, axis=1)
y = df[target]

if y.dtype == "object":
    y = y.map({"Yes":1, "No":0})

numeric_features = X.select_dtypes(include=["int64","float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

dt = DecisionTreeClassifier(random_state=42)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", dt)
])
param_grid = {
    "classifier__criterion": ["gini","entropy"],
    "classifier__max_depth": [3,5,7,None],
    "classifier__min_samples_split": [2,5,10]
}

grid = GridSearchCV(clf, param_grid, cv=5,
                    scoring="accuracy", n_jobs=-1)

grid.fit(X, y)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stay","Churn"],
            yticklabels=["Stay","Churn"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

if hasattr(best_model.named_steps["classifier"], "predict_proba"):
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()

    auc = roc_auc_score(
        y_test,
        best_model.predict_proba(X_test)[:,1]
    )
    print("ROC AUC:", auc)

final_tree = best_model.named_steps["classifier"]

ohe = best_model.named_steps["preprocessor"]\
        .named_transformers_["cat"]\
        .named_steps["onehot"]

cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_names])

plt.figure(figsize=(14,8))
plot_tree(final_tree,
          feature_names=feature_names,
          class_names=["Stay","Churn"],
          filled=True,
          fontsize=8)

plt.title("Decision Tree - Employee Churn")
plt.show()
```

## Output:
<img width="930" height="811" alt="Screenshot 2026-03-09 134427" src="https://github.com/user-attachments/assets/bd50066f-8d1b-4df2-b5b0-cdd72e88701e" />
<img width="1007" height="650" alt="Screenshot 2026-02-14 204954" src="https://github.com/user-attachments/assets/3164c610-ce25-4922-b499-94aa9abe826f" />
<img width="1110" height="830" alt="Screenshot 2026-03-09 134458" src="https://github.com/user-attachments/assets/33a5f273-a76e-4763-ab22-ebeb7ff8012f" />
<img width="1050" height="565" alt="Screenshot 2026-03-09 134517" src="https://github.com/user-attachments/assets/41baebf0-067f-4e95-8af3-915df471d1cf" />





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
