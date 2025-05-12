# ✅ universal_model_pipeline.py (Updated for classification + regression + XGBoost + SMOTE)
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, r2_score, mean_squared_error)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("static/charts", exist_ok=True)
    path = f"static/charts/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_regression_results(y_true, y_pred):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Regression Results")
    os.makedirs("static/charts", exist_ok=True)
    path = "static/charts/regression_plot.png"
    plt.savefig(path)
    plt.close()
    return path


def train_best_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    task_type = 'classification' if y.nunique() <= 10 else 'regression'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == 'classification':
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC(probability=True)
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestClassifier(),
            "Gradient Boosting Regressor": GradientBoostingClassifier()
        }

    best_model = None
    best_model_name = None
    best_score = -float('inf')
    best_report = None
    best_plot = None
    model_table = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == 'classification':
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            score = f1
            report = classification_report(y_test, y_pred, output_dict=True)
            cm_path = plot_conf_matrix(y_test, y_pred, name)

            model_table.append({
                "Model": name,
                "Accuracy": round(acc * 100, 2),
                "Macro F1": round(f1 * 100, 2),
                "Confusion Matrix": cm_path
            })

        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            score = r2
            report = {
                "r2_score": r2,
                "mse": mse
            }
            cm_path = plot_regression_results(y_test, y_pred)

            model_table.append({
                "Model": name,
                "R2 Score": round(r2, 4),
                "MSE": round(mse, 2),
                "Regression Plot": cm_path
            })

        if score > best_score:
            best_model = model
            best_model_name = name
            best_score = score
            best_report = report
            best_plot = cm_path

    # 🔥 GridSearch on XGBoost
    print("🔍 Running GridSearch on XGBoost...")
    if task_type == 'classification':
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        xgb_best = grid.best_estimator_
        y_pred = xgb_best.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        cm_path = plot_conf_matrix(y_test, y_pred, "XGBoost_Tuned")

        model_table.append({
            "Model": "XGBoost (Tuned)",
            "Accuracy": round(acc * 100, 2),
            "Macro F1": round(f1 * 100, 2),
            "Confusion Matrix": cm_path
        })

        if f1 > best_score:
            best_model = xgb_best
            best_model_name = "XGBoost (Tuned)"
            best_score = f1
            best_report = report
            best_plot = cm_path

        best_params = grid.best_params_

    else:
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        report = {
            "r2_score": r2,
            "mse": mse
        }
        cm_path = plot_regression_results(y_test, y_pred)

        model_table.append({
            "Model": "XGBoost",
            "R2 Score": round(r2, 4),
            "MSE": round(mse, 2),
            "Regression Plot": cm_path
        })

        if r2 > best_score:
            best_model = xgb
            best_model_name = "XGBoost"
            best_score = r2
            best_report = report
            best_plot = cm_path

        best_params = "Default (no tuning)"

    # ✅ Save best model
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model, {
        "Best Model": best_model_name,
        "Best Score": round(best_score * 100, 2) if task_type == 'classification' else round(best_score, 4),
        "Evaluation Report": best_report,
        "Plot Path": best_plot,
        "Comparison Table": model_table,
        "Best Parameters": best_params,
        "Task Type": task_type,
        "Model File": "outputs/model.pkl"
    }