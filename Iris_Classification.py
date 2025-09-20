import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Load dataset
# =========================
def load_iris_data(file_path):
    """Load the Iris dataset from CSV file"""
    df = pd.read_csv("/Iris.csv")
    return df


# =========================
# Explore dataset
# =========================
def explore_data(df):
    """Explore the dataset characteristics"""
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Statistics:")
    print(df.describe())

    print("\nSpecies Distribution:")
    print(df['Species'].value_counts())

    return df


# =========================
# Preprocess data
# =========================
def preprocess_data(df):
    """Prepare data for machine learning"""
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    return X, y


# =========================
# Train models
# =========================
def train_models(X_train, y_train):
    """Train classification models"""
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


# =========================
# Evaluate models & Save Results
# =========================
def evaluate_models(models, X_test, y_test, save_dir="results"):
    """Evaluate models, save metrics + graphs into CSV & PNGs"""
    os.makedirs(save_dir, exist_ok=True)
    results_list = []

    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Save confusion matrix
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join(save_dir, f"{name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Save accuracy bar graph
        plt.figure(figsize=(4,3))
        plt.bar([name], [accuracy], color="green")
        plt.ylim(0,1)
        plt.title(f"{name} Accuracy")
        plt.ylabel("Accuracy")
        acc_path = os.path.join(save_dir, f"{name}_accuracy.png")
        plt.savefig(acc_path)
        plt.close()

        # Store metrics
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                results_list.append({
                    "Model": name,
                    "Class": label,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1-score"],
                    "Support": metrics["support"],
                    "Accuracy": accuracy,
                    "Confusion_Matrix_Path": cm_path,
                    "Accuracy_Plot_Path": acc_path
                })

        # Print summary
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    # Save results into CSV
    df_results = pd.DataFrame(results_list)
    csv_path = os.path.join(save_dir, "iris_model_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults and graphs saved in: {save_dir}/")

    return df_results


# =========================
# Feature importance (RF only)
# =========================
def plot_feature_importance(model, feature_names, save_dir="results"):
    """Plot and save feature importance for Random Forest"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(8,5))
        plt.title('Feature Importance (Random Forest)')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')

        save_path = os.path.join(save_dir, "random_forest_feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Feature importance plot saved: {save_path}")


# =========================
# Make predictions on new samples
# =========================
def predict_new_sample(models, scaler, sample_data):
    """Predict species for a new sample"""
    sample_scaled = scaler.transform([sample_data])
    print(f"\nNew Sample: {sample_data}")
    predictions = {}
    for name, model in models.items():
        pred = model.predict(sample_scaled)[0]
        predictions[name] = pred
        print(f"{name}: {pred}")
    return predictions


# =========================
# Main function
# =========================
def main():
    # 1. Load data
    df = load_iris_data("Iris.csv")

    # 2. Explore
    explore_data(df)

    # 3. Preprocess
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train models
    models = train_models(X_train_scaled, y_train)

    # 6. Evaluate & Save results
    results_df = evaluate_models(models, X_test_scaled, y_test)

    # 7. Random Forest feature importance
    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"],
                                feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

    # 8. Predictions on new samples
    new_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Expected Iris-setosa
        [6.2, 2.9, 4.3, 1.3],  # Expected Iris-versicolor
        [6.3, 3.3, 6.0, 2.5]   # Expected Iris-virginica
    ]
    for sample in new_samples:
        predict_new_sample(models, scaler, sample)

    print("\nIris Classification Pipeline Complete!")


if __name__ == "__main__":
    main()
