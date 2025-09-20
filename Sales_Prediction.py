import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Utility functions
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def save_fig(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def load_and_explore(file_path='Advertising.csv'):
    """Load data and print basic info"""
    df = pd.read_csv(file_path, index_col=0)  # assumes first column is index
    print("Loaded Advertising dataset.")
    print(f"Shape: {df.shape}\n")
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Summary statistics:")
    print(df.describe().round(3), "\n")
    print("Correlation with Sales:")
    print(df.corr()['Sales'].sort_values(ascending=False).round(3), "\n")
    return df

# Visualizations
def visualize_and_save(df, save_dir):
    """Make scatter plots and correlation heatmap and save them"""
    ensure_dir(save_dir)
    fig_paths = {}

    # Pair scatter: TV, Radio, Newspaper vs Sales
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, col in zip(axes, ['TV', 'Radio', 'Newspaper']):
        ax.scatter(df[col], df['Sales'], alpha=0.6)
        ax.set_xlabel(col)
        ax.set_ylabel('Sales')
        ax.set_title(f'{col} vs Sales')
    fig_paths['spend_vs_sales'] = os.path.join(save_dir, 'spend_vs_sales.png')
    save_fig(fig, fig_paths['spend_vs_sales'])

    # Correlation heatmap
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    fig_paths['corr_matrix'] = os.path.join(save_dir, 'correlation_matrix.png')
    save_fig(fig, fig_paths['corr_matrix'])

    # Distribution of Sales
    fig = plt.figure(figsize=(6,4))
    sns.histplot(df['Sales'], bins=20, kde=True)
    plt.title('Sales Distribution')
    fig_paths['sales_dist'] = os.path.join(save_dir, 'sales_distribution.png')
    save_fig(fig, fig_paths['sales_dist'])

    print("Saved exploratory plots.")
    return fig_paths

# Train models
def train_models(X_train, y_train, X_test, use_scaler=False):
    """Train the set of models. Return dict: name -> fitted model"""
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    trained = {}
    for name, model in models.items():
        # For Ridge & Lasso we expect scaled inputs; caller should pass scaled arrays
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"Trained {name}")
    return trained

# Evaluate & save metrics
def evaluate_and_save(models, X_test_dict, y_test, feature_names, save_dir):
    """
    models: dict of fitted models
    X_test_dict: dictionary mapping model name -> appropriate X_test (scaled or not)
    y_test: true y_test (1D array/Series)
    feature_names: list of features
    save_dir: folder to save results
    """
    ensure_dir(save_dir)
    results_rows = []
    preds_dir = os.path.join(save_dir, 'predictions')
    ensure_dir(preds_dir)
    plots_dir = os.path.join(save_dir, 'plots')
    ensure_dir(plots_dir)

    for name, model in models.items():
        X_test_for_model = X_test_dict[name]
        y_pred = model.predict(X_test_for_model)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Save predictions vs actual CSV
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': y_test - y_pred
        })
        pred_csv_path = os.path.join(preds_dir, f'{name}_predictions.csv')
        pred_df.to_csv(pred_csv_path, index=False)

        # Residuals plot
        fig = plt.figure(figsize=(6,4))
        plt.scatter(y_pred, y_test - y_pred, alpha=0.6)
        plt.hlines(0, min(y_pred), max(y_pred), colors='r', linestyles='--')
        plt.xlabel('Predicted Sales')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title(f'{name} Residuals')
        resid_path = os.path.join(plots_dir, f'{name}_residuals.png')
        save_fig(fig, resid_path)

        # Predicted vs Actual plot
        fig = plt.figure(figsize=(6,6))
        maxv = max(y_test.max(), y_pred.max())
        minv = min(y_test.min(), y_pred.min())
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([minv, maxv], [minv, maxv], 'r--')
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title(f'{name} Predicted vs Actual')
        pv_path = os.path.join(plots_dir, f'{name}_predicted_vs_actual.png')
        save_fig(fig, pv_path)

        # Feature importance or coefficients plot
        fi_path = ''
        if name == 'RandomForest' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fig = plt.figure(figsize=(6,4))
            sns.barplot(x=importances, y=feature_names)
            plt.title(f'{name} Feature Importances')
            fi_path = os.path.join(plots_dir, f'{name}_feature_importances.png')
            save_fig(fig, fi_path)
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            fig = plt.figure(figsize=(6,4))
            sns.barplot(x=coefs, y=feature_names)
            plt.title(f'{name} Coefficients')
            fi_path = os.path.join(plots_dir, f'{name}_coefficients.png')
            save_fig(fig, fi_path)

        # Record result
        results_rows.append({
            'Model': name,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Predictions_CSV': pred_csv_path,
            'Residuals_Plot': resid_path,
            'Predicted_vs_Actual_Plot': pv_path,
            'Feature_Importance_Plot': fi_path
        })

        # Print summary
        print(f"\n{name} performance: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        print(f"Saved predictions to: {pred_csv_path}")
        print(f"Saved plots: {resid_path}, {pv_path}, {fi_path if fi_path else 'N/A'}")

    results_df = pd.DataFrame(results_rows)
    results_csv = os.path.join(save_dir, 'advertising_model_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nSummary results saved to: {results_csv}")

    return results_df


# ROI analysis and example predictions
def roi_and_examples(models, scaler, model_names_scaled, save_dir, feature_names):
    """Run example predictions and ROI scenarios and save outputs."""
    ensure_dir(save_dir)
    examples = [
        {'TV': 200, 'Radio': 40, 'Newspaper': 30},
        {'TV': 150, 'Radio': 25, 'Newspaper': 15},
    ]
    scenarios = [
        [100, 20, 10],
        [200, 40, 20],
        [300, 60, 30]
    ]
    rows = []
    for mname, model in models.items():
        for ex in examples:
            x = np.array([[ex['TV'], ex['Radio'], ex['Newspaper']]])
            x_in = scaler.transform(x) if mname in model_names_scaled else x
            pred = model.predict(x_in)[0]
            rows.append({'Model': mname, 'TV': ex['TV'], 'Radio': ex['Radio'], 'Newspaper': ex['Newspaper'], 'Predicted_Sales': pred})

    # ROI scenarios
    roi_rows = []
    for mname, model in models.items():
        for tv, radio, news in scenarios:
            x = np.array([[tv, radio, news]])
            x_in = scaler.transform(x) if mname in model_names_scaled else x
            sales = model.predict(x_in)[0]
            spend = tv + radio + news
            roi = (sales - spend) / spend * 100
            roi_rows.append({'Model': mname, 'TV': tv, 'Radio': radio, 'Newspaper': news, 'Predicted_Sales': sales, 'Total_Spend': spend, 'ROI_percent': roi})

    ex_df = pd.DataFrame(rows)
    roi_df = pd.DataFrame(roi_rows)
    ex_path = os.path.join(save_dir, 'example_predictions.csv')
    roi_path = os.path.join(save_dir, 'roi_scenarios.csv')
    ex_df.to_csv(ex_path, index=False)
    roi_df.to_csv(roi_path, index=False)
    print(f"Saved example predictions to {ex_path}")
    print(f"Saved ROI scenarios to {roi_path}")
    return ex_df, roi_df


def main(file_path='Advertising.csv', results_dir='results'):
    # Load
    df = load_and_explore(file_path)

    # Prepare save folders
    results_dir = ensure_dir(results_dir)
    plots_dir = ensure_dir(os.path.join(results_dir, 'exploratory_plots'))

    # Exploratory visualizations
    viz_paths = visualize_and_save(df, plots_dir)

    # Features and target
    feature_cols = ['TV', 'Radio', 'Newspaper']
    X = df[feature_cols].values
    y = df['Sales'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaler for models that require scaling
    scaler = StandardScaler()
    scaler.fit(X_train)  # fit on training data

    # Prepare per-model test inputs:
    # - For Linear & RandomForest we use raw X
    # - For Ridge & Lasso we use scaled X
    X_test_dict = {
        'Linear': X_test,
        'RandomForest': X_test,
        'Ridge': scaler.transform(X_test),
        'Lasso': scaler.transform(X_test)
    }

    # Train models (we will train using appropriate X_train)
    # Fit Linear and RandomForest on raw X_train; Ridge/Lasso on scaled
    models_to_train = {}
    # Linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    models_to_train['Linear'] = lin
    # RandomForest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models_to_train['RandomForest'] = rf
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaler.transform(X_train), y_train)
    models_to_train['Ridge'] = ridge
    # Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(scaler.transform(X_train), y_train)
    models_to_train['Lasso'] = lasso

    print("\nTraining complete for all models.")

    # Evaluate & save
    results_df = evaluate_and_save(models_to_train, X_test_dict, y_test, feature_cols, results_dir)

    # Example predictions & ROI saved
    ex_df, roi_df = roi_and_examples(models_to_train, scaler, model_names_scaled=['Ridge', 'Lasso'], save_dir=results_dir, feature_names=feature_cols)

    print("\nAll done. Results folder contains:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, '').count(os.sep)
        indent = '  ' * (level)
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  - {f}")

    return results_df

if __name__ == "__main__":
    # change file_path if needed (e.g., '/content/Advertising.csv' in Colab)
    results = main(file_path='Advertising.csv', results_dir='results_advertising')
