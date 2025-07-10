import numpy as np
import wandb
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
import mrmr
from xgboost import XGBClassifier
from bartpy2.sklearnmodel import SklearnModel

from torch.utils.data import DataLoader
from dataloaders.dataloader import FCMatrixDataset



def evaluate_model(model, X, y):
    """
	Model types: 
		- SVC (sklearn SVM)
		- XGBClassifier (XGBoost)
		- bartpy2 (sklearn based BART classifier, else-case)
    """
    if isinstance(model, XGBClassifier):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

    elif isinstance(model, SVC):
        y_pred = model.predict(X)
        # Check if probability=True was set
        if hasattr(model, 'probability') and model.probability:
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            # Use decision function if probabilities not available
            y_pred_proba = model.decision_function(X)
            # Scale to [0,1] range
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))

    elif isinstance(model, SklearnModel):
        # BART returns probabilities directly
        y_pred_proba = model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

    else:
        raise ValueError("Unknown model type")

    return {
        'auc_roc': roc_auc_score(y, y_pred_proba),
        'accuracy': accuracy_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    }

def init_model(model_name, config):
    if model_name == "xgboost":
        return XGBClassifier(**config, eval_metric='logloss')
    elif model_name == "svm":
        # Ensure probability=True for SVC
        return SVC(**config, probability=True)
    elif model_name == "bart":
        return SklearnModel(**config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_inner_cv_evaluation(config, X_train, y_train, model_name, seed=42):
    """
    Evaluate a single configuration across all inner folds
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []

    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Initialize and train model
        model = init_model(model_name, config)
        model.fit(X_fold_train, y_fold_train)
        metrics = evaluate_model(model, X_fold_val, y_fold_val)
        fold_scores.append(metrics['auc_roc'])

    # Calculate average score across folds
    mean_score = np.mean(fold_scores)
    return mean_score

def run_sweep_and_get_best_config(X_train, y_train, model_name, sweep_config, project_name):
    """
    Run sweep and track best configuration
    """
    best_score = -np.inf
    best_config = None

    def sweep_train():
        nonlocal best_score, best_config
        with wandb.init() as run:
            score = run_inner_cv_evaluation(
                wandb.config, X_train, y_train, model_name
            )
            if score > best_score:
                best_score = score
                best_config = dict(wandb.config)
            wandb.log({"avg_roc_auc": score})

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, sweep_train, count=10)

    return best_config

def run_nested_cv(X, y, model_names, sweep_configs, project_name, seed=42):
    """
    Run nested CV sequentially with optional feature selection
    """
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    results_df = pd.DataFrame(columns=[
        'model', 'fold', 'auc_roc', 'accuracy', 'f1', 'n_features'
    ])

    best_models = {model: [] for model in model_names}
    selected_features = {model: [] for model in model_names}

    # Outer CV loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\nProcessing outer fold {fold_idx + 1}/5")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Process each model sequentially
        for model_name in model_names:
            print(f"\nProcessing {model_name}")

            best_config = run_sweep_and_get_best_config(
                X_train,
                y_train,
                model_name,
                sweep_configs[model_name],
                project_name
            )

            # Train final model with best config
            best_model = init_model(model_name, best_config)
            best_model.fit(X_train, y_train)

            # Evaluate on test set
            test_metrics = evaluate_model(best_model, X_test, y_test)

            # Store results
            best_models[model_name].append(best_model)

            # Add to DataFrame
            results_df = pd.concat([results_df, pd.DataFrame({
                'model': [model_name],
                'fold': [fold_idx],
                'auc_roc': [test_metrics['auc_roc']],
                'accuracy': [test_metrics['accuracy']],
                'f1': [test_metrics['f1']],
                'n_features': [n_features]
            })], ignore_index=True)

            print(f"\nResults for {model_name} - Fold {fold_idx + 1}:")
            print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"F1-Score: {test_metrics['f1']:.4f}")
            print(f"Number of features: {n_features}")

    # Log final results
    with wandb.init(project=project_name, job_type="analysis"):
        for model_name in model_names:
            model_results = results_df[results_df['model'] == model_name]

            summary_stats = {
                'auc_roc': {
                    'mean': model_results['auc_roc'].mean(),
                    'std': model_results['auc_roc'].std()
                },
                'accuracy': {
                    'mean': model_results['accuracy'].mean(),
                    'std': model_results['accuracy'].std()
                },
                'f1': {
                    'mean': model_results['f1'].mean(),
                    'std': model_results['f1'].std()
                },
                'n_features': {
                    'mean': model_results['n_features'].mean(),
                    'std': model_results['n_features'].std()
                }
            }

            print(f"\nFinal results for {model_name}:")
            print(f"AUC-ROC: {summary_stats['auc_roc']['mean']:.4f} ± {summary_stats['auc_roc']['std']:.4f}")
            print(f"Accuracy: {summary_stats['accuracy']['mean']:.4f} ± {summary_stats['accuracy']['std']:.4f}")
            print(f"F1-Score: {summary_stats['f1']['mean']:.4f} ± {summary_stats['f1']['std']:.4f}")
            print(f"Avg Features: {summary_stats['n_features']['mean']:.1f} ± {summary_stats['n_features']['std']:.1f}")

            wandb.log({
                f"{model_name}_summary": summary_stats,
                f"{model_name}_detailed_results": model_results.to_dict()
            })

    return results_df, best_models


# Separate sweep configurations
sweep_configs = {
    'xgboost': {
        'method': 'random',
        'metric': {'name': 'avg_roc_auc',
                   'goal': 'maximize'},
        'parameters': {
            'mrmr_n': {'min': 8, 'max': 80},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.01, 'max': 0.25},
            'min_child_weight': {'min': 1, 'max': 10},
            'max_depth': {'min': 2, 'max': 8},
            'n_estimators': {'min': 50, 'max': 150},
            'subsample': {'min': 0.6, 'max': 0.9},
            'colsample_bytree': {'min': 0.6, 'max': 0.9},
            'gamma': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 1.0},
            'reg_lambda': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 5.0},
            'reg_alpha': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 5.0},
        }
    },
    'svm': {
        'method': 'random',
        'metric': {'name': 'avg_roc_auc',
                   'goal': 'maximize'},
        'parameters': {
            'mrmr_n': {'min': 8, 'max': 80},
            'degree': {'values': [2, 3]},
            'C': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 50.0},
            'kernel': {'values': ['rbf', 'linear', 'poly']},
            'gamma': {'distribution': 'log_uniform_values', 'min': 0.00001, 'max': 1.0},
        }
    },
    'bart': {
        'method': 'random',
        'metric': {'name': 'avg_roc_auc',
                   'goal': 'maximize'},
        'parameters': {
            'mrmr_n': {'min': 8, 'max': 80},
            'n_trees': {'min': 20, 'max': 300},
            'n_samples': {'min':150, 'max': 300},
            'alpha': {'min': 0.6, 'max': 0.9},
            'beta': {'min': 1.0, 'max': 3.0},
            'p_grow': {'min': 0.3, 'max':1.0},
            'sigma_a': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 8.0},
            'sigma_b': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 8.0},
        }
    }
}

if __name__ == "__main__":

    label_dir = "/path/to/data/labels.csv"
    data_dir = '/path/to/data' 
    save_dir = "/path/to/sweep_results.csv" # Set the save file location for final performance metrics

    project_name = "test project" # Set the name of your project, will be used for logs on wandb

    seed = 42
    np.random.seed(seed)

    model_names = ['svm', 'xgboost', 'bart']

    udi = "25751" # Only relevant for loading UKB FC data using a dataloader
    data = FCMatrixDataset(label_dir, data_dir, udi)
    full_loader = DataLoader(data, batch_size=len(data))
    X, y = next(iter(full_loader))

    X = X.numpy()
    y = y.numpy()
    print(X.shape)
    print(y.shape)

    results_df, best_models = run_nested_cv(X, y, model_names, sweep_configs, project_name, seed=seed)
    results_df.to_csv(save_dir)


