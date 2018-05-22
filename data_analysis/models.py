from multiprocessing import cpu_count
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix

# Settings

NCORES = cpu_count() - 1

# Outer cross validation
FOLDS = 5

# Number of inner folds on the training set for hyperparameter tuning
INNER_FOLDS = 3


def estimate_model(model, name: str, param_grid: dict,
                   X_train: np.array, y_train: np.array,
                   X_test: np.array, y_test: np.array, i: int,
                   hyper_param_tuning: bool = True):
    """
    Wrapper for Model Estimation with GridSearchCV.
    Returns model result dictionary with standard metrics.
    """
    scoring = 'accuracy'

    if hyper_param_tuning:
        model = GridSearchCV(model, param_grid=param_grid, cv=INNER_FOLDS,
                             scoring=scoring, n_jobs=NCORES, verbose=1)
    else:
        if param_grid:
            param_grid = {key: param_grid[key][0] for key in param_grid}
            model = model(**param_grid)
        else:
            model = model()

    model.fit(X=X_train, y=y_train)

    pred = model.predict(X=X_test)
    pred_proba = model.predict_proba(X=X_test)

    result = {
        'run': i,
        'model': name,
        'model_obj': model,
        'param_grid': param_grid,
        'xtest': X_test,
        'preds': pred,
        'pred_proba': pred_proba,
        'test_accuracy': accuracy_score(y_true=y_test, y_pred=pred),
        'test_f1': f1_score(y_true=y_test, y_pred=pred),
        'test_precision': precision_score(y_true=y_test, y_pred=pred),
        'test_recall': recall_score(y_true=y_test, y_pred=pred),
        'test_roc_auc': roc_auc_score(y_true=y_test, y_score=pred_proba[:, 1]),
        'roc': roc_curve(y_true=y_test, y_score=pred_proba[:, 1])
    }
    if hyper_param_tuning:
        result.update({
            'results': model.cv_results_,
            'best_params': model.best_params_,
        })

    return result


def estimate_model_multiclass(model, name: str, param_grid: dict,
                              X_train: np.array, y_train: np.array,
                              X_test: np.array, y_test: np.array, i: int,
                              hyper_param_tuning: bool = True):
    """
    Wrapper for Model Estimation with GridSearchCV.
    Returns model result dictionary with standard metrics.
    """
    scoring = 'accuracy'

    if hyper_param_tuning:
        model = GridSearchCV(model(), param_grid=param_grid, cv=INNER_FOLDS,
                             scoring=scoring, n_jobs=NCORES, verbose=1)
    else:
        if param_grid:
            param_grid = {key: param_grid[key][0] for key in param_grid}
            model = model(**param_grid)
        else:
            model = model()

    model.fit(X=X_train, y=y_train)

    pred = model.predict(X=X_test)
    #pred_proba = model.predict_proba(X=X_test)

    result = {
        'run': i,
        'model': name,
        'model_obj': model,
        'param_grid': param_grid,
        'xtest': X_test,
        'preds': pred,
        'test_accuracy': accuracy_score(y_true=y_test, y_pred=pred),
        'test_f1': f1_score(y_true=y_test, y_pred=pred, average='micro'),
        'test_precision': precision_score(y_true=y_test, y_pred=pred, average='micro'),
        'test_recall': recall_score(y_true=y_test, y_pred=pred, average='micro'),
    }
    if hyper_param_tuning:
        result.update({
            'results': model.cv_results_,
            'best_params': model.best_params_,
        })

    return result

