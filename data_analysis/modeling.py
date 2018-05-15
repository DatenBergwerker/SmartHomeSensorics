import sys
import logging
from datetime import datetime
from math import sqrt
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Settings

NCORES = cpu_count() - 1

# Outer cross validation
FOLDS = 5

# Number of inner folds on the training set for hyperparameter tuning
INNER_FOLDS = 3


# Function Definitions
def estimate_model(model, name: str, param_grid: dict,
                   X_train: np.array, y_train: np.array,
                   X_test: np.array, y_test: np.array, i: int):
    """
    Wrapper for Model Estimation with GridSearchCV.
    Returns model result dictionary with standard metrics.
    """
    scoring = 'accuracy'

    model = GridSearchCV(model, param_grid=param_grid, cv=INNER_FOLDS,
                         scoring=scoring, n_jobs=NCORES, verbose=1)
    model.fit(X=X_train, y=y_train)

    pred = model.predict(X=X_test)
    pred_proba = model.predict_proba(X=X_test)
    return {
        'run': i,
        'model': name,
        'model_obj': model,
        'param_grid': param_grid,
        'results': model.cv_results_,
        'best_params': model.best_params_,
        'test_accuracy': accuracy_score(y_true=y_test, y_pred=pred),
        'test_f1': f1_score(y_true=y_test, y_pred=pred),
        'test_precision': precision_score(y_true=y_test, y_pred=pred),
        'test_recall': recall_score(y_true=y_test, y_pred=pred),
        'test_roc_auc': roc_auc_score(y_true=y_test, y_score=pred_proba[:, 1]),
        'roc': roc_curve(y_true=y_test, y_score=pred_proba[:, 1])
    }


# Logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d.%m.%Y %H:%M:%S')

# Data Import
full_data = pd.read_csv(r'SmartHomeSensorics_selected_feature_matrix_cleaned.csv')

X = full_data.drop(['binary_target', 'index', 'entry_id', 'no_occupants', 'room_location',
                    'node_id', 'measurement_no', 'occupant_activity', 'relative_timestamp'], axis=1).values
y = full_data['binary_target'].values.reshape(X.shape[0], )

kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
scaler = StandardScaler()
imputer = Imputer()

# Model definitions
class_models = {
        'LogisticRegression': {'model': LogisticRegression(),
                               'params': {'fit_intercept': [False]}},
        'NaiveBayes': {'model': GaussianNB(),
                       'params': {}},
        # 'Support Vector Machine': {'model': SVC(probability=True),
        #                            'params': {
        #                                'C': [0.1, 1, 10, 100],
        #                                'kernel': ['rbf', 'poly'],
        #                                'degree': range(3, 6)}},
        'Nystroem-LinearSGD': {'model': Pipeline([('Nystroem_feature_map', Nystroem()),
                                                  ('SGD_Linear', SGDClassifier())]),
                               'params': {'SGD_Linear__fit_intercept': [False],
                                          'SGD_Linear__loss': ['log', 'modified_huber']}},
        'Random Forest Classifier': {'model': RandomForestClassifier(),
                                     'params': {
                                         'max_features': [int(sqrt(X.shape[1]))],
                                         'n_estimators': [val for val in
                                                          list(range(100, 501, 100))]}}
                }


# metric column extractor
overview_cols = [
    'run',
    'model',
    'test_accuracy',
    'test_f1',
    'test_precision',
    'test_recall',
    'test_roc_auc'
]

class_results = []
model_objs = []
class_roc = []
timings = []
logging.info(f'Model Training started.')

# main loop
for i, (train, test) in enumerate(kfold.split(X=X, y=y)):
    logging.info(f'--- Beginning iteration {i+1} of {FOLDS} ---')
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    X_train = imputer.fit_transform(X=X_train)
    X_test = imputer.transform(X=X_test)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Run model zoo
    for model in class_models:
        logging.info(f'Beginning {model} training.')
        time = datetime.now()
        results_class = estimate_model(model=class_models[model]['model'],
                                       name=model,
                                       param_grid=class_models[model]['params'],
                                       X_train=X_train,
                                       X_test=X_test,
                                       y_train=y_train,
                                       y_test=y_test,
                                       i=i)
        timings.append((i, model, (datetime.now() - time).seconds / 60))
        params = ' '.join(
            [str(key) + ' ' + str(results_class['best_params'][key])
             for key in results_class['best_params']])
        results_row = {column: results_class[column] for column in overview_cols}
        results_row.update({'params': params, 'time': timings[-1][2]})

        logging.info(f'''
                      {model} training and hyperparameter tuning finished.
                      Test accuracy: {results_row['test_accuracy']}
                      Elapsed time: {timings[-1][2]} minutes
                      ''')
        class_results.append(results_row)
        model_objs.append(results_class)
        class_roc.append((i, model, results_class['test_roc_auc'], results_class['roc']))

logging.info(f' All {FOLDS} folds finished. Writing results to output.')

overview = pd.DataFrame(class_results)
overview.to_csv('SmartHomeSensorics_modeling_results_metrics.csv', index=False)

# Constructing Roc datasets
roc_df = pd.DataFrame({'i': [], 'dep_var': [], 'model': [], 'auc': [], 'tpr': [], 'fpr': []},
                      columns=['i''model', 'auc', 'tpr', 'fpr'])
scaled_roc_df = pd.DataFrame({'i': [], 'model': [], 'auc': [], 'tprs': [], 'fpr': []},
                             columns=['i', 'model', 'auc', 'tprs', 'fpr'])
mean_fpr = np.linspace(0, 1, 100)
for element in class_roc:
    # get length of roc fpr vector
    rc_len = element[3][0].shape[0]
    fpr, tpr = element[3][0], element[3][1]
    res = {}
    res.update({'i': [element[0]] * rc_len, 'model': [element[1]] * rc_len,
                'auc': [element[2]] * rc_len, 'fpr': fpr, 'tpr': tpr})
    roc_df = roc_df.append(pd.DataFrame(res), ignore_index=True)

    # interpolated to have 100 treshold points
    rc_len = len(mean_fpr)
    tprs = np.interp(mean_fpr, fpr, tpr)
    tprs[0] = 0.0
    res = {}
    res.update({'i': [element[0]] * rc_len, 'model': [element[1]] * rc_len,
                'auc': [element[2]] * rc_len, 'fpr': mean_fpr, 'tprs': tprs})
    scaled_roc_df = scaled_roc_df.append(pd.DataFrame(res), ignore_index=True)

roc_df.to_csv('SmartHomeSensorics_modeling_results_roc.csv', index=False)
scaled_roc_df.to_csv('SmartHomeSensorics_modeling_results_scaled_roc.csv', index=False)
