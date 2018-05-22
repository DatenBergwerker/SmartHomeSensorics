import sys
import logging
from datetime import datetime
from math import sqrt
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_analysis.models import estimate_model, estimate_model_multiclass

# Settings

NCORES = cpu_count() - 1

# Outer cross validation
FOLDS = 10

# Number of inner folds on the training set for hyperparameter tuning
INNER_FOLDS = 3

# Do hyper param tuning
HYPER_PARAM_TUNING = False

# Logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d.%m.%Y %H:%M:%S')

# utilities
kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
scaler = StandardScaler()
imputer = Imputer()
ohc = OneHotEncoder()

# Data Import
full_data = pd.read_csv(r'SmartHomeSensorics_selected_feature_matrix_cleaned.csv')
full_data_room = full_data.loc[full_data.room_location == 'C']

# only keep windowed data
X = full_data.drop(['binary_target',
                    'index',
                    'entry_id',
                    'no_occupants',
                    'node_id',
                    'measurement_no',
                    'occupant_activity',
                    'state_door',
                    'state_window',
                    'room_location',
                    'measurement_no',
                    'relative_timestamp',
                    'absolute_timestamp',
                    'temperature',
                    'relative_humidity',
                    'light_sensor_1_wvl_nm',
                    'light_sensor_2_wvl_nm'
                    ], axis=1).values

y = full_data['binary_target'].values.reshape(X.shape[0], )
y_mc = full_data['occupant_activity'].values
#y_mc = ohc.fit_transform(y_mc)

# Model definitions
class_models = {
    'LogisticRegression': {'model': LogisticRegression,
                           'params': {'fit_intercept': [True]}},
    'NaiveBayes': {'model': GaussianNB,
                   'params': None},
    # 'Support Vector Machine': {'model': SVC(probability=True),
    #                            'params': {
    #                                'C': [0.1, 1, 10, 100],
    #                                'kernel': ['rbf', 'poly'],
    #                                'degree': range(3, 6)}},
    'Nystroem-LinearSGD': {'model': Pipeline([('Nystroem_feature_map', Nystroem()),
                                              ('SGD_Linear', SGDClassifier())]),
                           'params': {'SGD_Linear__fit_intercept': [True],
                                      'SGD_Linear__loss': ['log', 'modified_huber']}},
    'Random Forest Classifier': {'model': RandomForestClassifier,
                                 'params': {
                                     'max_features': [int(sqrt(X.shape[1]))],
                                     'n_estimators': [100],
                                     'max_depth': [20],
                                     'min_samples_split': [4]}},
    'K-Nearest-Neighbor': {'model': KNeighborsClassifier,
                           'params': {'n_neighbors': [3]}}
}

multiclass_models = {
    'NaiveBayes': {'model': GaussianNB,
                   'params': None},
    'Random Forest Classifier': {'model': RandomForestClassifier,
                                 'params': {
                                     'max_features': [int(sqrt(X.shape[1]))],
                                     'n_estimators': [100],
                                     'max_depth': [20],
                                     'min_samples_split': [4]}}
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
class_results_mc = []
model_objs = []
model_objs_mc = []
class_roc = []
timings = []
timings_mc = []
logging.info(f'Model Training started.')

# main loop
for i, (train, test) in enumerate(kfold.split(X=X, y=y)):
    logging.info(f'--- Beginning iteration {i+1} of {FOLDS} ---')
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    y_train_mc, y_test_mc = y_mc[train], y_mc[test]

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
                                       i=i,
                                       hyper_param_tuning=HYPER_PARAM_TUNING)
        timings.append((i, model, (datetime.now() - time).seconds / 60))
        if class_models[model]['params']:
            if HYPER_PARAM_TUNING:
                params = ' '.join(
                    [str(key) + ' ' + str(results_class['best_params'][key])
                     for key in results_class['best_params']])
            else:
                params = ' '.join(
                    [str(key) + ' ' + str(results_class['param_grid'][key])
                     for key in results_class['param_grid']])
        else:
            params = ''
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

    # Run model zoo
    for model in multiclass_models:
        logging.info(f'Beginning {model} training. - MultiClass')
        time_mc = datetime.now()
        results_class_mc = estimate_model_multiclass(model=multiclass_models[model]['model'],
                                                     name=model,
                                                     param_grid=multiclass_models[model]['params'],
                                                     X_train=X_train,
                                                     X_test=X_test,
                                                     y_train=y_train_mc,
                                                     y_test=y_test_mc,
                                                     i=i,
                                                     hyper_param_tuning=HYPER_PARAM_TUNING)
        timings_mc.append((i, model, (datetime.now() - time_mc).seconds / 60))
        if class_models[model]['params']:
            if HYPER_PARAM_TUNING:
                params = ' '.join(
                    [str(key) + ' ' + str(results_class_mc['best_params'][key])
                     for key in results_class_mc['best_params']])
            else:
                params = ' '.join(
                    [str(key) + ' ' + str(results_class_mc['param_grid'][key])
                     for key in results_class_mc['param_grid']])
        else:
            params = ''
        results_row_mc = {column: results_class_mc[column] for column in overview_cols[:-1]}
        results_row_mc.update({'params': params, 'time': timings[-1][2]})

        logging.info(f'''
                      {model} training and hyperparameter tuning finished.
                      Test accuracy: {results_row['test_accuracy']}
                      Elapsed time: {timings_mc[-1][2]} minutes
                      ''')
        class_results_mc.append(results_row_mc)
        model_objs.append(results_class_mc)

logging.info(f' All {FOLDS} folds finished. Writing results to output.')

overview = pd.DataFrame(class_results)
overview.to_csv('SmartHomeSensorics_modeling_results_metrics_room_A.csv', index=False)
overview_mc = pd.DataFrame(class_results_mc)
overview_mc.to_csv('SmartHomeSensorics_modeling_results_metrics_mc_room_A.csv', index=False)

# Constructing Roc datasets
roc_df = pd.DataFrame({'i': [], 'dep_var': [], 'model': [], 'auc': [], 'tpr': [], 'fpr': []},
                      columns=['i', 'model', 'auc', 'tpr', 'fpr'])
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
