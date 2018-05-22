import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.metrics import confusion_matrix

ITERATIONS = 50

data_matrix = pd.read_csv('SmartHomeSensorics_selected_feature_matrix_cleaned.csv')
X = data_matrix.drop(['binary_target',
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

# Construct feature vector for multiclass and binary classification
# last column contains binary target, the others the one hot encoded multiclass target
y = data_matrix['occupant_activity']
y = to_categorical(y)
num_classes = y.shape[1]
y = np.concatenate((y, data_matrix['binary_target'].values.reshape((y.shape[0], 1))), axis=1)


# TODO: implement f1, precision, recall hooks


def create_MLP_binary(input_dim: int):
    """
    Simple multi-layered perceptron for occupancy detection. Takes the input dimension as only parameter.
    """
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_dim=input_dim))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_MLP_multiclass(input_dim: int, target_dim: int):
    """
    Simple multi-layered perceptron for activity detection. Takes the input dimension as only parameter.
    """
    model = Sequential()
    model.add(Dense(units=512, activation='relu', input_dim=input_dim))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=target_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


imputer = Imputer()
scaler = StandardScaler()
split = ShuffleSplit(n_splits=1, test_size=0.3)

result_list_bc = []
conf_mats_bc = []
result_list_mc = []
conf_mats_mc = []

# Bootstrapped Cross Validation
for i in range(ITERATIONS):
    # Assign splits
    indices = [(train, test) for train, test in split.split(X)]
    train, test = indices[0][0], indices[0][1]
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    X_train = imputer.fit_transform(X=X_train)
    X_test = imputer.transform(X=X_test)

    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    mlp = create_MLP_binary(input_dim=X_train.shape[1])
    mlp_mc = create_MLP_multiclass(input_dim=X_train.shape[1], target_dim=num_classes)

    mlp.fit(x=X_train, y=y_train[:, -1], epochs=10, batch_size=512)
    result_list_bc.append(mlp.evaluate(x=X_test, y=y_test[:, -1]))
    conf_mats_bc.append(m)

    mlp_mc.fit(x=X_train, y=y_train[:, :-1], epochs=10, batch_size=512)
    result_list_mc.append(mlp_mc.evaluate(x=X_test, y=y_test[:, :-1]))

results = pd.DataFrame({'run': range(ITERATIONS),
                        'binary_score': result_list_bc,
                        'multiclass_score': result_list_mc})

results.to_csv('MLP_results_bootstrapped-cv_shared_room.csv', index=False)
