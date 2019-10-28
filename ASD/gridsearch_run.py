# import sys
# import os
# sys.path.insert(0, os.path.expanduser("~/Developer/interpretDL/Pipeline"))

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard


RANDOM_STATE = 42
np.random.seed(seed=RANDOM_STATE) 



def build_dnn(input_dim, nodes=(150,25), activation="elu",
              dropout_rate=(0.0,0.0,0.0), learning_rate = 0.001, reg_scale=0.01,
              momentum=0.0, nesterov=False):
    
    import tensorflow as tf
    import keras
    from keras import optimizers
    from keras import regularizers
    keras.backend.clear_session()
    
    tf.random.set_random_seed(RANDOM_STATE)
    my_reg = regularizers.l2(reg_scale) # Can change this if needed
    
    dnn = keras.models.Sequential()

    Dense = keras.layers.Dense

    # Using He initialization
    he_init = keras.initializers.he_normal(seed=RANDOM_STATE)
    dnn.add(keras.layers.Dropout(rate=dropout_rate[0], input_shape=(input_dim,)))
    dnn.add(Dense(units = nodes[0], activation=activation,
                  kernel_initializer=he_init, kernel_regularizer = my_reg))
    dnn.add(keras.layers.Dropout(dropout_rate[1]))
    dnn.add(Dense(units = nodes[1], activation=activation,
                  kernel_initializer=he_init, kernel_regularizer = my_reg))
    dnn.add(keras.layers.Dropout(dropout_rate[2]))
    
    dnn.add(Dense(units=1, activation="sigmoid",
                  kernel_initializer=he_init, kernel_regularizer = my_reg)) # 5 labels -> logits for now
    
    SGD=keras.optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=nesterov)
    
    dnn.compile(loss='binary_crossentropy',
                  optimizer=SGD,
                  metrics=['accuracy']) #Internally it seems to be same as binary accuracy
    
    return dnn

fname = "data/asd_lr_csf.csv"
raw_data = pd.read_csv(fname, index_col=0).values
X = raw_data[:, :-1]
Y = raw_data[:,-1].reshape(-1,1)
# ZScaler = StandardScaler().fit(X)
# X = ZScaler.transform(X)

# ## Architecture Search

exp_2 = np.logspace(6,8, num=3, base=2, dtype=int)
exp_2_combs = []
for i,x in enumerate(exp_2):
    for j in range(0,i+1):
        exp_2_combs.append((x,exp_2[j]))

prec_scorer = make_scorer(precision_score, pos_label=0)

# def prec_scorer(y_true, y_pred):
#     return precision_score(np.ravel(y_true), np.ravel(y_pred), pos_label=0)

# Performing 10-Fold split
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring={"acc":"accuracy","prec": prec_scorer}

# create model
model = KerasClassifier(build_fn=build_dnn, verbose=0)

# make pipeline
zscaler = StandardScaler()
pipeline = Pipeline([("scaler",zscaler),
                     ("dnn", model)])

# define the grid search parameters

# batch_size = [int(0.9 * X.shape[0])]
batch_size = [10]
num_nodes = [(32,32), (64,64), (16,16)]
epochs = [100, 200, 500, 1000]
class_weights = [compute_class_weight("balanced", np.unique(np.ravel(Y)), np.ravel(Y))]
rates = [(0,0.2,0.2), (0,0.5,0.5)]
scales=[0.01]
learning_rates=[0.001, 0.0001]
momentums=[0.9]
nesterovs=[True]

dnn_grid = dict(input_dim=[X.shape[1]], batch_size=batch_size, epochs=epochs, nodes=num_nodes, activation=["relu"],
                  class_weight=class_weights, dropout_rate=rates, reg_scale=scales,
                  learning_rate=learning_rates, momentum=momentums, nesterov=nesterovs
                 )

param_grid = {}

for param,val in dnn_grid.items():
    param_grid["dnn__"+str(param)] = val
print(param_grid)

grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=4,
                    cv=splitter, scoring=scoring, refit="prec", verbose=1)
print(grid)


grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_acc']
# stds = grid_result.cv_results_['std_test_acc']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%.4f +/- %.3f with: %r" % (mean, stdev, param))


results = pd.DataFrame(grid_result.cv_results_)
results.to_csv("gs_test.csv")
