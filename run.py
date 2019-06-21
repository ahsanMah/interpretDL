#%%

# Magic line to force reload all modules when this cell is run muktiple times
%load_ext autoreload
%autoreload 2

from ClusterPipeline import ClusterPipeline
# reload(sys.modules["ClusterPipeline"])

from common_imports import *
from helper import *

#%%
def build_dnn(num_features, num_nodes = 16, depth = 2, num_labels=2, activation = "elu"):
    
    import tensorflow as tf
    import keras
    keras.backend.clear_session()
    
    nn = keras.models.Sequential()
    Dense = keras.layers.Dense
    
    # Using He initialization
    he_init = tf.keras.initializers.he_uniform()
    
    nn.add(Dense(units = num_nodes, activation=activation, input_dim=num_features,
                kernel_initializer=he_init))
    
    for i in range(1,depth):
        nn.add(Dense(units = num_nodes, activation=activation,
                    kernel_initializer=he_init))

    nn.add(Dense(units=num_labels, activation= "softmax",
                kernel_initializer=he_init))
    
    nn.compile(loss="categorical_crossentropy",
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    return nn

#%%
original_data, modded_samples, training_labels, original_labels = simulate_blobs(class_size=1000)
# Separating a hold out set that will be used for validation later
X_train, y_train, X_test, y_test, y_original, X_valid, y_valid, y_valid_original = get_train_test_val(modded_samples, original_labels, training_labels)

#%%
num_features = modded_samples.shape[1]
nn = build_dnn(num_features, depth=2)
#%%
pipeline = ClusterPipeline(nn, [X_train, y_train], [X_test,y_test])

#%%
pipeline.train_model(20,10)

#%%