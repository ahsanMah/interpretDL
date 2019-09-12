#%%

# Magic line to force reload all modules when this cell is run multiple times
# %load_ext autoreload
# %autoreload 2

from ClusterPipeline import ClusterPipeline
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
nn = build_dnn(num_features=2, depth=2)

if __name__ == "__main__":
    original_data, modded_samples, training_labels, original_labels = simulate_blobs(class_size=3000)

    # Separating a hold out set that will be used for validation later
    X_train, y_train, y_original, X_valid, y_valid, y_valid_original = split_valid(modded_samples, original_labels, training_labels)    
    num_features = modded_samples.shape[1]
    # print("Model in parent proc:", nn)
    # print(nn.summary())

    pipeline = ClusterPipeline(nn, [X_train, y_train], [X_valid,y_valid])
    pipeline.train_model(batch_size=20,epochs=100, cross_validation=True, parallel=True)
    
    predictions, DNNs = pipeline.get_predictions()
    pipeline.train_clusterer(class_label=0, plot=False)
    samples, cluster_labels = pipeline.get_validation_clusters()
#%%
if __name__ == "__main__":
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score

    original_val_samples = pipeline.val_set.features[pipeline.val_pred_mask]
    original_val_labels = pipeline.val_set.labels[pipeline.val_pred_mask]

    svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("SVM", LinearSVC(**{'C': 10, 'loss': 'hinge', 'max_iter': 10000000, 'tol': 0.0001}))
    ])
    _score = cross_val_score(svm_clf, X = original_val_samples, y=original_val_labels, cv=10)
    print("SVM Accuracy: {:0.3f}(+/- {:.3f})".format(_score.mean(), _score.std()*2))
    
    center_class = original_val_samples[original_val_labels == 1]
    target_class = original_val_samples[original_val_labels == 0]
    center_labels = [-1]* len(original_val_labels[original_val_labels == 1])

    # Separate training set for each class with
    # equal amounts of subcluster and center blob
    xtrain = {}
    start = 0
    for i in range(0,cluster_labels.max()+1):
        
        _subclass = target_class[cluster_labels == i]
        _labels = cluster_labels[ cluster_labels == i]
        
        end = start+len(_subclass)
        _xtrain = np.concatenate((center_class[start:end], _subclass))
        _ytrain = np.concatenate((center_labels[start:end], _labels))
        
        xtrain[i] = (_xtrain, _ytrain)
        start += len(_subclass)
    # print(xtrain)
    scores = []
    sizes = [len(xtrain[i][1]) for i in xtrain]

    for i in xtrain:
        print("\tSubcluster",i)
        print("Size:", sizes[i] )
        
        _score = cross_val_score(svm_clf, X = xtrain[i][0], y=xtrain[i][1], cv=10)
        
    #     %time svm_clf.fit(_xtrain, _ytrain)
        scores.append(_score)
        print("SVM Accuracy: {:0.3f}(+/- {:.3f})".format(_score.mean(), _score.std()*2))
        
    print("----------------------")
    # subcluster_avg = np.mean([s.mean() for s in scores])
    # print("Mean Score: {:0.4f}".format(subcluster_avg))

    # This is actually the "true" mean: sum(correctly classified) / (total samples)
    weighted_avg = sum([sz*sc.mean() for sz,sc in zip(sizes,scores)])/sum(sizes)
    print("Weighted Mean: {:0.4f}".format(weighted_avg))

#%%
