#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Magic line to force reload all modules when this cell is run multiple times
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.offline import plot
import ipywidgets as widgets

# %matplotlib widget
py.offline.init_notebook_mode(connected=True)


# In[2]:


import numpy as np
import pandas as pd
import umap

from ClusterPipeline import ClusterPipeline
from helper import split_valid, plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


np.random.seed(seed=42) 


# In[7]:


X,y = make_classification(n_samples=15000, n_classes=3, weights=[0.5,0.25,0.25], n_features=10,
                          n_informative=5, n_redundant=5, class_sep=0.7, n_clusters_per_class=1,
                          hypercube=True, shuffle=True, random_state=42)


# In[8]:


data = pd.DataFrame(X)
labels = pd.Series(y)
# data.hist(figsize=(20,12))


# In[9]:


# Separating a hold out set that will be used for validation of the pipeline
train_data, train_labels, test_data, test_labels = split_valid(data, labels, valid_size=0.5)


# original_labels = train_labels.copy()
# train_labels[train_labels > 0] = 1
train_labels.value_counts()


# In[11]:


X_train = train_data
y_train = train_labels.copy()
y_train[y_train > 0] = 1

X_valid = test_data
y_valid = test_labels.copy()
y_valid[y_valid > 0] = 1


# In[12]:


# X_train, y_train, X_valid, y_valid = split_valid(train_data, train_labels, valid_size=0.5)


# In[13]:


def build_dnn(num_features, num_nodes = 16, depth = 2, num_labels=2, activation = "elu"):
    
    import tensorflow as tf
    import keras
    from keras import optimizers
    from keras import regularizers
    keras.backend.clear_session()
    
    reg_scale = 0.001 # For L1 Reg
    my_reg = regularizers.l1(reg_scale) # Can change this if needed
    
    dnn = keras.models.Sequential()

    Dense = keras.layers.Dense

    # Using He initialization
    he_init = keras.initializers.he_normal()
    
    dnn.add(Dense(units = 20, activation="elu", input_dim=num_features,
                  kernel_initializer=he_init, kernel_regularizer = my_reg))
    dnn.add(keras.layers.Dropout(0.5))
    dnn.add(Dense(units = 10, activation="elu",
                  kernel_initializer=he_init, kernel_regularizer = my_reg))
    dnn.add(keras.layers.Dropout(0.5))
    dnn.add(Dense(units = 5, activation="elu",
                  kernel_initializer=he_init, kernel_regularizer = my_reg))
    # dnn.add(keras.layers.Dropout(0.5))

    
    dnn.add(Dense(units=num_labels, activation="softmax",
                  kernel_initializer=he_init, kernel_regularizer = my_reg)) # 5 labels -> logits for now
    
#     nadam = keras.optimizers.Nadam()
    NSGD = keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True)
    
    dnn.compile(loss='categorical_crossentropy',
                  optimizer=NSGD,
                  metrics=['accuracy'])
    
    return dnn

def plot_clusters(pipeline):
    
    training_lrp = pipeline.training_lrp
    
    # Internally populates self.val_set_lrp
    samples, cluster_labels = pipeline.get_validation_clusters()
    
    val_lrp = pipeline.val_set_lrp
    # pipeline.reducer_pipeline[0].n_neighbors
    embedding_pipeline = Pipeline([
        ("reducer", umap.UMAP(random_state=42,
                        n_components = 3,
                        n_neighbors=15,
                        min_dist=0.1)),
    ("scaler", MinMaxScaler())
    ])
    embedding_pipeline.fit(training_lrp)

    embedding = embedding_pipeline.transform(training_lrp)

    emb3d = go.Scatter3d(
        x=embedding[:,0],
        y=embedding[:,1],
        z=embedding[:,2],
        mode="markers",
        name="Training",
        marker=dict(
            size=2,
            color=pipeline.clusterer.labels_,
            colorscale="Rainbow",
            opacity=0.8
        ),
        text=pipeline.clusterer.labels_
    )

    val_3d_embedding = embedding_pipeline.transform(val_lrp)

    val_emb3d = go.Scatter3d(
        x=val_3d_embedding[:,0],
        y=val_3d_embedding[:,1],
        z=val_3d_embedding[:,2],
        name="Validation",
        mode="markers",
        marker=dict(
            size=5,
            color=cluster_labels,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True
        ),
        text = cluster_labels
    )

    layout = go.Layout(
        title="3D LRP Embedding",
        autosize=False,
        width=1200,
        height=1000,
        paper_bgcolor='#F5F5F5',
    #     template="plotly"
    )


    data=[emb3d, val_emb3d]

    fig = go.Figure(data=data, layout=layout)
    # fig.update_layout(template="plotly")  /

    iplot(fig, filename='lrp-3d-scatter.html')

# In[14]:
nn = build_dnn(num_features=train_data.shape[1])
reducer = umap.UMAP(random_state=42,
                    n_components = 10,
                    n_neighbors=150,
                    min_dist=0)

if __name__ == '__main__':
    pipeline = ClusterPipeline(nn, [X_train, y_train], [X_valid,y_valid], target_class=1, reducer=reducer)
    cm = pipeline.train_model(batch_size=50,epochs=100, cross_validation=True, parallel=True, verbose=0)
    _, correct_pred_idxs = pipeline.train_clusterer(plot=False)
    print("Clusters Found:", max(pipeline.clusterer.labels_)+1)

    pipeline.reducer_pipeline[0].n_neighbors
    embedding_pipeline = Pipeline([
        ("reducer", umap.UMAP(random_state=42,
                        n_components = 3,
                        n_neighbors=100,
                        min_dist=0)),
    ("scaler", MinMaxScaler())
    ])

    _, cluster_labels = pipeline.get_validation_clusters()
    val_lrp = pipeline.val_set_lrp

    best_predictions, best_DNN = pipeline.get_predictions()

    # Only consider the samples from the class(es) which are expected to have subclusters
    target_class = best_predictions == pipeline.target_class
    control_class = best_predictions != pipeline.target_class

    val_samples_control = pipeline.val_set.features.values[pipeline.val_pred_mask][control_class]
    control_labels = pipeline.val_set.labels.values[pipeline.val_pred_mask][control_class]

    original_labels = test_labels[pipeline.val_pred_mask][target_class]
    original_labels = original_labels[cluster_labels > -1]
    clustered_labels = cluster_labels[cluster_labels > -1]

    embedding_pipeline.fit(pipeline.training_lrp)
    val_3d_embedding = embedding_pipeline.transform(val_lrp)
    val_3d_embedding = val_3d_embedding[cluster_labels > -1]

    val_emb3d = go.Scatter3d(
        x=val_3d_embedding[:,0],
        y=val_3d_embedding[:,1],
        z=val_3d_embedding[:,2],
        name="Validation",
        mode="markers",
        marker=dict(
            size=5,
            color=clustered_labels,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True
        ),
        text = clustered_labels
    )

    layout = go.Layout(
        title="3D LRP Embedding",
        autosize=False,
        width=1200,
        height=1000,
        paper_bgcolor='#F5F5F5',
    #     template="plotly"
    )

    data=[val_emb3d]
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='synth_val_clusters.html')

    val_emb3d["marker"]["color"] = original_labels
    data=[val_emb3d]
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='synth_val_orig_labls.html')

    from sklearn import metrics

    labels_true = original_labels
    labels_pred = clustered_labels

    print("AMI:", metrics.adjusted_mutual_info_score(labels_true, labels_pred))
    print("ARand:", metrics.adjusted_rand_score(labels_true, labels_pred))
    print("Completeness:", metrics.completeness_score(labels_true, labels_pred))
    print("Homogeneity:", metrics.homogeneity_score(labels_true, labels_pred))
    print("V-Measure:", metrics.v_measure_score(labels_true, labels_pred))