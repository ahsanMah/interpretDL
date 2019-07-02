# from common_imports import *
# from helper import *
import os
import dill
import hdbscan

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from s_dbw import S_Dbw
from multiprocess import Pool
from multiprocess import get_context
from time import time

class ClusterPipeline:
    """
        model: A keras deep neural network
        data: Original dataset which will be used for training and validation
    """

    class Dataset:
        '''
        Helper class to couple features with their labels easily
        '''
        def __init__(self, X,y):
            self.features = X
            self.labels = y

    class dfHotEncoder(BaseEstimator, TransformerMixin):
        """
        Builds a hot encoder from a pandas dataframe
        Since the function expects an array of "features" per sample,
        we reshape the values
        """
        def __init__(self, random_state=42):
            from sklearn.preprocessing import OneHotEncoder
            
            self.enc = OneHotEncoder(categories="auto", sparse=False)
            self.categories_ = None
            return None
        
        def fit(self, labels):
            self.enc.fit(labels.values.reshape(-1,1))
            self.categories_ = self.enc.categories_
            return self
        
        def transform(self, labels):
            return self.enc.transform(labels.values.reshape(-1,1))
    
    MODELDIR = "models/"
    FIGUREDIR = "pipeline_figs/"
    INITFILE = MODELDIR+"init.h5"
    BASENAME = MODELDIR+"dnn"

    # Filenames for storing / caching model params
    MODELFILE  = BASENAME + "_{id}.h5"
    SCALERFILE = BASENAME + "_{id}_zscaler.pickle"

    def __init__(self, model, train_set, val_set,
                analyzer="lrp",
                cluster_algo = "HDBSCAN",
                linearClassifier="SVM"):
        """
        model: keras object
        data: [training_data, validation_data]
        """
        self.model = model
        self.clusterer = None
        self.model.save_weights(self.INITFILE)

        self.train_set = self.Dataset(*train_set)
        self.val_set = self.Dataset(*val_set)

        self.hot_encoder = self.dfHotEncoder()
        self.hot_encoder.fit(self.train_set.labels)
        self.history = None
        self.zscalers = {}

        self.testing_idxs = []
        self.correct_preds_bool_arr = []
        self.lrp_results = []
        self.n_splits = 10

        self.dnn_analyzers = []
        self.val_set_lrp = []
        # Fold counter to keep track of which model is being trained
        # Each fold should train a fresh model and save its weights
        self.foldctr = 0

    def train(self, X, y, batch_size, epochs, X_test=[], y_test=[], verbose=0, foldnum=0):
        """
        Trains a model using keras API, after scaling the data
        """
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE
        self.model.load_weights(self.INITFILE)

        ZScaler = StandardScaler().fit(X)
        
        X_train = ZScaler.transform(X)
        y_train = self.hot_encoder.transform(y)

        history = self.model.fit(X_train, y_train,
                        epochs=epochs, batch_size = batch_size, verbose=verbose)
        
        self.model.save_weights(self.MODELFILE.format(id=foldnum))

        with open(self.SCALERFILE.format(id=foldnum), "wb") as pklfile:
            dill.dump(ZScaler, pklfile)

        return history, ZScaler

    def runDNNAnalysis(self, X_train, y_train, batch_size, epochs, X_test=[], y_test=[], foldnum=0):
        # import keras
        import innvestigate
        import innvestigate.utils as iutils

        history, ZScaler = self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, foldnum=foldnum)

        # Getting all the samples that can be correctly predicted
        # Note: Samples have already been scaled
        all_samples, _labels, correct_pred_idxs, final_acc = self.getCorrectPredictions(model=self.model,
                                                             samples=X_test, labels=y_test, ZScaler=ZScaler)

        # Stripping the softmax activation from the model
        model_w_softmax = self.model
        # print("Model in child:")
        # print(self.model.summary())
        model_softmax_stripped = iutils.keras.graph.model_wo_softmax(model_w_softmax)

        # Creating an analyzer
        lrp_E = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(
                model=model_softmax_stripped, epsilon=1e-3)

        lrp_results = lrp_E.analyze(all_samples)
        
        print("Finished training Fold: {} -> Loss:{:0.3f}, Acc:{:.4f}".format(foldnum, *final_acc))
        return (final_acc, lrp_results, correct_pred_idxs)

    
    def runFoldWorker(self, foldnum, train_index, test_index, batch_size, epochs):
        print("Running worker:",foldnum)
        X_train = self.train_set.features.iloc[train_index]
        y_train = self.train_set.labels.iloc[train_index]
        X_test  = self.train_set.features.iloc[test_index]
        y_test = self.train_set.labels.iloc[test_index]
        final_acc, lrp_results, correct_pred_idxs = self.runDNNAnalysis(X_train, y_train, X_test=X_test, y_test=y_test,
                                                    epochs=epochs, batch_size=batch_size, foldnum=foldnum)

        return (lrp_results, correct_pred_idxs, test_index)

    def cross_validation(self, batch_size, epochs, num_folds = 10, parallel=True):
        """
        Runs a 10-Fold cross validation by default
        """

        start_time = time()
        
        self.lrp_results = []
        self.correct_preds_bool_arr = []
        results = []
        pool_args = [(fnum, tr_idx, tst_idx, batch_size, epochs) for fnum, (tr_idx, tst_idx) in enumerate(self.getKFold(n_splits=num_folds))]
         
        if parallel:
            print("Running Pool...")
            with get_context("spawn").Pool(processes = os.cpu_count()//2) as pool:
                print("Initialized Pool...")
                results = pool.starmap(self.runFoldWorker, pool_args)
        else:    
            print("Running Serial Crossvalidation")
            for _args in pool_args:
                results.append(self.runFoldWorker(*_args))

        print("Runtime: {:.3f}s".format(time()-start_time))

        for lrp_results, correct_idxs, test_idxs in results:
            self.lrp_results.extend(lrp_results)
            self.correct_preds_bool_arr.extend(correct_idxs)
            self.testing_idxs.extend(test_idxs)

        print("Correct:", len(self.correct_preds_bool_arr))
        print("Test Size:", len(self.testing_idxs))

        return (self.lrp_results, self.correct_preds_bool_arr)

    def train_model(self, batch_size=20, epochs=200, cross_validation=False, parallel=True):

        if cross_validation:
            self.cross_validation(batch_size,epochs, parallel=parallel)
        else:
            X_train, y_train = self.train_set.features, self.train_set.labels
            final_acc, lrp_results, correct_preds = self.runDNNAnalysis(X_train, y_train, epochs=epochs, batch_size=batch_size)
            #Plot LRP
        return


    def train_clusterer(self, class_label, plot=False):
        '''
        Expects a class label to cluster
        This should be the class that the user expects to have subclusters
        '''

        correct_pred_labels = self.train_set.labels.iloc[self.testing_idxs][self.correct_preds_bool_arr]
        split_class = correct_pred_labels == class_label
        split_class_lrp = np.array(self.lrp_results)[split_class]

        data = np.clip(split_class_lrp, 0,None)
        sdata = MinMaxScaler().fit_transform(data)
        labels = correct_pred_labels[split_class]
        
        cluster_sizes = range(15,301,15)
        scores = self.clusterPerf(sdata, labels, cluster_sizes, plot)
        print("Minimum Size:")
        print(scores.idxmin())
        
        minsize = int(scores["Halkidi-Filtered Noise"].idxmin())
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=minsize, prediction_data=True)
        self.clusterer.fit(sdata)

        return scores

    def predict_cluster(self, lrp_data, plot=False):
        data  = np.clip(lrp_data,0,None)
        sdata = MinMaxScaler().fit_transform(data)
        cluster_labels, strengths = hdbscan.approximate_predict(self.clusterer, sdata)

        plt.close("Validation Set Clusters")
        fig, axs = plt.subplots(1, figsize=(15,6), num="Validation Set Clusters")
        plt.title("Validation Set Clusters")

         ## Number of clusters in labels, ignoring noise if present.
        num_clusters = cluster_labels.max() + 1

        color_palette = sns.color_palette("bright", num_clusters)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0, 0, 0) 
                          for x in cluster_labels]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, strengths)]

        axs.scatter(*sdata.T, c=cluster_member_colors, alpha=0.6)
        # plt.colorbar()

        if plot: plt.show()
        plt.savefig(self.FIGUREDIR + "prediction_cluster.png")

        return cluster_labels, strengths

    def foldmodel(self, foldnum=0):
        self.model.load_weights(self.MODELFILE.format(id=foldnum))
        return self.model

    def load_scalers(self):
        self.zscalers = []
        
        for foldnum in range(self.n_splits):
            with open(self.SCALERFILE.format(id = foldnum), "rb") as pklfile:
                self.zscalers.append(dill.load(pklfile))

    def get_predictions(self):

        # Populate zscalers if empty
        if not self.zscalers: self.load_scalers()
        
        predictions = []
        for i,zscaler in enumerate(self.zscalers):
            _samples = zscaler.transform(self.val_set.features)
            model_preds = np.array([(np.argmax(x), np.max(x)) for x in self.foldmodel(i).predict(_samples)])     
            incorrect = model_preds[:,0] != self.val_set.labels
            model_preds[incorrect] = -1
            predictions.append(model_preds)
        
        # Combine all DNN predictions into a matrix
        predictions = np.stack(predictions, axis=1)
        
        # Samples with at least one correct predicition
        valid_samples = (predictions[:,:,0] > -1).any(axis=1)
        predictions = predictions[valid_samples]

        # print("Predictions:", predictions)

        # The DNN number with the highest confidence
        # One for each sample
        best_DNN, best_predictions = [],[]
        if len(predictions)>0:
            best_DNN = np.argmax(predictions[:,:,1], axis=1)
            best_predictions = predictions[range(predictions.shape[0]), best_DNN,0].astype(int)

        return best_predictions, best_DNN

    def load_analyzers(self):
        import innvestigate
        import innvestigate.utils as iutils

        self.dnn_analyzers = []

        for dnn_idx in range(self.n_splits):
            model_w_softmax = self.foldmodel(dnn_idx)
            model_wo_softmax = iutils.keras.graph.model_wo_softmax(model_w_softmax)
            # Creating an analyzer
            self.dnn_analyzers.append(
                innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(
                model=model_wo_softmax, epsilon=1e-3))

        print("Loaded LRP models...")
        return 

    def get_validation_lrp(self):

        if not self.dnn_analyzers: self.load_analyzers()
        analyze = lambda idx,x: self.dnn_analyzers[idx].analyze(x.reshape(1,-1))
        best_predictions, best_DNN = self.get_predictions()

        self.val_set_lrp = []
        for dnn_idx, sample in zip(best_DNN, self.val_set.features.values):
            self.val_set_lrp.extend(analyze(dnn_idx,sample))

        return self.val_set_lrp
    
    # def plot_clusters()

    def get_validation_clusters(self):

        if not self.val_set_lrp: self.get_validation_lrp()
        
        cluster_labels, strengths = self.predict_cluster(self.val_set_lrp, plot=True)

        # plt.close("Validation Set Clusters")
        # fig, axs = plt.subplots(1, 1, figsize=(15,6), num="Cluster Comparison")
        # plt.title("Validation Set Clusters")

        #  ## Number of clusters in labels, ignoring noise if present.
        # num_clusters = cluster_labels.max() + 1

        # color_palette = sns.color_palette("bright", num_clusters)
        # cluster_colors = [color_palette[x] if x >= 0
        #                   else (0, 0, 0) 
        #                   for x in cluster_labels]
        # cluster_member_colors = [sns.desaturate(x, p) for x, p in
        #                          zip(cluster_colors, strengths)]

        # axs[0].scatter(*sdata.T, c=cluster_member_colors, alpha=0.6)
        # plt.colorbar()

        # if plot: plt.show()
        # plt.savefig(self.FIGUREDIR + "prediction_cluster.png")

        return 

    
    ### Helpers
    def getKFold(self, train_data=None, n_splits=10):
        from sklearn.model_selection import StratifiedKFold as KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 ) #Default = 10

        X = train_data.features if train_data else self.train_set.features
        y = train_data.labels if train_data else self.train_set.labels 

        for train_index, test_index in kf.split(X,y):
            yield train_index, test_index
        
        return

    def split_valid(self, features, original_labels, training_labels, valid_size=0.5):
        train_index, validation_index = get_split_index(features, original_labels, test_size=valid_size)[0]
        
        X_valid, y_valid, y_valid_original = features.iloc[validation_index], training_labels.iloc[validation_index], original_labels.iloc[validation_index]
        X_train, y_train, y_original = features.iloc[train_index], training_labels.iloc[train_index], original_labels.iloc[train_index]
        
        return X_train, y_train, y_original, X_valid, y_valid, y_valid_original


    def getCorrectPredictions(self, model, 
                              samples=[],
                              labels=[], ZScaler=None):
        '''
        Assumes categorical output from DNN
        Will default to getting correct predcitions from the validation set
        using the current tained model
        '''
        import numpy as np
        
        if len(samples) == 0 and len(labels) == 0:
            samples = self.val_set.features
            labels = self.val_set.labels

        if ZScaler: samples = ZScaler.transform(samples)

        predictions = model.predict(samples)
        preds = np.array([np.argmax(x) for x in predictions])
        true_labels = np.array([x for x in labels])

        correct_idxs = preds == true_labels

        print("Prediction Accuracy") 
        labels = self.hot_encoder.transform(labels)
        loss_and_metrics = model.evaluate(samples, labels)
        print("Scores on data set: loss={:0.3f} accuracy={:.4f}".format(*loss_and_metrics))
        
        print("Fold Correct:", len(correct_idxs))

        return samples[correct_idxs], labels[correct_idxs], correct_idxs, loss_and_metrics


######## HELPER FUNCTIONS ############

    def clusterPerf(self, data, labels, cluster_sizes, plot=False):

        # if plot:
        plt.close("Cluster Comparison") #1+len(cluster_sizes)
        fig, axs = plt.subplots(1+len(cluster_sizes), 1, figsize=(15,6*(+len(cluster_sizes))), num="Cluster Comparison")
        plt.title("Cluster Comparison")

        axs[0].scatter(*data.T, s=50, linewidth=0, c=labels, alpha=0.5, cmap="Set1")
        axs[0].set_title("Original Distribution")

        _metrics = []

        for i,size in enumerate(cluster_sizes):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
            clusterer.fit(data)
            cluster_labels = clusterer.labels_

            ## Number of clusters in labels, ignoring noise if present.
            num_clusters = cluster_labels.max() + 1

            color_palette = sns.color_palette("bright", num_clusters)
            cluster_colors = [color_palette[x] if x >= 0
                            else (0, 0, 0)
                            for x in clusterer.labels_]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                    zip(cluster_colors, clusterer.probabilities_)]

            # print(cluster_labels)
            noise, halkidi_s_Dbw, halkidi_ignore_noise, halkidi_bind, sil_score = [np.NaN]*5

            noise = list(cluster_labels).count(-1)/len(cluster_labels)
            
            if num_clusters > 1:
                halkidi_s_Dbw = S_Dbw(data, cluster_labels, alg_noise="comb", method='Halkidi',
                            centr='mean', nearest_centr=True, metric='euclidean')
                
                halkidi_ignore_noise = S_Dbw(data, cluster_labels, alg_noise="filter", method='Halkidi',
                            centr='mean', nearest_centr=True, metric='euclidean')
                
                halkidi_bind = S_Dbw(data, cluster_labels, alg_noise="bind", method='Halkidi',
                            centr='mean', nearest_centr=True, metric='euclidean')
                
                sil_score = metrics.silhouette_score(data, cluster_labels, metric="euclidean")
            
            _metrics.append([num_clusters,noise,sil_score, halkidi_s_Dbw, halkidi_ignore_noise, halkidi_bind])

            # if plot:
            axs[i+1].scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.6)
            axs[i+1].set_title("Minimum Cluster Size: {}".format(size))
            axs[i+1].text(0.95,0.95,"Clusters Found: {}".format(num_clusters),
                        horizontalalignment='right', verticalalignment='top',
                        fontsize=14, transform=axs[i+1].transAxes)

        plt.tight_layout()
        if plot: plt.show()

        plt.savefig(self.FIGUREDIR+"cluster_perf_comp.png")
        plt.close("Cluster Comparison")
        
        scores = pd.DataFrame(_metrics, columns=["Clusters", "Noise", "Silhouette","Halkidi", "Halkidi-Filtered Noise", "Halkidi-Bounded Noise"], index=cluster_sizes)
        
        return scores

def calculateEntropy(data, plot=False):
    '''
    Expects data to be 2D numpy array
    '''
    from scipy.stats import entropy
    
    nsamples = len(data)
    nbins = 30

    xedges = np.linspace(0,15,nbins+1)
    yedges = np.linspace(0,15,nbins+1)
    
    x = np.clip(data[:,0], xedges[0], xedges[-1])
    y = np.clip(data[:,1], yedges[0], yedges[-1])
    
    grid, xedges, yedges = np.histogram2d(x, y, bins=[xedges,yedges])
    densities = (grid/nsamples).flatten()
    
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 8))

        ax.imshow(grid, interpolation='nearest', origin='low',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="jet")
#         plt.colorbar()
        plt.show()
    
    return entropy(densities)