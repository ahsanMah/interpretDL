# from common_imports import *
# from helper import *
import os
import dill
import hdbscan
import umap

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from s_dbw import S_Dbw
from multiprocess import Pool
from multiprocess import get_context
from time import time

RANDOM_STATE = 42

class ClusterPipeline:
    """
        model: A keras deep neural network
        data: Original dataset which will be used for training and validation
    """


    ###### Defining useful custom classes and transformers ######
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
            return self.enc.transform(np.array(labels).reshape(-1,1))

    class DataFrameScaler(BaseEstimator, TransformerMixin):
        """
        Returns a numpy matrix of ZScaled columns identified by attribute_names 
        """
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            self.scaler = StandardScaler().fit(X[self.attribute_names])
            return self
        def transform(self, X):
            cat_cols = X.drop(columns=self.attribute_names).values
            scaled_cols = self.scaler.transform(X[self.attribute_names])
            return np.concatenate((scaled_cols, cat_cols), axis=1)

    MODELDIR = "models/"
    FIGUREDIR = "pipeline_figs/"
    INITFILE = MODELDIR+"init.h5"
    BASENAME = MODELDIR+"dnn"

    # Filenames for storing / caching model params
    MODELFILE  = BASENAME + "_{id}.h5"
    SCALERFILE = BASENAME + "_{id}_zscaler.pickle"
    ANALYZERFILE = BASENAME + "_{id}_analyzer.pickle"
    TRAINFILE = BASENAME + "_{id}_train.pickle"

    def __init__(self, model, train_set, val_set,
                target_class=0,
                reducer=umap.UMAP(random_state=42),
                analyzer_type="lrp.epsilon",
                numerical_cols = None,
                cluster_algo = "HDBSCAN",
                linearClassifier="SVM"):
        """
        model: keras object
        data: [training_data, validation_data]
        reducer: UMAP object with parameters appropriate for the data
        """
        self.model = model
        self.clusterer = None
        self.analyzer_type = analyzer_type

        self.model.save_weights(self.INITFILE)
        self.reducer_pipeline = Pipeline([
            ("umap", reducer),
            ("scaler",  MinMaxScaler())
        ])

        self.train_set = self.Dataset(*train_set)
        self.val_set = self.Dataset(*val_set)
        self.target_class = target_class
        self.numerical_cols = numerical_cols if numerical_cols else self.train_set.features.columns


        self.hot_encoder = self.dfHotEncoder()
        self.hot_encoder.fit(self.train_set.labels)
        self.history = None
        self.zscalers = {}

        self.predictions = []
        self.testing_idxs = []
        self.correct_preds_bool_arr = []
        self.lrp_results = []
        self.training_lrp = []
        
        #  Mask for retreiving validation samples that were predicted correctly
        self.val_pred_mask = []
        self.n_splits = 10

        self.dnn_analyzers = []
        self.val_set_lrp = []

    def train(self, X, y, batch_size, epochs, X_test=[], y_test=[], verbose=0, foldnum=0):
        """
        Trains a model using keras API, after scaling the data
        """
        # from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE
        self.model.load_weights(self.INITFILE)

        ZScaler = self.DataFrameScaler(self.numerical_cols).fit(X)
        X_train = ZScaler.transform(X)
        X_train,y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train,np.ravel(y)) # Both are np arrays now
        y_train = self.hot_encoder.transform(y_train)

        history = self.model.fit(X_train, y_train,
                        epochs=epochs, batch_size = batch_size, verbose=verbose)
        
        # Save states of training run
        self.model.save_weights(self.MODELFILE.format(id=foldnum))

        with open(self.SCALERFILE.format(id=foldnum), "wb") as pklfile:
            dill.dump(ZScaler, pklfile)

        with open(self.TRAINFILE.format(id=foldnum), "wb") as pklfile:
            dill.dump(X_train, pklfile)

        return history, ZScaler

    def runDNNAnalysis(self, X_train, y_train, batch_size, epochs, X_test=[], y_test=[], foldnum=0, verbose=0):
        # import keras
       
        history, ZScaler = self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, foldnum=foldnum, verbose=verbose)
        train_acc = [history.history["loss"][-1] , history.history["acc"][-1]]
        print("Fold: {} -> Loss:{:0.3f}, Acc:{:.4f}".format(foldnum, *train_acc))

        # Getting all the samples that can be correctly predicted
        # Note: Samples have already been scaled
        all_samples, predictions, correct_pred_idxs, final_acc = self.getCorrectPredictions(model=self.model,
                                                             samples=X_test, labels=y_test, ZScaler=ZScaler)
        
        correct_samples = all_samples[correct_pred_idxs]

        # Creating an analyzer
        analyzer = self.get_analyzer(self.model, ZScaler.transform(X_train.values))
        relevance_results = analyzer.analyze(correct_samples)
        
        return (predictions, relevance_results, correct_pred_idxs)

    def get_analyzer(self, model, X_train):
        import innvestigate
        import innvestigate.utils as iutils

        analyzer_kwargs = {
            "pattern.attribution":
                {"pattern_type":"relu"},
            "lrp.epsilon":
                {"epsilon":1e-3}
         }

        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

        analyzer = innvestigate.create_analyzer(self.analyzer_type, model_wo_softmax,
                                                **analyzer_kwargs[self.analyzer_type])
        analyzer.fit(X_train, batch_size=20, verbose=1, disable_no_training_warning=True)
        
        return analyzer
    
    def runFoldWorker(self, foldnum, train_index, test_index, batch_size, epochs):
        print("Running worker:",foldnum)
        X_train = self.train_set.features.iloc[train_index]
        y_train = self.train_set.labels.iloc[train_index]
        X_test  = self.train_set.features.iloc[test_index]
        y_test = self.train_set.labels.iloc[test_index]
        predictions, lrp_results, correct_pred_idxs = self.runDNNAnalysis(X_train, y_train, X_test=X_test, y_test=y_test,
                                                    epochs=epochs, batch_size=batch_size, foldnum=foldnum)

        return (predictions, lrp_results, correct_pred_idxs, test_index)

    def cross_validation(self, batch_size, epochs, num_folds = 10, parallel=True):
        """
        Runs a 10-Fold cross validation by default
        """

        start_time = time()
        
        results = []
        pool_args = [(fnum, tr_idx, tst_idx, batch_size, epochs) 
                      for fnum, (tr_idx, tst_idx) in enumerate(self.getKFold(n_splits=num_folds))]
         
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
        
        return self.extractResults(results)

    def extractResults(self, results):
        # print(results)
        for predictions, lrp_results, correct_idxs, test_idxs in results:
            self.predictions.extend(predictions)
            self.lrp_results.extend(lrp_results)
            self.correct_preds_bool_arr.extend(correct_idxs)
            self.testing_idxs.extend(test_idxs)

        print("Correct:", self.correct_preds_bool_arr.count(True))
        print("Test Size:", len(self.testing_idxs))
        return

    def train_model(self, batch_size=20, epochs=200, verbose=0, cross_validation=False, parallel=True):
        
        self.lrp_results = []
        self.correct_preds_bool_arr = []
        self.predictions = []
        self.testing_idxs = []

        if cross_validation:
            self.cross_validation(batch_size,epochs, parallel=parallel)
        else:
            X_train, y_train = self.train_set.features, self.train_set.labels
            pool_args = [(fnum, tr_idx, tst_idx, batch_size, epochs) 
                        for fnum, (tr_idx, tst_idx) in enumerate(
                                self.get_split_index(features=X_train, labels=y_train)
                        )]

            results = [self.runFoldWorker(*pool_args[0])]
            self.extractResults(results)
        return


    def train_clusterer(self, class_label=None, min_cluster_sizes=[], plot=False):
        '''
        Expects a class label to cluster
        This should be the class that the user expects to have subclusters
        '''

        if None == class_label: class_label = self.target_class

        correct_pred_labels = self.train_set.labels.iloc[self.testing_idxs][self.correct_preds_bool_arr]
        split_class = correct_pred_labels == class_label
        split_class_lrp = np.array(self.lrp_results)[split_class]

        self.training_lrp = np.clip(split_class_lrp, 0,None)
        labels = correct_pred_labels[split_class]
        
        self.reducer_pipeline = self.reducer_pipeline.fit(self.training_lrp)
        embeddings = self.reducer_pipeline.transform(self.training_lrp)

        if not min_cluster_sizes:
            n_neighbours = self.reducer_pipeline["umap"].n_neighbors
            min_cluster_sizes = range(n_neighbours-3, n_neighbours+3)

        scores = self.clusterPerf(embeddings, labels, min_cluster_sizes, plot)
        print("Minimum Size:")
        print(scores.idxmin())
        
        minsize, minsamp = scores["Halkidi-Filtered Noise"].idxmin()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=minsize, min_samples=minsamp, prediction_data=True)
        self.clusterer.fit(embeddings)

        return scores
        

    def predict_cluster(self, lrp_data, plot=False):
        embeddings=self.reducer_pipeline.transform(lrp_data)

        cluster_labels, strengths = hdbscan.approximate_predict(self.clusterer, embeddings)

        if plot:
            plt.close("Validation Relevance")
            fig, axs = plt.subplots(1, figsize=(15,6), num="Validation Relevance")
            # divider = make_axes_locatable(axs)
            # cax = divider.append_axes('right', size='3.5%', pad=0.5)
            plt.title("Validation Relevance")

            ## Number of clusters in labels, ignoring noise if present.
            num_clusters = cluster_labels.max() + 1

            color_palette = sns.color_palette("bright", num_clusters)
            cluster_colors = [color_palette[x] if x >= 0
                            else (0, 0, 0) 
                            for x in cluster_labels]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                    zip(cluster_colors, strengths)]

            _mappable = axs.scatter(*sdata.T, c=cluster_member_colors, s=10, alpha=0.7)
            
            # Doesnt work properly
            # fig.colorbar(_mappable, cax=cax, orientation='vertical')

            plt.savefig(self.FIGUREDIR + "prediction_lrp.png")
            plt.show()

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
        # Note: Samples incorrectly predicted by all DNNs are dropped
        self.val_pred_mask =  (predictions[:,:,0] > -1).any(axis=1)
        predictions = predictions[self.val_pred_mask]

        print("Prediction Accuracy: {:.4f}".format(predictions.shape[0]/self.val_set.labels.shape[0]))

        # The DNN number with the highest confidence
        # One for each sample
        best_DNN, best_predictions = [],[]
        if len(predictions)>0:
            best_DNN = np.argmax(predictions[:,:,1], axis=1)
            best_predictions = predictions[range(predictions.shape[0]), best_DNN,0].astype(int)
        
        #Note: Best predicitons are the same as val_set.labels[self.val_pred_mask]
        return best_predictions, best_DNN

    def load_analyzers(self):
        print("Loading LRP Analyzers...")
        
        import innvestigate.utils as iutils
        import innvestigate

        self.dnn_analyzers = []

        for dnn_idx in range(self.n_splits):
            with open(self.TRAINFILE.format(id=dnn_idx), "rb") as pklfile:
                _x_train = dill.load(pklfile)
                # Creating an analyzer
                analyzer = self.get_analyzer(self.foldmodel(dnn_idx), _x_train)
                self.dnn_analyzers.append(analyzer)
        print("Done!")
        return 

    def get_validation_lrp(self):
       
        if not self.dnn_analyzers: self.load_analyzers()
        analyze = lambda idx,x: self.dnn_analyzers[idx].analyze(x.reshape(1,-1))
        best_predictions, best_DNN = self.get_predictions()

        # Only consider the samples from the class(es) which are expected to have subclusters
        target_class = best_predictions == self.target_class
        class_samples = self.val_set.features.values[self.val_pred_mask][target_class]
        class_DNN = best_DNN[best_predictions == self.target_class]

        self.val_set_lrp = []
        for dnn_idx, sample in zip(class_DNN, class_samples):
            self.val_set_lrp.extend(analyze(dnn_idx,sample))

        self.val_set_lrp = np.clip(self.val_set_lrp, 0, None)
        return self.val_set_lrp
    
    def get_validation_clusters(self, plot=False):

        if len(self.val_set_lrp) == 0: self.get_validation_lrp()
        
        cluster_labels, strengths = self.predict_cluster(self.val_set_lrp)

        target_class = self.val_set.labels[self.val_pred_mask] == self.target_class
        class_samples = self.val_set.features.values[self.val_pred_mask][target_class]
        
        # Plot samples with cluster labels
        if plot:
            plt.close("Validation Set Clusters")
            fig, axs = plt.subplots(1, 1, figsize=(15,6), num="Validation Set Clusters")
            plt.title("Validation Set Clusters")

            ## Number of clusters in labels, ignoring noise if present.
            num_clusters = cluster_labels.max() + 1

            color_palette = sns.color_palette("bright", num_clusters)
            cluster_colors = [color_palette[x] if x >= 0
                            else (0, 0, 0) 
                            for x in cluster_labels]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                    zip(cluster_colors, strengths)]

            axs.scatter(*class_samples.T, c=cluster_member_colors)
            # plt.colorbar()

            plt.savefig(self.FIGUREDIR + "prediction_cluster.png")

            if plot: plt.show()

        return class_samples, cluster_labels
    

    def getSubclusters(self, reduce=True):
        """
        Returns dictionary of training,label pairs for every subcluster
        """

        subcluster_labels = range(0,max(self.clusterer.labels_)+1)

        # Get testing samples from cross validation
        reindexer = self.testing_idxs

        # That were correctly predicted
        correct_preds = self.correct_preds_bool_arr

        # Separating Control from Target
        target_samples  = self.train_set.labels.iloc[reindexer][correct_preds] == self.target_class
        control_samples = ~target_samples

        target_class_features = self.train_set.features.iloc[reindexer][correct_preds][target_samples]
        target_class_labels   = self.train_set.labels.iloc[reindexer][correct_preds][target_samples]
        # target_class_features.head()

        cluster_train = {}

        for cluster_label in subcluster_labels:
            
            tsamples = target_class_features[(self.clusterer.labels_ == cluster_label)]
            tlabels  = target_class_labels[(self.clusterer.labels_ == cluster_label)]


        #     csamples = self.train_set.features.iloc[reindexer][correct_preds][control_samples]
            csamples = self.train_set.features[self.train_set.labels != self.target_class]
            csamples = csamples[:len(tsamples)]

            clabels = self.train_set.labels.iloc[reindexer][correct_preds][control_samples]
            clabels = clabels[:len(tsamples)]

            if reduce:
                _clustered = pd.DataFrame(self.training_lrp[(self.clusterer.labels_ == cluster_label)],
                                        columns = self.train_set.features.columns)
                
            #     thresh = min(val_clustered.describe().loc["75%"])
                thresh = _clustered.max().min()
                # print(thresh)
                reduced_cols = self.get_relevant_cols(_clustered, thresh=thresh).columns
                # print(reduced_cols)
                
                tsamples = tsamples[reduced_cols]
                csamples  = csamples[reduced_cols]
                
            # Now stack it with control values of same size...
            X_train_sc = pd.concat([csamples, tsamples], axis="index")
            y_train_sc = pd.concat([clabels, tlabels], axis="index")
            
            cluster_train[cluster_label] = (X_train_sc, y_train_sc)

        return cluster_train

    ### Helpers
    def get_split_index(self, features, labels, test_size=0.3):
        import numpy as np
        from sklearn.model_selection import StratifiedShuffleSplit
        
        features = np.array(features)
        # The train set will have equal amounts of each target class
        # Performing single split
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        return [[train_index, test_index] for train_index,test_index in split.split(features, labels)]

    # def split_valid(self, features, training_labels, valid_size=0.5):
    #     train_index, validation_index = get_split_index(features, training_labels, test_size=valid_size)[0]
        
    #     X_valid, y_valid = features.iloc[validation_index], training_labels.iloc[validation_index]
    #     X_train, y_train = features.iloc[train_index], training_labels.iloc[train_index]
        
    #     return X_train, y_train, X_valid, y_valid
    

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
        
        print("Fold Correct:", correct_idxs.sum())

        return samples, preds, correct_idxs, loss_and_metrics

    def get_relevant_cols(self, df, thresh = 1e-2):

        all_above_thresh = (df < thresh).all(0) #Check if all values in columns satisfy the criteria
        max_above_thresh = (df.max() < thresh)
        quantile_above_thresh = (df.quantile(0.8) <= thresh)

        criteria = quantile_above_thresh
        irrelevant_cols = df.columns[criteria] 
        irrelevant_cols
        
        relevant_features_only = df.drop(columns = irrelevant_cols)
        
        return relevant_features_only

######## HELPER FUNCTIONS ############

    def clusterPerf(self, data, labels, cluster_sizes, plot=False):
        ii32 = np.iinfo(np.int32)
        
        # FIXME: Assumes 2D data only
        if plot:
            plt.close("Cluster Comparison") #1+len(cluster_sizes)
            fig, axs = plt.subplots(1+len(cluster_sizes), 1, figsize=(15,6*(+len(cluster_sizes))), num="Cluster Comparison")
            plt.title("Cluster Comparison")

            axs[0].scatter(*data.T, s=50, linewidth=0, c=labels, alpha=0.5, cmap="Set1")
            axs[0].set_title("Original Distribution")

        _metrics = []

        min_samp_start = cluster_sizes[0]

        for i,size in enumerate(cluster_sizes):
            min_samples = range(min_samp_start,size+1)

            for j,min_s in enumerate(min_samples):
                clusterer = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=min_s)
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
                noise, halkidi_s_Dbw, halkidi_ignore_noise, halkidi_bind, sil_score = [ii32.max]*5

                noise = list(cluster_labels).count(-1)/len(cluster_labels)
                
                if num_clusters > 1:
                    halkidi_s_Dbw = S_Dbw(data, cluster_labels, alg_noise="comb", method='Halkidi',
                                centr='mean', nearest_centr=True, metric='euclidean')
                    
                    halkidi_ignore_noise = S_Dbw(data, cluster_labels, alg_noise="filter", method='Halkidi',
                                centr='mean', nearest_centr=True, metric='euclidean')

                _metrics.append([num_clusters,noise, halkidi_s_Dbw, halkidi_ignore_noise])

                if plot:
                    axs[i+1].scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.6)
                    axs[i+1].set_title("Minimum Cluster Size: {}".format(size))
                    axs[i+1].text(0.95,0.95,"Clusters Found: {}".format(num_clusters),
                            horizontalalignment='right', verticalalignment='top',
                            fontsize=14, transform=axs[i+1].transAxes)

        if plot:
            plt.tight_layout()
            plt.show()
            plt.savefig(self.FIGUREDIR+"cluster_perf_comp.png")
            plt.close("Cluster Comparison")
        
        index  = [y for x in cluster_sizes for y in list(zip([x]*x,range(min_samp_start,x+1))) ]
        scores = pd.DataFrame(_metrics, columns=["Clusters", "Noise","Halkidi", "Halkidi-Filtered Noise"], index=index)
        
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