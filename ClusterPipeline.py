# from common_imports import *
# from helper import *
import os
import dill
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
        self.model.save_weights(self.INITFILE)

        self.train_set = self.Dataset(*train_set)
        self.val_set = self.Dataset(*val_set)

        self.hot_encoder = self.dfHotEncoder()
        self.hot_encoder.fit(self.train_set.labels)
        self.history = None
        self.model_zoo = []
        self.zscalers = {}

        self.n_splits = 10
        # Fold counter to keep track of which model is being trained
        # Each fold should train a fresh model and save its weights
        self.foldctr = 0

    def train(self, X, y, batch_size, epochs, X_test=[], y_test=[], verbose=0, foldnum=0):
        """
        Trains a model using keras API, after scaling the data
        """
        from sklearn.preprocessing import StandardScaler
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

    def runDNNAnalysis(self, X_train, y_train, batch_size, epochs, foldnum=0):
        # import keras
        import innvestigate
        import innvestigate.utils as iutils

        history, ZScaler = self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, foldnum=foldnum)

        # Getting all the samples that can be correctly predicted
        # Note: Samples have already been scaled
        all_samples, _labels, correct_pred_idxs, final_acc = self.getCorrectPredictions(model=self.model, ZScaler=ZScaler)

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
        final_acc, lrp_results, correct_pred_idxs = self.runDNNAnalysis(X_train, y_train, 
                                                    epochs=epochs, batch_size=batch_size, foldnum=foldnum)

        return (lrp_results, correct_pred_idxs)

    def cross_validation(self, batch_size, epochs, num_folds = 10, parallel=True):
        # Populate Zoo here perhaps
        # Use the h5 weight files...
        # self.model_zoo.append(model)
        
        start_time = time()
        
        histories = []
        testing_indxs =[]
        predictions = []
        true_labels = []
        cv_lrp_results = []
        zoo = []
        results = []

        num_procs = 4

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

        for lrp_results, correct_idxs in results:
            cv_lrp_results.extend(lrp_results)
            testing_indxs.extend(correct_idxs)

        return (cv_lrp_results, testing_indxs)

    def train_model(self, batch_size=20, epochs=200, cross_validation=False):
        
        self.model_zoo = []

        if cross_validation:
            self.cross_validation(batch_size,epochs)
        else:
            X_train, y_train = self.train_set.features, self.train_set.labels
            final_acc, lrp_results, correect_preds = self.runDNNAnalysis(X_train, y_train, epochs=epochs, batch_size=batch_size)
            #Plot LRP
        return

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
        
        #FIXME: If every DNN gives an incorrect output
        predictions = []
        for i,zscaler in enumerate(self.zscalers):
            _samples = zscaler.transform(self.val_set.features)
            model_preds = np.array([(np.argmax(x), np.max(x)) for x in self.foldmodel(i).predict(_samples)])     
            incorrect = model_preds[:,0] != self.val_set.labels
            model_preds[incorrect] = -1
            predictions.append(model_preds)

        # Combine all DNN predictions into a matrix
        predictions = np.stack(predictions, axis=1)

        print(predictions)

        # The DNN number with the highest confidence
        # One for each sample
        best_DNN = np.argmax(predictions[:,:,1], axis=1)
        best_predictions = predictions[range(len(self.val_set.labels)), best_DNN,0].astype(int)

        return best_predictions, best_DNN

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
                              samples=None,
                              labels=None, ZScaler=None):
        '''
        Assumes categorical output from DNN
        Will default to getting correct predcitions from the validation set
        using the current tained model
        '''
        import numpy as np
        
        if samples == None and labels==None:
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
        
        # correct_predictions = 

        return samples[correct_idxs], labels[correct_idxs], correct_idxs, loss_and_metrics


######## HELPER FUNCTIONS ############

'''
Expects data to be 2D numpy array
'''
def calculateEntropy(data, plot=False):
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