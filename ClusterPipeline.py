from common_imports import *
from helper import *

# from multiprocessing import Pool
# from multiprocess import Pool
from time import time

# num_procs = 4

# if __name__ == "__main__":
#     pass
# else:
#     POOL = Pool(num_procs)
#     print("Declared pool worker...")

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

        # Fold counter to keep track of which model is being trained
        # Each fold should train a fresh model and save its weights
        self.foldctr = 0

    def train(self, X, y, batch_size, epochs, X_test=[], y_test=[], verbose=0, foldctr=0):
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
        
        self.model.save_weights(
            "{basename}_{id}.h5".format(basename=self.BASENAME, id=foldctr)
        )

        return history, ZScaler

    def runDNNAnalysis(self, X_train, y_train, batch_size, epochs, foldctr=0):
        import innvestigate
        import innvestigate.utils as iutils
        
        history, ZScaler = self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, foldctr=foldctr)

        # Getting all the samples that can be correctly predicted
        # Note: Samples have already been scaled
        all_samples, _labels, correct_pred_idxs, final_acc = self.getCorrectPredictions(model=self.model, ZScaler=ZScaler)

        # Stripping the softmax activation from the model
        model_w_softmax = self.model
        model_softmax_stripped = iutils.keras.graph.model_wo_softmax(model_w_softmax)

        # Creating an analyzer
        lrp_E = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(
                model=model_softmax_stripped, epsilon=1e-3)

        lrp_results = lrp_E.analyze(all_samples)
        
        print("Finished training Fold: {} -> Loss:{:0.3f}, Acc:{:.4f}".format(foldctr, *final_acc))
        return (final_acc, lrp_results, correct_pred_idxs)

    
    def cross_validation(self, batch_size, epochs, num_folds = 10):
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

        num_procs = 3

        pool_args = [(self, fnum, tr_idx, tst_idx, batch_size, epochs) for fnum, (tr_idx, tst_idx) in enumerate(self.getKFold(n_splits=num_folds))]
         
        # # if __name__ == '__main__':
        # print("Running Pool...")
        # results = pool.map(lambda _args : runFoldWorker(*_args), pool_args)
        
        # print("Running Serial Crossvalidation")
        for _args in pool_args:
            results.append(runFoldWorker(*_args))

        print("Runtime: {:.3f}s".format(time()-start_time))

        return (cv_lrp_results, testing_indxs)

    def train_model(self, batch_size=20, epochs=200, cross_validation=False):
        
        self.model_zoo = []

        if not cross_validation:
            X_train, y_train = self.train_set.features, self.train_set.labels
            final_acc, lrp_results = self.runDNNAnalysis(X_train, y_train, epochs=epochs, batch_size=batch_size)

            #Plot LRP
        else:
            self.cross_validation(batch_size,epochs,num_folds=4)
        return


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

def runFoldWorker(pipeline, foldnum, train_index, test_index, batch_size, epochs):
    print("Inside worker:",foldnum)
    X_train = pipeline.train_set.features.iloc[train_index]
    y_train = pipeline.train_set.labels.iloc[train_index]
    X_test  = pipeline.train_set.features.iloc[test_index]
    y_test = pipeline.train_set.labels.iloc[test_index]
    final_acc, lrp_results, correct_pred_idxs = pipeline.runDNNAnalysis(X_train, y_train, 
                                                epochs=epochs, batch_size=batch_size, foldctr=foldnum)

    return (lrp_results, correct_pred_idxs)
