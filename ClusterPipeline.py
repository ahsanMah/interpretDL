from common_imports import *
from helper import *

class ClusterPipeline:
    """
        model: A keras deep neural network
        data: Original dataset which will be used for training and validation
    """


    class dfHotEncoder(BaseEstimator, TransformerMixin):
        """
        Builds a hot encoder froma pandas dataframe
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
    

    def __init__(self, model, data,
                analyzer="lrp",
                cluster_algo = "HDBSCAN",
                linearClassifier="SVM"):
        """
        model: keras object
        data: [training_data, validation_data]
        """
        self.model = model
        self.train_set, self.val_set = data

        self.hot_encoder = dfHotEncoder()
        self.hot_encoder.fit(self.train_set[1]) #train_set=[X_train,y_train]
        self.history = None
    
    def train(self, model, X, y, batch_size, epochs, X_test=[], y_test=[], verbose=1, plot=True):
        
        ZScaler = StandardScaler().fit(X)
        
        X_train = ZScaler.transform(X)
        y_train = self.hot_encoder.transform(y)

        history = model.fit(X_train, y_train,
                        epochs=epochs, batch_size = batch_size, verbose=verbose)
        
    #     if plot: plot_history(history)
        
        return history, ZScaler



    def train_model(self, batch_size=20, epochs=200, cross_validation=False):
        
        if not cross_validation:
            X_train, y_train = self.train_set
            history, ZScaler = self.train(self.model, X_train,y_train, batch_size=batch_size, epochs=epochs)

            self.getCorrectPredictions(self.model, ZScaler)
        else:
            pass
        return


    def split_valid(self, features, original_labels, training_labels, valid_size=0.5):
        train_index, validation_index = get_split_index(features, original_labels, test_size=valid_size)[0]
        
        X_valid, y_valid, y_valid_original = features.iloc[validation_index], training_labels.iloc[validation_index], original_labels.iloc[validation_index]
        X_train, y_train, y_original = features.iloc[train_index], training_labels.iloc[train_index], original_labels.iloc[train_index]
        
        return X_train, y_train, y_original, X_valid, y_valid, y_valid_original


    '''
    Assumes categorical output from DNN
    Will always get correct predcitions from the validation set
    using the current tained model
    '''
    def getCorrectPredictions(self, model, ZScaler=None):
        
        import numpy as np
        
        samples = self.val_set[0]
        if ZScaler: samples = ZScaler.transform(samples)
        
        labels = self.hot_encoder.transform(self.val_set[1])

        predictions = model.predict(samples)
        preds = np.array([np.argmax(x) for x in predictions])
        true_labels = np.array([x for x in labels])

        correct = preds == true_labels

        print("Prediction Accuracy")
        loss_and_metrics = model.evaluate(samples, labels)
        print("Scores on data set: loss={:0.3f} accuracy={:.4f}".format(*loss_and_metrics))
        
        # correct_predictions = 

        return samples[correct], labels[correct], correct