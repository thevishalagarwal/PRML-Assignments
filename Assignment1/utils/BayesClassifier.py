import numpy as np
from scipy.stats import multivariate_normal

class BayesClassifier:
    
    def __init__(self):
        self.mean = None
        self.covariance = None
        self.apriori = None
        self.n_class = None
        self.classes = None
        
    def fit(self, X, y):
        print '\nFitting your data...'
        
        m, n = X.shape
        self.classes, count_class = np.unique(y, return_counts=True)
        self.n_class = self.classes.shape[0]
        
        self.mean = np.empty((0, n))
        self.covariance = np.empty((0, n, n))
        self.apriori = np.array([])
        
        for i in range(self.n_class):
            index = np.where(y==self.classes[i])[0]
            x = X[index, :]
            
            u = np.mean(x, axis=0, keepdims=True)
            self.mean = np.append(self.mean, u, axis=0)
            
            cov = (np.cov(x, rowvar=False)).reshape((1,n,n))
            self.covariance = np.append(self.covariance, cov, axis=0)
            
            p = x.shape[0] / (m*1.0)
            self.apriori = np.append(self.apriori, p)
        
        print 'Successfully completed fitting!\n'
    
    def predict(self, X):
        print '\nPredicting on your test data...'
        m, n = X.shape
        posterior = np.empty((0, m))
        
        if len((self.covariance).shape)==2:
            for i in range(self.n_class):
                p = (multivariate_normal.pdf(X, mean=self.mean[i,:], cov=self.covariance))*self.apriori[i]
                p = p.reshape((1,m))
                posterior = np.append(posterior, p, axis=0)
        else:
            for i in range(self.n_class):
                p = (multivariate_normal.pdf(X, mean=self.mean[i,:], cov=self.covariance[i,:,:]))*self.apriori[i]
                p = p.reshape((1,m))
                posterior = np.append(posterior, p, axis=0)
            
        label = np.argmax(posterior, axis=0)
        
        for i in range(self.n_class):
            index = np.where(label==i)[0]
            label[index] = self.classes[i]
        
        print'Prediction completed!\n'
        return label
    
    def evaluate(self, y_pred, y_test, show=False):
        print 'Evaluating your model...'
        print 'Computing accuracy...'
        
        y_pred = y_pred.reshape((y_pred.shape[0],))
        y_test = y_test.reshape((y_test.shape[0],))
        
        m = y_pred.shape[0]
        error = np.ones(shape=y_pred.shape)
        
        index = (y_pred==y_test)
        error[index] = 0
        
        error_count = np.sum(error)
        
        accuracy = (m-error_count)/(m*1.0)
        
        confusion_matrix = np.zeros((self.n_class, self.n_class), dtype=int)

        for i in range(self.n_class):
            index = np.where(y_test==self.classes[i])[0]
            pred = y_pred[index]
            for j in range(self.n_class):
                confusion_matrix[i,j] = list(pred).count(self.classes[j])
        
        return accuracy, confusion_matrix