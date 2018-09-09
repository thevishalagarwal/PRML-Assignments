import numpy as np
from scipy import linalg as la


class PCA():

	def __init__(self):
		self.weights = None

	def fit(self, X, n_components=0):

		if n_components==0:
			n_components = X.shape[1]

		cov = np.cov(X, rowvar=False)
		
		eigen_val, eigen_vec = la.eigh(cov)
		
		idx = np.argsort(eigen_val)[::-1]
		eigen_vec = eigen_vec[:,idx]
		eigen_val = eigen_val[idx]

		self.weights = eigen_vec[:,:n_components]

	def transform(self, X):

		return np.dot(X, self.weights)


class LDA():

	def __init__(self):
		self.weights = None

	def fit(self, X, y, n_components=0):

		m, n = X.shape

		if n_components==0:
			n_components = n

		classes = np.unique(y, return_counts=False)
		num_classes = classes.shape[0]
		
		# Within-class scatter
		sw = np.zeros((n,n))
		
		for i in range(num_classes):
			index = np.where(y==i)[0]
			x = X[index,:]
			sw += (np.cov(x, rowvar=False)*(x.shape[0]-1))

		# Between-class scatter
		sb = np.zeros((n,n))

		mean = np.mean(X, axis=0, keepdims=True)

		for i in range(num_classes):
			index = np.where(y==i)[0]
			x = X[index,:]
			mean_i = np.mean(x, axis=0, keepdims=True)

			sb += ( x.shape[0] * np.dot( (mean_i - mean).T, (mean_i - mean) ) )

		s = np.dot(la.inv(sw), sb)

		eigen_val, eigen_vec = la.eigh(s)

		idx = np.argsort(eigen_val)[::-1]
		eigen_vec = eigen_vec[:,idx]
		eigen_val = eigen_val[idx]

		self.weights = eigen_vec[:,:n_components]


	def transform(self, X):

		return np.dot(X, self.weights)

