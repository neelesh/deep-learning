import numpy as np

class Perceptron:

	def __init__(self, N, alpha=0.1):
		# weight matrix
		self.W = np.random.rand(N+1)/np.sqrt(N)
		#learning rate
		self.alpha = alpha

	def step(self, x):
		# step function
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs=10):
		# insert a column of 1's in the last column
		# allows us to use the bias's as a traninable paramter within the W matrix
		X = np.c_[X, np.ones((X.shape[0]))]

		for epoch in np.arange(0, epochs):
			# loop over each data point
			for (x, target) in zip(X, y):
				# Prediction:
				# dot product between the input features and W matrix, then step function
				p = self.step(np.dot(x, self.W))

				# only update p if our prediction does not match the target.
				if p != target:
					# determine the error and update W matrix
					error = p - target
					self.W += -self.alpha * error * x

	def predict(self, X, addBias=True):
		# make sure the input is 2D
		X = np.atleast_2d(X)

		if addBias:
			# We wish to have a column of 1's which will go on to be our biases
			X = np.c_[X, np.ones((X.shape[0]))]

		# Take the dp of the input features and W matrix
		# Then pass through step function
		return self.step(np.dot(X, self.W))