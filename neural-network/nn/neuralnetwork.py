import numpy as np


class NeuralNetwork:
	def __init__(self, layers, alpha=0.1):
		self.W = []
		self.layers = layers
		self.alpha = alpha

		# loop from first layer to n-2
		for i in np.arange(0, len(layers) - 2):
			# randomly initialise W that connect each layer
			w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)  # assuming bias trick
			self.W.append(w / np.sqrt(layers[i]))

		# last 2 layers are special (last layer + biases)
		# input connections need bias, but the output does not
		w = np.random.randn(layers[-2] + 1, layers[-1])
		self.W.append(w / np.sqrt(layers[-2]))

	def __repr__(self):
		# string that represents network
		return "NeuralNetwork: {}".format(
			"-".join(str(l) for l in self.layers))

	def sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		# Calculate derivative of the sigmoid
		# Assuming x has already been passed through sigmoid function!
		return x * (1 - x)

	def fit(self, X, y, epochs=1000, display_update=100):
		# column of 1’s as the last entry in the feature
		# matrix allows us to treat the bias
		# as a trainable parameter within the weight matrix
		X = np.c_[X, np.ones((X.shape[0]))]

		for epoch in np.arange(0, epochs):
			# train
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)

			# display training information when complete
			if epoch == 0 or (epoch + 1) % display_update == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] epoch={}, loss={:.7f}".format(
					epoch + 1, loss))

	def fit_partial(self, x, y):
		A = [np.atleast_2d(x)]

		# FEEDFORWARD:
		# for each layer in the network
		for layer in np.arange(0, len(self.W)):
			net = A[layer].dot(self.W[layer])
			out = self.sigmoid(net)
			A.append(out)

		# BACKPROPOGATE
		# compute difference between 'prediction' and truth
		error = A[-1] - y

		# chain rule deltas initialise
		D = [error * self.sigmoid_deriv(A[-1])]

		# loops over network layers backwards
		# ignore last two layers in the network (final layer + bias)
		for layer in np.arange(len(A) - 2, 0, -1):
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

		# reverse the deltas, because they're currently backwards
		D = D[::-1]

		# WEIGHT UPDATE
		for layer in np.arange(0, len(self.W)):
			# apply the deltas to the layers
			# while applying a small learning rate
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

	def predict(self, X, addBias=True):
		p = np.atleast_2d(X)

		if addBias:
			# insert column of 1's for the bias'
			p = np.c_[p, np.ones((p.shape[0]))]

		# for each layer in the network
		for layer in np.arange(0, len(self.W)):
			# compute output prediction
			p = self.sigmoid(np.dot(p, self.W[layer]))
		return p

	def calculate_loss(self, X, targets):
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		loss = 0.5 * np.sum((predictions - targets) ** 2)

		return loss