# Mini-batched SGD

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# return the activation value for input x
def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

def predict(X,W):
	# dot product between features and weights
	predictions = sigmoid_activation(X.dot(W))

	# apply threshold (step function)
	# binary class labels
	predictions[predictions <= 0.5] = 0
	predictions[predictions > 0] = 1

	return predictions

def next_batch(data, labels, batchSize):
	for i in np.arange(0, data.shape[0], batchSize):
		yield (data[i:i + batchSize], labels[i:i + batchSize])


# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="Number of epochs (number of times you go through the training set")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Rate of learning")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="Size of Stochastic Gradient Descent batches")
args = vars(ap.parse_args())


# Let's generate a classification problem.
(data, labels) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
labels = labels.reshape((labels.shape[0],1))
# Insert a column of 1's, which will allow us to have our bias (b) within our weights (W)
# This is the Bias Trick
data = np.c_[data, np.ones((data.shape[0]))]

# 50 / 50 split
(train_data, test_data, train_labels, test_labels ) = train_test_split(data, labels, test_size=0.5, random_state=42)


# Weight matrix
weights = np.random.randn(data.shape[1], 1)
# Losses to be plotted
losses = []

# Train
print("[INFO] training...")
for epoch in np.arange(0, args["epochs"]):
	epoch_loss = []
	for (batch_data, batch_labels) in next_batch(data, labels, args["batch_size"]):
		predictions = sigmoid_activation(batch_data.dot(weights))

		error = predictions - batch_labels
		epoch_loss.append(np.sum(error ** 2))

		gradient = batch_data.T.dot(error)

		weights += -args["alpha"] * gradient

	loss= np.average(epoch_loss)
	losses.append(loss)

	if epoch == 0 or (epoch+1)%5 == 0:
		print("[INFO] epoch={}, loss={:.7f}".format(int(epoch+1), loss))

# Evaluate
print("[INFO] evaluating...")
predictions = predict(test_data, weights)
print(classification_report(test_labels, predictions))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(test_data[:, 0], test_data[:, 1], marker="x", c=test_labels[:,0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()