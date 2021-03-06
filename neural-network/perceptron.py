from nn import Perceptron
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# # OR labels
# y = np.array([[0], [1], [1], [1]])

# # AND labels
# y = np.array([[0], [0], [0], [1]])

# # NOR dataset
# y = np.array([[1], [0], [0], [0]])

# # NAND dataset
y = np.array([[1], [1], [1], [0]])

# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# evaluate
print("[INFO] testing perceptron...")

# loop over the data points
for (x, target) in zip(X, y):
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))
