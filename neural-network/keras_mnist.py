from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

# argument parsing
ap = argparse.ArgumentParser()

# output is where to save our figure.
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# Get the MNIST dataset (0-9 handwritten digits)
print("[INFO] loading MNIST dataset")
dataset = datasets.fetch_mldata("MNIST Original")

# scale the raw pixel intensities between 0 to 1.0.
data = dataset.data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

# turn integer labels into vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))  # fully connected
model.add(Dense(128, activation="sigmoid"))  # learn 128 weights
model.add(Dense(10, activation="softmax"))  # 10 more weights on the final layer

# training using Stochastic Gradient Descent
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])  # cross-entropy loss function
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# Evaluate Network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
     predictions.argmax(axis=1),
     target_names=[str(x) for x in lb.classes_]))

# plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
