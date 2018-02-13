from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from datasets import SimpleDatasetLoader
from preprocessing import SimplePreprocessor
import argparse

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",required=True, help="path to dataset")
args = vars(ap.parse_args())


# Grab the list of image paths
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

# Load and preprocess images
preprocessor = SimplePreprocessor(32,32)
loader = SimpleDatasetLoader(preprocessors=[preprocessor])
(data, labels) = loader.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Encode the labels as integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)


(train_data, test_data, train_labels, test_labels ) = train_test_split(data, labels, test_size=0.25, random_state=5)

# Regularised training
for r in (None, "l1", "l2"):
	print("[INFO] training a model with '{}' penalty".format(r))
	model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=42,)
	model.fit(train_data, train_labels)

	# evaluate
	score = model.score(test_data,test_labels)
	print("[INFO] `{}` penalty accuracy: {:.2f}%".format(r,score*100))