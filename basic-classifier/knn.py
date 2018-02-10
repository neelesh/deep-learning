from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# Argument parsing - python knn.py --dataset ./animals
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="dataset path")
ap.add_argument("-k", "--neighbours", type=int, default=1, help="Number of nearest neighbours for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="Number of jobs for k-NN distance (-1 uses all cores)")
args = vars(ap.parse_args())

# Store the paths to the images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Create an image preprocessor and load the dataset
simple_preprocessor= SimplePreprocessor(32,32)
loader = SimpleDatasetLoader(preprocessors = [simple_preprocessor])

(data, labels) = loader.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Print memory consumption
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))


# Encode labels as integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Partition the data into training and testing, 75% - 25%
# trainX, testX - The training and testing data
# trainY, testY - the training and testing labels
(train_data, test_data, train_labels, test_labels ) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Training time.
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbours"], n_jobs=args["jobs"])
model.fit(train_data, train_labels)
print(classification_report(test_labels, model.predict(test_data), target_names=encoder.classes_))