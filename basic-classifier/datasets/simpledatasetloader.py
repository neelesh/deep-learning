import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose = -1):
		# features and labels
		data = []
		labels = []

		# for each input image
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the label
			image = cv2.imread(imagePath)

			# assuming dataset_path/{class}/{image}.jpg
			label = imagePath.split(os.path.sep)[-2]

			# in the event of preprocessors
			if self.preprocessors is not None:
				# apply the each preprocess on the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# the preprocessed image is now a feature vector
			# add it to the data array, along side it's label
			data.append(image)
			labels.append(label)

			# verbose will allow us to print information every X iteration.
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print ("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

		# return tuple (data, labels)
		return (np.array(data), np.array(labels))