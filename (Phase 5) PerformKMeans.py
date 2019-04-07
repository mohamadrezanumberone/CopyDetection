from numpy import loadtxt
from sklearn.cluster import KMeans
import numpy as np
import os
import shutil
import glob


class groupby(object):
	# [k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B
	# [list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D
	def __init__(self, iterable, key=None):
		if key is None:
			key = lambda x: x
		self.keyfunc = key
		self.it = iter(iterable)
		self.tgtkey = self.currkey = self.currvalue = object()

	def __iter__(self):
		return self

	def __next__(self):
		while self.currkey == self.tgtkey:
			self.currvalue = next(self.it)  # Exit on StopIteration
			self.currkey = self.keyfunc(self.currvalue)
		self.tgtkey = self.currkey
		return (self.currkey, self._grouper(self.tgtkey))

	def _grouper(self, tgtkey):
		while self.currkey == tgtkey:
			yield self.currvalue
			self.currvalue = next(self.it)  # Exit on StopIteration
			self.currkey = self.keyfunc(self.currvalue)


def itemgetter(*items):
	if len(items) == 1:
		item = items[0]

		def g(obj):
			return obj[item]
	else:
		def g(obj):
			return tuple(obj[item] for item in items)
	return g


def load_data(directory):
	samples = loadtxt(directory, delimiter=',', unpack=False)
	return samples


def load_files():
	reduced_feature_directory = 'Reduced Features'
	file_address = glob.glob(reduced_feature_directory + "/reducedFeatures*.txt")
	return file_address


def savecentroids(centroids, depth, file_name):
	centroids_string = []
	centroids_depth = []
	for i in range(0, len(centroids)):
		temp = ','.join(str(x) for x in centroids[i])
		centroids_string.append(temp)
		centroids_depth.append(depth)
	kmeans_directory = 'Kmeans_'
	file_name = file_name[file_name.rfind('\\') + 1:file_name.rfind('.')] + '/'

	kmeans_directory += file_name
	if not os.path.exists(kmeans_directory):
		os.makedirs(kmeans_directory)
		with open(os.path.join(kmeans_directory, 'cluster_centroids.txt'), 'w+') as filehandle:
			for item in centroids_string:
				filehandle.write('%s\n' % item)
		with open(os.path.join(kmeans_directory, 'cluster_name.txt'), 'w+') as filehandle:
			for item in centroids_depth:
				filehandle.write('%s\n' % item)
	else:
		with open(os.path.join(kmeans_directory, 'cluster_centroids.txt'), 'a+') as filehandle:
			for item in centroids_string:
				filehandle.write('%s\n' % item)
		with open(os.path.join(kmeans_directory, 'cluster_name.txt'), 'a+') as filehandle:
			for item in centroids_depth:
				filehandle.write('%s\n' % item)


def perform_kmeans(samples, number_of_clusters, layers, layer_number, depth, file_name):
	if layer_number == layers:
		return

	kmeans = KMeans(n_clusters=number_of_clusters, n_init=10, max_iter=300)

	labels = np.array([kmeans.fit_predict(samples)])

	# Add cluster label at the end of the sample feature
	# samples = np.concatenate((samples, labels.T), axis=1)

	# Grouping samples into distinct clusters
	# clusters = [list(g) for _, g in groupby(sorted(samples, key=itemgetter(samples[0].__len__() - 1)), itemgetter(samples[0].__len__() - 1))]

	unique_labels = np.unique(labels[0])

	clusters = list()
	for unique_label in np.nditer(unique_labels):
		# indices = [index for index, x in enumerate(labels) if x == unique_label]
		indices = np.where(labels[0] == unique_label)[0]
		# elements = [samples[k] for k in indices]
		elements = samples[indices]
		clusters.append(elements[:])

	# Get centroids from cluster
	centroids = kmeans.cluster_centers_

	savecentroids(centroids, depth, file_name)

	# iterate in clusters
	for i in range(0, clusters.__len__()):
		perform_kmeans(clusters[i], number_of_clusters, layers, layer_number + 1,  depth+"."+str(i+1), file_name)


if __name__ == '__main__':
	reduced_feature_address = load_files()
	for i in range(0, len(reduced_feature_address)):
		samples = load_data(reduced_feature_address[i])
		perform_kmeans(samples, 3, 3, 0, '1', reduced_feature_address[i])
