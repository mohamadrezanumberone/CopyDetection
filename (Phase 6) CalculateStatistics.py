import numpy as np
from numpy import loadtxt
import os
import glob
import matplotlib.pyplot as plt
from shutil import copyfile
import operator


def calculate_mean_var(unique_index, assigned_cluster, samples):
	mean_array = list()
	var_array = list()
	for i in range(0, len(unique_index)):
		indices = [index for index, x in list(enumerate(assigned_cluster)) if x == unique_index[i]]
		elements = [vector for index, vector in list(enumerate(samples)) if index in indices]
		elements_mean = np.mean(elements, axis=0)
		elements_var = np.var(elements, axis=0)
		mean_array.append(elements_mean)
		var_array.append(elements_var)
	return mean_array, var_array


def assign_to_cluster(samples, cluster_name, cluster_centroid):
	# Find assigned cluster for each vector
	assigned_cluster = []
	for i in range(0, len(samples)):
		item = samples[i]
		cluster_levels = 3
		index = 0
		index_new = '1'
		while cluster_levels != 0:
			distance = []
			if cluster_levels != 3:
				index = cluster_name.index(index_new)
			distance.append(np.linalg.norm(item - cluster_centroid[index]))
			distance.append(np.linalg.norm(item - cluster_centroid[index + 1]))
			distance.append(np.linalg.norm(item - cluster_centroid[index + 2]))
			index, _ = min(enumerate(distance), key=operator.itemgetter(1))
			index_new += "." + str(index + 1)
			cluster_levels -= 1
		assigned_cluster.append(index_new)
	return assigned_cluster


if __name__ == '__main__':

	address = ['Kmeans_reducedFeatures/', 'Kmeans_reducedFeatures_normal/', 'Kmeans_reducedFeatures_standard_scaler/']
	samples_address = ['Reduced Features/reducedFeatures.txt', 'Reduced Features/reducedFeatures_normal.txt', 'Reduced Features/reducedFeatures_standard_scaler.txt']
	for i in range(0, len(address)):
		"""
		Load Necessary Data
		"""
		# Load cluster centroids
		cluster_centroid = loadtxt(address[i] + 'cluster_centroids.txt',delimiter=',', unpack=False)

		# Load cluster names
		text_file = open(address[i] + 'cluster_name.txt', "r")
		cluster_name = text_file.read().split('\n')

		# Load data
		samples = loadtxt(samples_address[i], delimiter=',', unpack=False)

		# Load file names
		file_name = loadtxt('Features/feature_file_name.txt', delimiter=',', unpack=False)

		"""
		Assign To Cluster
		"""
		# Assign each frame vector to nearest centroid in hierarchical order
		assigned_cluster = assign_to_cluster(samples, cluster_name, cluster_centroid)

		"""
		Copy Frame Number File, Frame Features File To Cluster Folder
		Create A File To Save Assigned Clusters
		"""
		copyfile('Features/feature_file_name.txt', address[i] + 'frame_number.txt')
		copyfile('Reduced Features/reducedFeatures.txt', address[i] + 'samples.txt')
		with open(address[i] + 'cluster_assigned.txt', 'w+') as file_handle:
			for item in assigned_cluster:
				file_handle.write('%s\n' % item)

		"""
		Clusters Mean And Variance
		"""
		if not os.path.exists(address[i] + 'Statistics'):
			os.makedirs(address[i] + 'Statistics')
		# Calculate variance and mean for each cluster
		cluster_mean_array = list()
		cluster_var_array = list()
		cluster_by_level = list()
		for index in range(1, 8, 2):
			assigned_cluster_by_level = [x[:index] for x in assigned_cluster]
			unique_index = list(sorted(set(assigned_cluster_by_level)))
			cluster_mean, cluster_var = calculate_mean_var(unique_index, assigned_cluster_by_level, samples)
			cluster_mean_array += cluster_mean
			cluster_var_array += cluster_var
			cluster_by_level += unique_index

		np.savetxt(address[i] + 'Statistics/cluster_mean.txt', cluster_mean_array, delimiter=',', newline='\n')
		np.savetxt(address[i] + 'Statistics/cluster_var.txt', cluster_var_array, delimiter=',', newline='\n')

		with open(address[i] + 'Statistics/cluster_name_by_level.txt', 'w+') as filehandle:
			for item in cluster_by_level:
				filehandle.write('%s\n' % item)
		# Save image from mean bar plot for each level of kmeans tree
		for k in range(0, len(cluster_by_level)):
			x = [j for j in range(0, len(cluster_mean_array[k]))]
			plt.ioff()
			fig = plt.figure(figsize=(13.0, 10.0))
			fig.add_subplot(111).bar(x, cluster_mean_array[k])
			fig.add_subplot(111).set_xlabel('Features Index')
			fig.add_subplot(111).set_ylabel('Individual Feature Mean')
			fig.savefig(address[i] + 'Statistics/' + cluster_by_level[k].replace(".", "_")+' Mean.png', dpi=100)
			plt.close(fig)

		# Save image from variance plot for each level of kmeans tree
		for k in range(0, len(cluster_by_level)):
			x = [j for j in range(0, len(cluster_var_array[k]))]
			plt.ioff()
			fig = plt.figure(figsize=(13.0, 10.0))
			fig.add_subplot(111).plot(x, cluster_var_array[k])
			fig.add_subplot(111).set_xlabel('Features Index')
			fig.add_subplot(111).set_ylabel('Individual Feature Mean')
			fig.savefig(address[i] + 'Statistics/' + cluster_by_level[k].replace(".", "_") + ' Variance.png', dpi=100)
			plt.close(fig)
