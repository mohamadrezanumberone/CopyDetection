import numpy as np
from numpy import loadtxt
import os
import glob
import operator
from scipy.stats import spearmanr
from scipy import stats


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
		cluster_centroid = loadtxt(address[i] + 'cluster_centroids.txt', delimiter=',', unpack=False)

		# Load cluster names
		text_file = open(address[i] + 'cluster_name.txt', "r")
		cluster_name = text_file.read().split('\n')

		# Load data
		samples = loadtxt(samples_address[i], delimiter=',', unpack=False)

		"""
		Assign To Cluster
		"""
		# Assign each frame vector to nearest centroid in hierarchical order
		if not os.path.exists(address[i] + 'BlockBasedApproach'):
			os.makedirs(address[i] + 'BlockBasedApproach')

		for index in range(100, samples.shape[1] + 1, 100):
			sample_by_block = samples[:, :index]
			cluster_centroid_by_block = cluster_centroid[:, :index]
			assigned_cluster = assign_to_cluster(sample_by_block, cluster_name, cluster_centroid_by_block)
			with open(address[i] + 'BlockBasedApproach/cluster_assigned_block_' + str(index) + '.txt', 'w+') as file_handle:
				for item in assigned_cluster:
					file_handle.write('%s\n' % item)

		for index in range(100, samples.shape[1] + 1, 100):
			text_file = open(address[i] + 'BlockBasedApproach/cluster_assigned_block_' + str(index) + '.txt', "r")
			block_based_cluster_name = text_file.read().split('\n')

			text_file = open(address[i] + 'BlockBasedApproach/cluster_assigned_block_500.txt', "r")
			actual_cluster_name = text_file.read().split('\n')

			report = list()
			for j in range(3, 8, 2):
				block_based_cluster_name_level = [x[2:j] for x in block_based_cluster_name]
				actual_cluster_name_by_level = [x[2:j] for x in actual_cluster_name]
				corr, p_value = spearmanr(actual_cluster_name_by_level, block_based_cluster_name_level)
				tau, p_value = stats.kendalltau(actual_cluster_name_by_level, block_based_cluster_name_level)
				temp = 'block size = ' + str(index) + ' | level #' + str(int((j-1)/2)) + ': corr = ' + str(corr) + ' p_value = ' + str(p_value) + '\n'
				temp += 'block size = ' + str(index) + ' | level #' + str(int((j-1)/2)) + ': tau =  ' + str(tau) + ' p_value = ' + str(p_value) + '\n'
				temp += '__________________________________________________________'
				report.append(temp)

			with open(address[i] + 'BlockBasedApproach/report.txt', 'a+') as file_handle:
				for item in report:
					file_handle.write('%s\n' % item)

