from numpy import loadtxt
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def dimension_reduction():
	# Load data from features directory
	feature_directory = 'Features'
	lines = loadtxt(feature_directory + "/features.txt", delimiter=',', unpack=False)

	# set up PCA module
	pca = PCA(n_components=500)

	# move features to PCA space and get the results consist of 500 best components
	pca_transformed = pca.fit_transform(lines)

	if not os.path.exists('Reduced Features'):
		os.makedirs('Reduced Features')

	# save reduced features
	reduced_feature_directory = 'Reduced Features'
	np.savetxt(reduced_feature_directory + '/reducedFeatures.txt', pca_transformed, delimiter=',', newline='\n')

	# normalize data with z-norm
	elements_mean = np.mean(pca_transformed, axis=0)
	elements_var = np.var(pca_transformed, axis=0)
	normal_data = list()
	for i in range(0, pca_transformed.shape[0]):
		vector = pca_transformed[i]
		vector = [a - b for a, b in zip(vector, elements_mean)]
		vector = [a / b for a, b in zip(vector, elements_var)]
		normal_data.append(vector)

	# save normalized reduced features
	normal_data = np.array(normal_data)
	np.savetxt(reduced_feature_directory + '/reducedFeatures_normal.txt', normal_data, delimiter=',', newline='\n')

	# calculate standard scaler
	scaler = StandardScaler()
	scaled_data = scaler.fit_transform(pca_transformed)
	np.savetxt(reduced_feature_directory + '/reducedFeatures_standard_scaler.txt', scaled_data, delimiter=',', newline='\n')

if __name__ == '__main__':
	dimension_reduction()
