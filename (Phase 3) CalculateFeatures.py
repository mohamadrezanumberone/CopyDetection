from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import glob
from numpy import loadtxt


def calculate_features(directory, sample_count):
	conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
	datagen = ImageDataGenerator(rescale=1. / 255)
	batch_size = 10
	features = np.zeros(shape=(int(sample_count / 10), 4 * 4 * 512))
	generator = datagen.flow_from_directory(
			directory,
			target_size=(150, 150),
			batch_size=batch_size,
			class_mode='categorical')
	i = 0
	n = 0
	for inputs_batch, labels_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		features_batch = np.reshape(features_batch, (batch_size, 4 * 4 * 512))
		mean_array = np.mean(features_batch, axis=0)
		features[i] = mean_array
		i += 1
		if i * batch_size >= sample_count:
			break
		n += 10
		print("%.2f Percent Completed So Far! Processed Number: %d" % ((n / sample_count)*100, n))
	return features

if __name__ == '__main__':
	base_directory = 'D:/Lesson/Master Degree/Thesis/Project/BaseLine Project Python/'
	patched_frames_directory = os.path.join(base_directory, 'Patchted Frames/')

	all_frames_number = len(glob.glob1(patched_frames_directory + 'Class 0', "*.jpg"))

	frame_features = calculate_features(patched_frames_directory, all_frames_number)

	feature_directory = os.path.join(base_directory, 'Features')
	if not os.path.exists(feature_directory):
		os.makedirs(feature_directory)

	np.savetxt(os.path.join(feature_directory, 'features.txt'), frame_features, delimiter=',', newline='\n', fmt='%1.10f')

	frames_folder = os.path.join(base_directory, 'Frames')
	files = glob.glob(os.path.join(frames_folder, '*.jpg'))
	file_names = [[0 for x in range(2)] for y in range(len(files))]
	for i in range(0, len(files)):
		temp = os.path.splitext(os.path.basename(files[i]))[0]
		temp = temp.split("_")
		file_names[i][0] = int(temp[0])
		file_names[i][1] = int(temp[1])

	file_names = sorted(file_names, key=lambda x: (x[0], x[1]))
	with open(base_directory+'Features/feature_file_name.txt', 'w+') as filehandle:
		for item in file_names:
			filehandle.write('%s\n' % ','.join(str(x) for x in item))
