import cv2
import os
import glob
from threading import Thread
import time


def patching_frames(directory, start, end, thread_id):
	patched_frames_folder = 'Patchted Frames/Class 0/'
	if not os.path.exists(patched_frames_folder):
		os.makedirs(patched_frames_folder)
	files = glob.glob(os.path.join(directory, '*.jpg'))
	files = files[start:end]
	start_time = time.time()
	for i in range(0, len(files)):
		file_name = (os.path.basename(files[i]))[:-4]
		image = cv2.imread(files[i])
		height, width, channels = image.shape
		patch_width = int(width / 2)
		patch_height = int(height / 2)
		overlap = int(width * 0.15)
		all_patches = []
		patch_1 = image[0: patch_height, 0:patch_width + overlap].copy()
		patch_2 = image[0: patch_height, patch_width - overlap: width].copy()
		patch_3 = image[patch_height: height, 0:patch_width + overlap].copy()
		patch_4 = image[patch_height: height, patch_width - overlap: width].copy()
		patch_5 = image[int(patch_height / 2): height - int(patch_height / 2), int(patch_width / 2) - int(overlap / 2): width - int(patch_width / 2) + int(overlap / 2)].copy()
		all_patches.append(patch_1)
		all_patches.append(patch_2)
		all_patches.append(patch_3)
		all_patches.append(patch_4)
		all_patches.append(patch_5)
		all_patches.append(cv2.flip(patch_1, flipCode=1))
		all_patches.append(cv2.flip(patch_2, flipCode=1))
		all_patches.append(cv2.flip(patch_3, flipCode=1))
		all_patches.append(cv2.flip(patch_4, flipCode=1))
		all_patches.append(cv2.flip(patch_5, flipCode=1))

		for j in range(0, all_patches.__len__()):
			new_file_name = "{0}_{1}.jpg".format(file_name, (j + 1))
			cv2.imwrite(patched_frames_folder + new_file_name, all_patches[j])
		if i % (len(files)/2) == 0:
			print("Thread #%d: %.2f Percent Completed So Far! Processed Number: %d" % (thread_id, (i / len(files)) * 100, i))
	elapsed_time = time.time() - start_time
	print("Thread #" + str(thread_id) + " Total Times Elapsed: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

if __name__ == '__main__':
	frame_directory = 'Frames'
	number_of_threads = 10
	number_of_files = len(glob.glob(os.path.join(frame_directory, '*.jpg')))
	slice_size = int(number_of_files / number_of_threads) + number_of_threads
	index = 0
	for i in range(1, 11):
		t = Thread(target=patching_frames, args=(frame_directory, index, index + slice_size, i,))
		index += slice_size
		t.start()

