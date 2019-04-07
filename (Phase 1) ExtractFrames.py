import cv2
import os
import glob
import numpy as np
from numpy import loadtxt


def extract_frames(files_grabbed):
	file_index = 1

	for location in files_grabbed:
		# Capture Video File
		cap = cv2.VideoCapture(location)

		# Get Total Frames and FPS From Video File
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		frames = []

		# Generate Frame Numbers That Need To Be Extracted
		first_frame = 1
		second_frame = int(fps/2)

		while second_frame <= total_frames:
			frames.append(first_frame)
			frames.append(second_frame)
			first_frame += fps
			second_frame += fps

		if first_frame <= total_frames:
			frames.append(first_frame)

		# Create Necessary Folders
		frames_folder = 'Frames'
		if not os.path.exists(frames_folder):
			os.makedirs(frames_folder)

		# Save Frames
		for i in range(0, len(frames)):
			cap.set(1, frames[i])
			ret, frame = cap.retrieve()
			file_name = "{0}_{1}.jpg".format(file_index, frames[i])
			cv2.imwrite("Frames/" + file_name, frame)
			if i % 50 == 0 and i != 0:
				print(str(i) + 'th Frame of file \"' + location + '\" Write Successful!')
		cap.release()
		cv2.destroyAllWindows()
		file_index += 1

if __name__ == '__main__':
	# Define Video Directory
	videos_directory = 'Videos'
	# Get All Files That Have Specific File Types
	types = ('*.flv', '*.mp4')
	files_grabbed = []
	# Get file names from provided video folder
	for files in types:
		files_grabbed.extend(glob.glob(os.path.join(videos_directory, files)))
	# extract frames from files
	extract_frames(files_grabbed)
