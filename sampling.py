import glob
import random
import os, shutil


def main():
	file_paths = glob.glob('../darknet/result/*.jpg')
	random.shuffle(file_paths)

	for img in file_paths[:50]:
		name = os.path.basename(img)
		out_path = os.path.join('../sampling_data', name)
		shutil.copyfile(img, out_path)


if __name__ == '__main__':
	main()