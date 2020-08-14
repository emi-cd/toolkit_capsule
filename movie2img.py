import cv2
import os, sys
import glob

def save_frames(video_path, dir_path, step, ext='JPG'):
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return

	basename = os.path.splitext(os.path.basename(video_path))[0]
	os.makedirs(dir_path, exist_ok=True)
	base_path = os.path.join(dir_path, basename)

	digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

	n = 0
	while True:
		cap.set(cv2.CAP_PROP_POS_FRAMES, n*step)
		ret, frame = cap.read()
		if ret:
			cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
			n += 1
		else:
			return


def main(argv):
	# python movie2img.py ../origin_capsule_mov ../origin_capsule_mov/splited_img 150
	for filepath in glob.glob(argv[0] + '/*.MP4'):
		print(filepath)
		save_frames(filepath, argv[1], int(argv[2]))


if __name__ == '__main__':
	main(sys.argv[1:])
