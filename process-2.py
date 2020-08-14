import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
task_name = os.path.join(current_dir, 'task_capsule/')

# Directory where the data will reside, relative to 'darknet.exe'
path_data = task_name + 'data/'

# Create and/or truncate train.txt and test.txt
file_paths = glob.glob(os.path.join(path_data, "*.JPG"))
file_paths.sort()
file_train = open(task_name + 'train.txt', 'w')  
file_valid = open(task_name + 'valid.txt', 'w')
file_test = open(task_name + 'test.txt', 'w')

print(file_paths)

# # Populate train.txt and test.txt
# for pathAndFilename in file_paths[658:2114]: 
# 	file_train.write(pathAndFilename + "\n")

# for pathAndFilename in file_paths[:658]: 
# 	file_valid.write(pathAndFilename + "\n")

# for pathAndFilename in file_paths[2114:]:  
#     file_test.write(pathAndFilename + "\n")