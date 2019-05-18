# organize imports
import os
import glob
import datetime

# print start time
print("[INFO] program started on - " + str(datetime.datetime.now))

# get the input and output path
input_path  = "/home/yhhan/git/deeplink_public/1.DeepLearning/07.CNN/transfer_learning/dataset/jpg"
output_path = "/home/yhhan/git/deeplink_public/1.DeepLearning/07.CNN/transfer_learning/dataset/train"

# get the class label limit
class_limit = 17

# take all the images from the dataset
# /home/yhhan/git/deeplink_public/1.DeepLearning/07.CNN/transfer_learning/dataset/jpg/image_0499.jpg

image_paths = glob.glob(input_path + "/*.jpg")

image_paths = sorted(image_paths)

print(image_paths)

