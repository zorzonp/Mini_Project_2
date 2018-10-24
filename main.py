####################################################################
##
##  Authors:		Peter Zorzonello
##  Last Update:	10/20/2018
##  Class:			EC601 - A1
##  File_Name:		Main.py
##
##  Description:	
##    This is a test file to test all of the API calls in helper.
##    This file will show how the model performed.
##
####################################################################


#import my API
import helper

#Import other libraries
import numpy as np
import pandas as pd
import os, os.path
import matplotlib.pyplot as plt

#global Variables 
batch_size = 0
num_files_train = 0

batch_size = 9000


#the data directory is where all the images are
path = 'data'

#this can be any size but the bigger it is the slower it runs
image_size = 28

color_mode = 'rgb'
mode = 'binary'


classes = ['dog', 'cat']

#get batch size from number of files in directory
for name in os.listdir(path+'/train/class_a/'):
		num_files_train = num_files_train + 1
#get batch size from number of files in directory
for name in os.listdir(path+'/train/class_b/'):
		num_files_train = num_files_train + 1

num_batch = num_files_train/batch_size


print("num_files_train: ", num_files_train)
print("num_batchs: ", num_batch)


#get the test and train data from the images in 'data/'
train_data = helper.getTrainData(path, image_size, color_mode, batch_size, mode)
test_data, test_files_names = helper.getTestData(path, image_size, color_mode, batch_size, mode)

#get the user to choose which modle number to use
model_num = helper.getModelNumFromUser()

#Use the chosen model
if model_num == '3':
	model = helper.getModelThree(image_size)
elif model_num == '2':
	model = helper.getModelTwo(image_size)
else:
	model = helper.getModelOne(image_size)

opt = helper.getOptimizer()
#get the number of epochs the model should use
num_epoch = helper.getEpoch()

i = 0
print("Fit Model")
while i < num_batch:
	train_set_images, train_set_labels = helper.getTestSet(train_data)

	model = helper.compile(model, optimizer = opt)

	#fit the data in the modle
	model = helper.fit(model, train_set_images, train_set_labels, num_epoch)
	i = i+1

print("Evaluate Model")
test_set_images, test_set_labels = helper.getTestSet(test_data)


#evaluate the model
loss, accuracy = helper.evalModel(model, test_set_images, test_set_labels)

#print data on how well the model did
print('Test accuracy: ', accuracy)
print('Test loss: ', loss)
helper.printSummary(model_num, opt, num_epoch)

#Do a prediction
print("Predict")
predictions = model.predict_classes(test_set_images)
print(predictions)
print(test_set_labels)

try:
	#get the first occurance of a CAT
	lbl_index = 0
	for label in test_set_labels:
		if label == 1:
			break
		lbl_index = lbl_index + 1

	#print the predicted class as text
	tmp_index = int(predictions[lbl_index])
	print("Class predict: ", classes[tmp_index])

	#if the class predicted matched the label then print pass, else print fail
	if(predictions[lbl_index] == test_set_labels[lbl_index]):
		print("PASS!")
	else:
		print("FAIL comparison!!")
except Exception as e:
	print("There was no cat picture in the testing set.")

try:
	#get the first occurance of a DOG
	lbl_index = 0
	for label in test_set_labels:
		if label == 0:
		break
		lbl_index = lbl_index + 1

	#print the predicted class as text
	tmp_index = int(predictions[lbl_index])
	print("Class predict:", classes[tmp_index])

	#if the class predicted matched the label then print pass, else print fail
	if(predictions[lbl_index] == test_set_labels[lbl_index]):
		print("PASS!")
	else:
		print("FAIL comparison!!")
except Exception as e:
	print("There was no dog picture in the testing set.")





