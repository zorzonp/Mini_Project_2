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

#global Variables 
batch_size = 20
num_val_samples = 137
steps = 2295*1.4/batch_size

#the data directory is where all the images are
path = 'data'

#this can be any size but the bigger it is the slower it runs
image_size = 28

color_mode = 'rgb'
mode = 'binary'
test_steps = num_val_samples/batch_size

#get the test and train data from the images in 'data/'
train_data = helper.getTrainData(path, image_size, color_mode, batch_size, mode)
test_data = helper.getTestData(path, image_size, color_mode, batch_size, mode)

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
model = helper.compile(model, optimizer = opt)


#get the number of epochs the model should use
num_epoch = helper.getEpoch()

#fit the data in the modle
model = helper.fit(model, train_data, test_data, num_epoch, test_steps, steps )

test_set_images, test_set_labels = helper.getTestSet(test_data)

#evaluate the model
loss, accuracy = helper.evalModel(model, test_set_images, test_set_labels)

#print data on how well the model did
print('Test accuracy: ', accuracy)
print('Test loss: ', loss)

helper.printSummary(model_num, opt, num_epoch)

# helper.debugTestImage(index = 0, 
# 					test_images = test_set_images, 
# 					test_labels = test_set_labels)
