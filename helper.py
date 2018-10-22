####################################################################
##
##  Authors:		Peter Zorzonello
##  Last Update:	10/20/2018
##  Class:			EC601 - A1
##  File_Name:		Helper.py
##
##  Description:	
##    This the file that defines the API. 
##
####################################################################

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

#imports a tool to plot/display figures and images
import matplotlib.pyplot as plt
import numpy as np

## getTrainData
##
## DESCRIPTION:
##		Gets the traning data using the Keras API. 
##      Tranning data comes from images on the local computer
##
## INPUTS:
## 		path:       String   A path to the data dir(no tailing '/')
## 		image_size: Integer  Value for hight and width to resize image to
## 		color:      String   Color mode for KERAS API. Must be 'rgb' or 'grayscale'
## 		batch:      Integer  Size of the batches of data to use
## 		mode:       String	 Determines the type of label arrays 
##
## OUTPUT:
## 		train_generator: the traning data for the modle
##
def getTrainData(path, image_size, color, batch, mode):

	try:
		# data generator for training set needed for keras to read the images
		train_datagen = ImageDataGenerator(
			rescale = 1./255,
			shear_range = 0.2, 
			zoom_range = 0.2,
			horizontal_flip = True) 

		# generator for reading train data from folder
		train_generator = train_datagen.flow_from_directory(
			path+'/train',
			target_size = (image_size, image_size),
			color_mode = color,
			batch_size = batch,
			class_mode = mode)

		return train_generator
	except Exception as e:
		print("getTrainData failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getTestData
##
## DESCRIPTION:
##		Gets the test data using the Keras API. 
##      Test data comes from images on the local computer
##
## INPUTS:
## 		path:       String   A path to the data dir(no tailing '/')
## 		image_size: Integer  Value for hight and width to resize image to
## 		color:      String   Color mode for KERAS API. Must be 'rgb' or 'grayscale'
## 		batch:      Integer  Size of the batches of data to use
## 		mode:       String	 Determines the type of label arrays 
##
## OUTPUT:
## 		validation_generator: the traning data for the modle
##
def getTestData(path, image_size, color, batch, mode):
	try:
		test_datagen = ImageDataGenerator(rescale = 1./255)

		validation_generator = test_datagen.flow_from_directory(
			path+'/valid',
			target_size = (image_size, image_size),
			color_mode = color,
			batch_size = batch,
			class_mode = mode)

		test_files_names = validation_generator.filenames

		return validation_generator, test_files_names
	except Exception as e:
		print("getTestData failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()


def getTestGen(path, image_size, color, mode):
	try:
		test_datagen = ImageDataGenerator(rescale = 1./255)

		# generator for reading test data from folder
		test_generator = test_datagen.flow_from_directory(
			path+'/test',
			target_size = (image_size, image_size),
			color_mode = color,
			batch_size = 1,
			class_mode = mode,
			shuffle = False)

		test_files_names = test_generator.filenames

		return test_generator, test_files_names
	except Exception as e:
		print("getTestGen failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getModelOne
##
## DESCRIPTION:
##		Create a model of a neural network.
##
## INPUTS:
## 		image_size:	Integer. The size of the image.
##
## OUTPUT:
## 		model: The model to use for the nureal network
##
def getModelOne(image_size):
	try:
		model = Sequential()
		#this line is needed, For some reason the way the data is formated this must be this way
		model.add(Conv2D(32, (3,3), input_shape = (image_size, image_size, 3), activation = 'relu'))
		model.add(Flatten())

		#must be last or else does not work
		model.add(Dense(1, activation = 'sigmoid'))
		
		return model
	except Exception as e:
		print("getModelOne failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getModelTwo
##
## DESCRIPTION:
##		Create a model of a neural network.
##
## INPUTS:
## 		image_size:	Integer. The size of the image.
##
## OUTPUT:
## 		model: The model to use for the nureal network
##
def getModelTwo(image_size):
	try:
		model = Sequential()
		#this line is needed, For some reason the way the data is formated this must be this way
		model.add(Conv2D(32, (3,3), input_shape = (image_size, image_size, 3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2,2)))	

		model.add(Conv2D(32, (3,3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2,2)))	
		model.add(Dropout(0.2))

		model.add(Conv2D(64, (3,3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (2,2)))	

		#this line is needed, without it the model does not work
		model.add(Flatten())
		model.add(Dropout(0.2))
		model.add(Dense(2, activation = 'softmax'))
		model.add(Dense(128, activation = 'relu'))
		model.add(Dropout(0.2))
		#must be last or else does not work
		model.add(Dense(1, activation = 'sigmoid'))

		return model
	except Exception as e:
		print("getModelTwo failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getModelThree
##
## DESCRIPTION:
##		Create a model of a neural network.
##
## INPUTS:
## 		image_size:	Integer. The size of the image.
##
## OUTPUT:
## 		model: The model to use for the nureal network
##
def getModelThree(image_size):
	try:
		model = Sequential()
		#this line is needed, For some reason the way the data is formated this must be this way
		model.add(Conv2D(32, (3,3), input_shape = (image_size, image_size, 3), activation = 'relu'))
		model.add(MaxPooling2D(pool_size = (4,4)))
		
		model.add(Dense(36, activation = 'relu'))
		model.add(Dropout(0.2))
		model.add(Dense(300))
		#this line is needed, without it the model does not work
		model.add(Flatten())
		model.add(Dense(128, activation = 'relu'))
		model.add(Dropout(0.5))
		#must be last or else does not work
		model.add(Dense(1, activation = 'sigmoid'))
		return model
	except Exception as e:
		print("getModelThree failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## compile
##
## DESCRIPTION:
##		Compiles the model so it can be used.
##
## INPUTS:
## 		model:		The model to compile
##		optimizer:	The optmizer to use for compiling
##
## OUTPUT:
## 		model: The model to use for the nureal network
##
def compile(model, optimizer):
	try:
		model.compile(loss = 'binary_crossentropy',
					optimizer = optimizer,
					metrics = ['accuracy'])
		return model
	except Exception as e:
		print("compile failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## fit
##
## DESCRIPTION:
##		Trains the model with traning and testing data 
##
## INPUTS:
## 		model:		The model to compile
##		train_images:	The traning data images
##		train_labels:	The traning data labels
##		epoch_num:	The number of epochs to run the training for
##
## OUTPUT:
## 		model: The model after it has been trained
##
def fit(model, train_images, train_labels, epoch_num):
	try:
		model.fit(train_images, train_labels, epochs=epoch_num, batch_size=1)

		return model
	except Exception as e:
		print("fit failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getTestSet
##
## DESCRIPTION:
##		Gets a set from the test data
##
## INPUTS:
## 		dataGen: The test data
##
## OUTPUT:
## 		test_samples:	The images in the test data
##		test_labels: 	The labels for the test data
##
def getTestSet(dataGen):
	try:
		print("getting next batch of data....")
		test_samples, test_labels = next(dataGen)
		return test_samples, test_labels
	except Exception as e:
		print("getTestSet failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## evalModel
##
## DESCRIPTION:
##		Evaluate the model to see how well it performs.
##
## INPUTS:
## 		model:			The model to evaluate
##		test_samples:	The test images to evaluate the model with
##		test_labels:	The test labels to evaluate the model
##
## OUTPUT:
## 		loss:		The loss of the model
##		accuracy: 	The accuracy of the model
##
def evalModel(model, test_samples, test_labels):
	try:
		loss, accuracy = model.evaluate(test_samples, test_labels)

		return loss, accuracy
	except Exception as e:
		print("evalModel failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## debugTestImage
##
## DESCRIPTION:
##		Usefule for debugging. Prints an image to the screen.
##
## INPUTS:
## 		index:			The index in the array of labels/images to print
##		test_images:	The test images
##		test_labels:	The test labels
##
## OUTPUT:
## 		loss:		The loss of the model
##		accuracy: 	The accuracy of the model
##
def debugTestImage(index, test_images, test_labels):
	try:
		print(test_labels[index])
		plt.figure()
		plt.imshow(test_images[index])
		plt.grid(False)
		plt.xlabel(test_labels[index])
		plt.show(test_labels[index])
	except Exception as e:
		print("debugTestImage failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getModelNumFromUser
##
## DESCRIPTION:
##		Ask the user what model to use.
##
## INPUTS:
##		None
##
## OUTPUT:
## 		model_num:		The model to use
##
def getModelNumFromUser():
	try:
		while True:
			model_num = input("What model do you want to use? (1, 2, 3, or exit)\n")

			if (model_num == '1' or model_num == '2' or model_num == '3'):
				return model_num
			elif (model_num == 'exit'):
				exit()
	except Exception as e:
		print("getModelNumFromUser failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()


## getEpoch
##
## DESCRIPTION:
##		Ask the user how many epochs to use.
##
## INPUTS:
##		None
##
## OUTPUT:
## 		epochs:		The number of epochs to use
##
def getEpoch():
	try:
		while True:
			epochs_num = input("How many epochs? (keep in mind more epochs = longer time but greater accuracy)\nEnter exit to quit.\n")

			if epochs_num != 'exit':
				try:
					epochs = int(epochs_num)
					return epochs

				except:
					print("Not a valid input. Must be a number.")
			else:
				exit()
	except Exception as e:
		print("getEpoch failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

## getOptimizer
##
## DESCRIPTION:
##		Ask the user how which optmizer to use
##
## INPUTS:
##		None
##
## OUTPUT:
## 		opt:		The optimizer the user picked
##
def getOptimizer():
	try:
		while True:
			opt = input("Choose one of these optmizers:\n1) RMSProp\n2) Adam\n3) Adagrad\nEnter exit to quit\n")

			if opt == 'exit':
				exit()
			elif opt == '1':
				return 'rmsprop'
			elif opt == '2':
				return 'adam'
			elif opt == '3':
				return 'adagrad'
	except Exception as e:
		print("getOptimizer failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()


## printSummary
##
## DESCRIPTION:
##		Prints data about what test was run
##
## INPUTS:
##		None
##
## OUTPUT:
## 		None
##
def printSummary(model_num, opt, num_epoch):
	try:
		print('Model Number chose: ', model_num)
		print('Optmizer chosen: ', opt)
		print('Number of epochs: ', num_epoch)
	except Exception as e:
		print("printSummary failed with exception: ", e)
		print("Terminating now, Sorry.")
		exit()

