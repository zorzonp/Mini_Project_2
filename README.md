# Mini Project 2
## About the Project 
This project is a neural network assignment. Examins different models to determin how they perform. 

To build the project I utalised Keras API and Matplotlib.

In order to use the Keras API to get the images into a format that can be evaluated by the modle they must be in a data structure that looks like:
data/
	test/
	valid/
		class_a/
		class_b/
	train/
		class_a/
		class_b/

The images I am using to train and test my modle are the Dog Vs. Cat dataset from Kaggle:

I utlized demos to read in images and to aid in building the modles. See the Sources section for the demo's I used.

For this Project I created 3 models that the user can choose from. If ruuning using the API just call the function for the model you want. 
If runninng using the supplied main.py file it will query the user to see what modle to use. It will also ask what optmizer you want to use and how many epochs to use. Using around 10 epochs is best but takes a really long time. Also 


## Running
You need to install keras and matplotlib prior to running.
I had issues running/installing these. I needed to install them mutiple times, with different version of Python. I also had to install Anaconda and Miniconda.

I have python 2.7, 3, 3.5, and 3.7 on my Mac. I had to install theses libraries mutiple times. I also needed to reinstall Python.
In the end I needed to run using Miniconda3.


## Sources
DEMO: https://www.tensorflow.org/tutorials/keras/basic_classification
DEMO: https://www.kaggle.com/ievgenvp/keras-flow-from-directory-on-python/notebook
KERAS API: https://keras.io