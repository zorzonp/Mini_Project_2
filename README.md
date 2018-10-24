# Mini Project 2
## About the Project 
This project is a neural network assignment. Examines different models to determine how they perform. 

To build the project I utilized Keras API and Matplotlib.

In order to use the Keras API to get the images into a format that can be evaluated by the model they must be in a data structure that looks like [3]:
data/

	test/
	
	valid/
	
		class_a/
		
		class_b/
		
	train/
	
		class_a/
		
		class_b/
		

The images I am using to train and test my model are the Dog Vs. Cat dataset from Kaggle [1]

I utilized demos to read in images and to aid in building the models. See the Sources section for the demo's I used [3][2].

## Project Features
For this Project I created 3 models that the user can choose from. If running using the API just call the function for the model you want. 
If running using the supplied main.py file it will query the user to see what model to use. It will also ask what optimizer you want to use and how many epochs to use. Using around 10 epochs is best but takes a really long time. Also, when using the main.py file the resize for the image is 28x28. If using the API, you can choose the size, however the bigger the image the slower it takes to train the model and the less the accuracy score is. 

It should be noted that I tried to create these models myself using the demos as reference. I cannot get them to have an accuracy above 70%. I do not believe the data set is the issue as since it was from Kaggle I am sure it is robust enough. Also, I believe there were enough sample as the training data consists of an equal number of dog and cat pictures (9,000 each). 

The way the data set is created places the images in random order each time it is run. As such in order to accurately test I look for the first occurrence of a cat and the first occurrence of a dog after passing a set of data into the predict function. I then use their index to get their prediction and actual label. I compare the two. If they match the prediction was correct, otherwise the prediction was incorrect. 


## Running
You need to install Keras and matplotlib prior to running.
I had issues running/installing these. I needed to install them multiple times, with different version of Python. I also had to install Anaconda and Miniconda.

### Problems

I have python 2.7, 3, 3.5, and 3.7 on my Mac. I had to install the libraries used in the project multiple times. SOme how they ended up uninstalling the Python framework from the operating system so I also needed to reinstall Python.
In the end I needed to run using Miniconda3.

I am not sure what you will have installed on your computer but you should test out these libraries before running. Start Python in the console/terminal by opening the terminal and typing python.

One you start python try importing the libraries. See helper.py for the libraries the API uses and main.py for the libraries the test script uses. If you can successfully import the libraries then you should have no problem running the files. If not try installing the libraries under a different version on Python. 


## Sources

[1]"Dogs vs. Cats Redux: Kernels Edition | Kaggle", Kaggle.com, 2018. [Online]. Available: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data. [Accessed: 24- Oct- 2018].

[4]"Keras Documentation", Keras.io, 2018. [Online]. Available: https://keras.io. [Accessed: 24- Oct- 2018].

[3]"Keras flow_from_directory on Python | Kaggle", Kaggle.com, 2018. [Online]. Available: https://www.kaggle.com/ievgenvp/keras-flow-from-directory-on-python/notebook. [Accessed: 24- Oct- 2018].

[2]"Train your first neural network: basic classification  |  TensorFlow", TensorFlow, 2018. [Online]. Available: https://www.tensorflow.org/tutorials/keras/basic_classification. [Accessed: 24- Oct- 2018].

