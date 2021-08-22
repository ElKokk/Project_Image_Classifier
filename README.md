# Developing an Image Classifier with Deep Learning

In this first part of the project, we'll work through a Jupyter notebook to implement an image classifier with PyTorch.


# Building the command line application

After we've built and trained a deep neural network on the flower data set, we'll convert it into an application that others can use. The application is a pair of Python scripts that run from the command line. 


The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image.