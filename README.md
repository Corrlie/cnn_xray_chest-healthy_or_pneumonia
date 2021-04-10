Project of convolutional neural network for image classification of xray chest images of healthy and pneumonia lungs.


Packages used:
tensorflow, keras, cv2 (opencv), numpy, os, random


Created model predicts whether lungs are healthy or infected with pneumonia. 


Validation data accuracy: 0.95;

Test data accuracy: 0.91;

Total number of params: 7,221,251.


Project is based on "Chest X-Ray Images (Pneumonia)" dataset from kaggle, link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
The dataset contains 3 ready directories of train, val, and test data. While making this project several hundred of images were moved from train to val directory to increase the number of validation data. 


The project contains 'data_augmentation.py' file which is used to additionaly augment data by flipping and randomly changing the brightness, but also to set one dimension (width and height) to all of the images and save them (thus it also overwrites existing, resized raw images from kaggle). It is neccessary to point the path of one's dataset directories path. By running this file one can get 4 times more available images.

In the 'cnn_xray_chest.py' the model of convolutional neural network is created. It also saves trained model in json and h5 format.

The 'test_prediction.py' file is used to check the test accuracy of trained model, loaded from saved json and h5 files.