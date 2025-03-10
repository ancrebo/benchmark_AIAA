# benchmark_AIAA

This repository contains the files for the prediction of the streamwise velocity from the measurement of the shear stress on the wall. The repository contains three main files:
1. read_database.py  : file for calculating the normalization and prepare the tfrecords required for the training of the model.
2. train_database.py : file for training the model using the tfrecords. This file also postprocess the results.
3. check_database.py : file for obtaining the results given the trained model.

For the use of the code follow the next steps:
1. Configure the problem by running: /main_code/configurations/configure_problem.py
2. Preprocess the data by running: /main_code/read_database.py
3. Train the model and obtain the results using: /main_code/train_database.py
4. If required plot the results again by running: /main_code/check_database.py

The code has been tested using TensorFlow 2.10.

The following results are obtained. 
