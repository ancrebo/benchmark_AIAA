# benchmark_AIAA

This repository contains the files for the prediction of the streamwise velocity fluctuation, $u$, at a wall-normal distance of  $y^+\approx 15$, from the measurement of the shear stress on the wall, $\frac{\partial u}{\partial y}$. The repository contains three main files:
1. read_database.py  : file for calculating the normalization and prepare the tfrecords required for the training of the model.
2. train_database.py : file for training the model using the tfrecords. This file also postprocess the results.
3. check_database.py : file for obtaining the results given the trained model.

For the use of the code follow the next steps:
1. Configure the problem by running: /main_code/configurations/configure_problem.py
2. Preprocess the data by running: /main_code/read_database.py
3. Train the model and obtain the results using: /main_code/train_database.py
4. If required plot the results again by running: /main_code/check_database.py

The code has been tested using TensorFlow 2.10.

The model is trained as shown below:

<img width="640" height="480" alt="hist" src="https://github.com/user-attachments/assets/e99103cf-47bf-47ac-8c69-f3a8742ed967" />


This training corresponds to a mean error of 4.05% of the maximum value of the velocity at $y^+\approx 15$. The input field, $\frac{\partial u}{\partial y}$, is the following:

<img width="640" height="480" alt="input" src="https://github.com/user-attachments/assets/0296522a-d6d8-4bdb-8d49-276109961e46" />

The output (ground truth): 

<img width="640" height="480" alt="output" src="https://github.com/user-attachments/assets/625a61e3-3371-4134-88fc-0418ba6feaee" />

And the predicted field:

<img width="640" height="480" alt="pred" src="https://github.com/user-attachments/assets/d1bd2310-8af8-48b5-b11b-542e82fc78e2" />


