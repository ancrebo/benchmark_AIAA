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

![Image](https://github.com/user-attachments/assets/0fe9b446-441d-45b9-9111-e49eb7fb4b66)

This training corresponds to a mean error of 2.93% of the maximum value of the velocity at $y^+\approx 15$. The input field, $\frac{\partial u}{\partial y}$, is the following:

![Image](https://github.com/user-attachments/assets/d0e01458-6306-4650-bfbe-38b4e10102c7)

The output (ground truth): 

![Image](https://github.com/user-attachments/assets/21d4a8ca-98e7-4b7c-b4a2-b0e2fe7de5ac)

And the predicted field:

![Image](https://github.com/user-attachments/assets/59929f99-9e2f-47a9-9157-3effa51ae152)
