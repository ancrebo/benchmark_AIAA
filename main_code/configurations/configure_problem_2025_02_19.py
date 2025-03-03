# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:11:53 2025

@author: Andres Cremades Botella andrescb@kth.se

File to store the data for the problem. 

Parameters:
    - config_folder : path to the folder to store the configurations of the problem
    - folder_file   : file to store the configuration of the folder
    - folder_data   : data to store in the folder_file

- 1st file: folder structure of the system. 
            Contains:
                - folder_grid          : folder in which the original grid is stored
                - file_grid            : file in which the original grid is stored
                - folder_database      : folder in which the original data is stored
                - file_database        : file in of each instantaneous snapshot of the flow use $INDEX$ for substituting by the index of the file
                - zfill_database       : number of digits of the index
                - folder_processeddata : folder to store the data extracted for training the model (in this benchmark the gradient dudy on the
                                                                                                    wall and the velocity u at a wall-normal 
                                                                                                    distance y+=15)
                - file_processeddata   : file to store the data extracted from a single gridpoint
                - statistics_folder    : folder in which the statistics are stored
                - mean_file            : file to store the mean magnitudes of the flow
                - unorm_file           : file to store the normalization values of the velocity
                - xclip                : positions to clip the domain in x
                - zclip                : positions to clip the domain in z
                - tfrecords_folder     : folder for the tfrecords
                - padding              : value of the padding (reduction of domain in the output)
                - field_ini_train      : index of the initial field for the training
                - field_fin_train      : index of the final field for the training
                - ngpu                 : number of gpus (-1 take all available)
                - test_ratio           : ration between the test sample and the database 
                - batch_size           : batch size for the training
                - prefetch             : value for the prefetch of the training (-1 is default)
                - learat               : learning ratio of the training
                - nfil                 : reference number of filters of the convolutional neural network
                - kernel               : size of the kernel
                - stride               : value of the stride
                - activation           : activation function (relu, tanh, sigmoid)
                - epochs               : epochs to train
                - save_every           : save every x epochs
                - model_file           : file to save the model
"""

#%% 
import json

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the folder file
# -------------------------------------------------------------------------------------------------------------------------------------------------
config_folder = "../../configurations"
folder_file   = "folders_structure_2025_02_19.json"


# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the folder file
# -------------------------------------------------------------------------------------------------------------------------------------------------
folder_data = {
    "folder_database"      : "../BLdns1_3D_t00001-t01000",
    "file_database"        : "BLdns1_3D_t$INDEX$.h5",
    "folder_grid"          : "../grid",
    "file_grid"            : "BLdns1_grid.h5",
    "zfill_database"       : 5,
    "folder_processeddata" : "../BLdns1_dudy_u15",
    "file_processeddata"   : "BLdns1_dudy_u15_t$INDEX$.h5",
    "statistics_folder"    : "../statistics",
    "mean_file"            : "BLdns1_means.h5",
    "unorm_file"           : "unorm_file.h5",
    "xclip"                : [1500,2000],
    "zclip"                : [100,200],
    "tfrecords_folder"     : "../tfrecords",
    "padding"              : 5,
    "field_ini_train"      : 1,
    "field_fin_train"      : 5,
    "ngpu"                 : -1,
    "test_ratio"           : 0.2,
    "batch_size"           : 1,
    "prefetch"             : -1,
    "learat"               : 1e-4,
    "nfil"                 : 4,
    "kernel"               : 3,
    "stride"               : 1,
    "activation"           : "relu",
    "epochs"               : 2000,
    "save_every"           : 20,
    "model_file"           : "model.h5"
    }


# -------------------------------------------------------------------------------------------------------------------------------------------------
# Save the folder file
# -------------------------------------------------------------------------------------------------------------------------------------------------
with open(config_folder+'/'+folder_file, "w") as ff:
    json.dump(folder_data, ff, indent=4)