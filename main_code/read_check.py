# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:06:38 2025

@author: Andres Cremades Botella andrescb@kth.se

File for reading the database BLdns1_3D_t00001.h5 from https://deepblue.lib.umich.edu/data/concern/data_sets/nc580m959?locale=en
This file reads the data in the database, imports the velocity fields, generates the input and the output of the model:
    - input  : shear stress in the wall
    - output : velocity at y+=15
and generates the TFRecords.
"""

#%%
from py_bin.py_functions.read_json import read_json
import numpy as np
from py_bin.py_functions.read_input_output import read_input_output
from py_bin.py_functions.serialize_example import serialize_example
import tensorflow as tf
from tqdm import tqdm
import os
import h5py

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the parameters and folders structure
# -------------------------------------------------------------------------------------------------------------------------------------------------

config_folder     = "../../configurations"
folder_file       = "folders_structure_2025_08_06.json"

data_folder       = read_json(data_in={"folder" : config_folder,
                                       "file"   : folder_file})

field_ini         = data_folder["field_ini_train"]
field_fin         = data_folder["field_fin_train"]
file_norm         = data_folder["file_norm"]
statistics_folder = data_folder["statistics_folder"]
folder_input      = data_folder["folder_input"]
folder_output     = data_folder["folder_output"]
zfill_database    = data_folder["zfill_database"]

ind_not           = []
print("Preparing data",flush=True)

for index in range(field_ini,field_fin):
    print(f"Reading file {index} to calculate the normalization.",flush=True)
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read grid and velocity
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    try:
        data_io = read_input_output(data_in={"data_folder" : data_folder, "index" : index})
    except:
        ind_not.append(index)
        continue
    u_y15   = data_io["input"]
    dudy    = data_io["output"]
    

file_not  = h5py.File(statistics_folder + '/' + 'file_not.h5',"w")
file_not.create_dataset("ind_not",data=ind_not)
file_not.close()

