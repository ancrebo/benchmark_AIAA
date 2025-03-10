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
folder_file       = "folders_structure_2025_02_19.json"

data_folder       = read_json(data_in={"folder" : config_folder,
                                       "file"   : folder_file})

field_ini         = data_folder["field_ini_train"]
field_fin         = data_folder["field_fin_train"]
file_norm         = data_folder["file_norm"]
statistics_folder = data_folder["statistics_folder"]

print("Preparing data",flush=True)

for index in range(field_ini,field_fin):
    print(f"Reading file {index} to calculate the normalization.",flush=True)
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read grid and velocity
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_io = read_input_output(data_in={"data_folder" : data_folder, "index" : index})
    u_y15   = data_io["input"]
    dudy    = data_io["output"]
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if index == field_ini:
        u_y15_min = np.min(u_y15)
        u_y15_max = np.max(u_y15)
        dudy_min  = np.min(dudy)
        dudy_max  = np.max(dudy)
    else:
        u_y15_min = np.min([u_y15_min,np.min(u_y15)])
        u_y15_max = np.max([u_y15_max,np.max(u_y15)])
        dudy_min  = np.min([dudy_min,np.min(dudy)])
        dudy_max  = np.max([dudy_max,np.max(dudy)])
        
ff_norm   = h5py.File(statistics_folder + '/' + file_norm,"w")
ff_norm.create_dataset("u_y15_min",data=u_y15_min)
ff_norm.create_dataset("u_y15_max",data=u_y15_max)
ff_norm.create_dataset("dudy_min",data=dudy_min)
ff_norm.create_dataset("dudy_max",data=dudy_max)
ff_norm.close()

xclip     = data_folder["xclip"]
zclip     = data_folder["zclip"]
padding   = data_folder["padding"]
elem_spec = (tf.TensorSpec(shape=(xclip[1]-xclip[0],zclip[1]-zclip[0],1),dtype="float32"),
             tf.TensorSpec(shape=(xclip[1]-xclip[0],zclip[1]-zclip[0],1),dtype="float32"))
os.makedirs(data_folder["tfrecords_folder"],exist_ok=True)
for index in range(field_ini,field_fin):
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # TFRecords
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_io         = read_input_output(data_in={"data_folder" : data_folder, "index" : index})
    u_y15           = data_io["input"]
    dudy            = data_io["output"]  
    dudy_norm       = (dudy-dudy_min)/(dudy_max-dudy_min)
    u_y15_norm      = (u_y15-u_y15_min)/(u_y15_max-u_y15_min)
    data_X          = np.zeros((1,xclip[1]-xclip[0],zclip[1]-zclip[0],1))
    data_Y          = np.zeros((1,xclip[1]-xclip[0]-2*padding,zclip[1]-zclip[0]-2*padding,1))   
    data_X[0,:,:,0] = dudy_norm
    data_Y[0,:,:,0] = u_y15_norm   
    data_XY         = tf.data.Dataset.from_tensor_slices((data_X,data_Y))  
    output_path     = data_folder["tfrecords_folder"]+f'/dataset_{index}.tfrecord'
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for features, labels in tqdm(data_XY, desc=f"Writing {output_path}"):
            data_serialize = {"feature":features.numpy(),"label":labels.numpy()}
            example        = serialize_example(data_in=data_serialize)["example"]
            writer.write(example)

    