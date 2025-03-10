# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:11:57 2025

@author: Andres Cremades Botella andrescb@kth.se

File to serialize the data of the model
"""
#%%
import numpy as np
import tensorflow as tf


def load_dataset(data_in : dict[list,dict[list,list,int]] = 
                           {"tfrecord_files": [],
                            "data_folder" : {"xclip"   : [],
                                             "zclip"   : [],
                                             "padding" : 0
                                             }
                            }) -> dict:
    """
    Function to read the TFRecords

    Parameters
    ----------
    data_in : dict[list,dict[list,list,int]], optional
        The default is {"tfrecord_files": [],
                        "data_folder" : {"xclip"   : [],
                                         "zclip"   : [],
                                         "padding" : 0
                                         }
                        }.
        Data:
            - tfrecord_files : list with all the TFRecords to read
            - data_folder    : dictionary containing the data of the folders:
                + xclip   : positions to clip the domain in x
                + zclip   : positions to clip the domain in z
                + padding : value of the padding (reduction of domain in the output)

    Returns
    -------
    dict
        Dictionary the TFRecords information:
            - parsed_data : information of the TFRecords

    """
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Check the inputs
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    for key_name,value in data_in.items():
        if key_name not in {"tfrecord_files","data_folder"}:
            raise TypeError(f"{key_name} is not expected")
            
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_folder
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "tfrecord_files" in data_in:
        tfrecord_files = list(data_in["tfrecord_files"])
    else:
        raise TypeError("key missing: tfrecord_files")
        
    if "data_folder" in data_in:
        data_folder = dict(data_in["data_folder"])      
    else:
        raise TypeError("key missing: data_folder")  
        
    if "xclip" in data_folder:
        xclip = list(data_folder["xclip"])            
    else:
        raise TypeError("key missing in data_folder: xclip")  
          
    if "zclip" in data_folder:
        zclip = list(data_folder["zclip"])
    else:
        raise TypeError("key missing in data_folder: zclip") 
           
    if "padding" in data_folder:
        padding = int(data_folder["padding"])
    else:
        raise TypeError("key missing in data_folder: padding")

    
    def parse_function(proto):
        """
        Function for parsing the data.

        """
        feature_description = {'feature': tf.io.FixedLenFeature([size_x*size_z*1],"float32"),
                               'label': tf.io.FixedLenFeature([(size_x-2*padding)*(size_z-2*padding)*1],"float32")}
        parsed_features     = tf.io.parse_single_example(proto,feature_description)
        feature             = tf.reshape(parsed_features['feature'],[size_x,size_z,1])
        label               = tf.reshape(parsed_features['label'],[(size_x-2*padding),(size_z-2*padding),1])
        return feature,label
    
    # -------------------------------------------------------------------------------------------------------------------
    # This function reads multiple TFRecord files and returns a parsed dataset.
    # -------------------------------------------------------------------------------------------------------------------
    size_x   = xclip[1]-xclip[0]
    size_z   = zclip[1]-zclip[0]
    
    files    = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset  = files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.experimental.AUTOTUNE)
    data_out = {"parsed_data" : dataset.map(parse_function)}
    return  data_out
