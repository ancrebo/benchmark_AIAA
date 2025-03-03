# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:11:57 2025

@author: Andres Cremades Botella andrescb@kth.se

File to serialize the data of the model
"""
#%%
import numpy as np
import tensorflow as tf

def serialize_example(data_in : dict[np.ndarray,np.ndarray] = 
                      {"feature" : np.array([0]),
                       "label"   : np.array([0])
                       }) -> dict:
    """
    Function to serialize the input and output of the model

    Parameters
    ----------
    data_in : dict[np.ndarray,np.ndarray], optional
        The default is {"feature" : np.array([0]),
                        "label"   : np.array([0])
                        }.
        Data:
            - feature : input data
            - label   : output data

    Returns
    -------
    dict
        Dictionary storing the velocity:
            - example : serialized data

    """
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Check the inputs
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    for key_name,value in data_in.items():
        if key_name not in {"feature","label"}:
            raise TypeError(f"{key_name} is not expected")
            
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read feature
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "feature" in data_in:
        feature = data_in["feature"]
        if not isinstance(feature, np.ndarray):
            raise TypeError(f"feature must be a numpy array, got {type(feature).__name__}")
    else:
        raise TypeError("key missing: feature")
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read label
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "label" in data_in:
        label = data_in["label"]
        if not isinstance(label, np.ndarray):
            raise TypeError(f"feature must be a numpy array, got {type(label).__name__}")
    else:
        raise TypeError("key missing: label")
        
    # -------------------------------------------------------------------------------------------------------------------
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    # -------------------------------------------------------------------------------------------------------------------
    feature = {'feature': tf.train.Feature(float_list=tf.train.FloatList(value=feature.flatten())),
               'label': tf.train.Feature(float_list=tf.train.FloatList(value=label.flatten()))}

    # -------------------------------------------------------------------------------------------------------------------
    # Create a Features message using tf.train.Example.
    # -------------------------------------------------------------------------------------------------------------------
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    data_out      = {"example":example_proto.SerializeToString()}
    return data_out