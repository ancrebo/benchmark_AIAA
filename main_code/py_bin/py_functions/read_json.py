# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:12:07 2025

@author: Andres Cremades Botella andrescb@kth.se

File to read json files
"""

#%%
from typing import Dict, List, Union

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions
# -------------------------------------------------------------------------------------------------------------------------------------------------
def read_json(data_in : dict[str,str,Union[List[str],None]] = 
              {"folder" : "-",
               "file"   : "-",
               "keys"   : None
               }) -> dict:
    """
    Function for reading json files. It returns the whole file ("keys":None) or some of the keys ("keys":["key1","key2",...]).

    Parameters
    ----------
    data_in : dict[str,str,Union[list[str],None]], optional
        The default is {"folder" : "-",
                        "file"   : "-",
                        "keys"   : None
                        }.
        Data:
            - folder : folder to get the file
            - file   : file to read
            - keys   : keys to read from the json, if None read all

    Returns
    -------
    dict
        Data that has been read from the json file.

    """
    import json
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Check the inputs
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    for key_name,value in data_in.items():
        if key_name not in {"folder","file","keys"}:
            raise TypeError(f"{key_name} is not expected")
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read folder
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "folder" in data_in:
        folder = str(data_in["folder"])
    else:
        raise TypeError("key missing: folder")
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read file
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "file" in data_in:
        file = str(data_in["file"])
    else:
        raise TypeError("key missing: file")
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read keys
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "keys" in data_in:
        keys = list(data_in["keys"])
    else:
        keys = None
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read json file
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    with open(folder+'/'+file, "r") as ff:
        data = json.load(ff)
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Select keys from file
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if keys is not None:
        data2 = {}
        for key_name in keys:
            if key_name in data:
                data2[key_name] = data[key_name]
            else:
                raise TypeError(f"{key_name} is not a proper key for {folder+'/'+file}")
        return data2
    else:
        return data
            