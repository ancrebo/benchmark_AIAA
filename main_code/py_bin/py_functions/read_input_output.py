# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:11:57 2025

@author: Andres Cremades Botella andrescb@kth.se

File to read the input and output of the model
"""
#%%
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions
# -------------------------------------------------------------------------------------------------------------------------------------------------
def read_input_output(data_in : dict[dict[list,list,int],int] = 
                      {"data_folder" : {"xclip"             : [],
                                        "zclip"             : [],
                                        "padding"           : 0
                                        },
                       "index"       : 0
                       }) -> dict:
    """
    Function to generate the input and output of the model

    Parameters
    ----------
    data_in : dict[dict[str,str,str,str,str,str,list,list,int],int], optional
        The default is {"data_folder" : {"xclip"             : [],
                                         "zclip"             : [],
                                         "padding"           : 0
                                         },
                        "index"       : 0
                        }.
        Data:
            - data_folder : dictionary containing the data of the folders:
                + xclip             : positions to clip the domain in x
                + zclip             : positions to clip the domain in z
                + padding           : value of the padding (reduction of domain in the output)
            - index       : index of the snapshot to read

    Returns
    -------
    dict
        Dictionary storing the velocity:
            - u : streamwise velocity
            - v : wall-normal velocity
            - w : spanwise velocity

    """
    import h5py
    import numpy as np
    from py_bin.py_functions.read_grid_database import read_grid_database
    from py_bin.py_functions.read_velocity_database import read_velocity_database
    from py_bin.py_functions.read_mean_database import read_mean_database
    from py_bin.py_functions.calc_velfluc import calc_velfluc
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Check the inputs
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    for key_name,value in data_in.items():
        if key_name not in {"data_folder","index"}:
            raise TypeError(f"{key_name} is not expected")
            
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_folder
    # ---------------------------------------------------------------------------------------------------------------------------------------------
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
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read index
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "index" in data_in:
        index = int(data_in["index"])
    else:
        raise TypeError("key missing: index")
        
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Calculate fluctuations
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_grid           = read_grid_database(data_in={"data_folder" : data_folder})
    data_velocity       = read_velocity_database(data_in={"data_folder" : data_folder, "index" : index})
    data_mean           = read_mean_database(data_in={"data_folder" : data_folder})
    data_grid["y_plus"] = data_grid["yd"].reshape(1,-1)*data_mean["utau"].reshape(-1,1)/data_mean["nu"]
    
    data_velfluc        = calc_velfluc(data_in={"data_velocity" : data_velocity, "data_mean" : data_mean, "data_grid" : data_grid})
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Plot velocity in y+=15
    # ---------------------------------------------------------------------------------------------------------------------------------------------

    index_y15 = np.argmin(abs(data_grid["y_plus"][0,:]-15))
    dudy      = data_velfluc["u"][xclip[0]:xclip[1],1,zclip[0]:zclip[1]]-data_velfluc["u"][xclip[0]:xclip[1],0,zclip[0]:zclip[1]]
    u_y15     = data_velfluc["u"][xclip[0]+padding:xclip[1]-padding,index_y15,zclip[0]+padding:zclip[1]-padding]
    data_out  = {"input":u_y15,"output":dudy}
    return data_out