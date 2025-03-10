# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:06:03 2025

@author: Andres Cremades Botella andrescb@kth.se

File to read the grid of the database
"""
#%%
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions
# -------------------------------------------------------------------------------------------------------------------------------------------------

def read_grid_database(data_in : dict[dict[str,str]] = 
                       {"data_folder" : {"folder_grid" : "-",
                                         "file_grid"   : "-"
                                    }
                        }) -> dict:
    """
    Function to read the database grid

    Parameters
    ----------
    data_in : dict[dict[str,str]], optional
        The default is {"data_folder" : {"folder_grid" : "-",
                                         "file_grid"   : "-"
                                         }
                        }.
        Data:
            - data_folder : dictionary containing the data of the folders:
                + folder_grid : folder storing the grid of the data base
                + file_grid   : file storing the grid of the data base

    Returns
    -------
    dict
        Information of the grid:
            - x       : position of the grid points in the streamwise direction
            - yd      : position of the grid in the wall-normal direction
            - zd      : position of the grid in the spanwise direction
            - shape_x : dimension of the channel streamwise
            - shape_y : dimension of the channel wall-normal
            - shape_z : dimension of the channel spanwise

    """
    import h5py
    import numpy as np
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Check the inputs
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    for key_name,value in data_in.items():
        if key_name not in {"data_folder"}:
            raise TypeError(f"{key_name} is not expected")
            
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_folder
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "data_folder" in data_in:
        data_folder = dict(data_in["data_folder"])
    else:
        raise TypeError("key missing: data_folder")
        
    if "folder_grid" in data_folder:
        folder_grid = str(data_folder["folder_grid"])
    else:
        raise TypeError("key missing in data_folder: folder_grid")
    if "file_grid" in data_folder:
        file_grid   = str(data_folder["file_grid"])
    else:
        raise TypeError("key missing in data_folder: file_grid")
            
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # File to read
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    file_path           = folder_grid + '/' + file_grid
    h5_file             = h5py.File(file_path,'r')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Save in data_out
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_out            = {}
    data_out["x"]       = np.array(h5_file["x"], dtype="float").reshape(-1)
    data_out["yd"]      = np.array(h5_file["yd"], dtype="float").reshape(-1)
    data_out["zd"]      = np.array(h5_file["zd"], dtype="float").reshape(-1)
    data_out["shape_x"] = len(data_out["x"])
    data_out["shape_y"] = len(data_out["yd"])
    data_out["shape_z"] = len(data_out["zd"])
    h5_file.close()
    return data_out