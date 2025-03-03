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
def read_input_output(data_in : dict[dict[str,str,str,str,str,str,np.ndarray,np.ndarray,int],int] = 
                      {"data_folder" : {"folder_database"   : "-",
                                        "file_database"     : "-",
                                        "folder_grid"       : "-",
                                        "file_grid"         : "-",
                                        "statistics_folder" : "-",
                                        "mean_file"         : "-",
                                        "xclip"             : [],
                                        "zclip"             : [],
                                        "padding"           : 0
                                        },
                       "index"       : 0
                       }) -> dict:
    """
    Function to generate the input and output of the model

    Parameters
    ----------
    data_in : dict[dict[str,str,str,str,str,str,np.ndarray,np.ndarray,int],int], optional
        The default is {"data_folder" : {"folder_database"   : "-",
                                         "file_database"     : "-",
                                         "folder_grid"       : "-",
                                         "file_grid"         : "-",
                                         "statistics_folder" : "-",
                                         "mean_file"         : "-",
                                         "xclip"             : [],
                                         "zclip"             : [],
                                         "padding"           : 0
                                         },
                        "index"       : 0
                        }.
        Data:
            - data_folder : dictionary containing the data of the folders:
                + folder_database   : folder storing the data base
                + file_database     : file storing the snapshot to read
                + folder_grid       : folder storing the grid of the data base
                + file_grid         : file storing the grid of the data base
                + statistics_folder : folder for storing the statistics of the problem
                + mean_file         : file storing the mean values to read
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
        data_folder = data_in["data_folder"]
        if not isinstance(data_folder, dict):
            raise TypeError(f"data_folder must be a dictionary, got {type(data_folder).__name__}")
        else:
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read folder_database
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "folder_database" in data_folder:
                folder_database = data_folder["folder_database"]
                if not isinstance(folder_database, str):
                    raise TypeError(f"key folder_database from dictionary data_folder must be a string, got {type(folder_database).__name__}")
            else:
                raise TypeError("key missing in data_folder: folder_database")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read file_database
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "file_database" in data_folder:
                file_database = data_folder["file_database"]
                if not isinstance(file_database, str):
                    raise TypeError(f"key file_database from dictionary data_folder must be a string, got {type(file_database).__name__}")
            else:
                raise TypeError("key missing in data_folder: file_database")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read zpad_database
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "zfill_database" in data_folder:
                zfill_database = data_folder["zfill_database"]
                if not isinstance(zfill_database, int):
                    raise TypeError(f"key zfill_database from dictionary data_folder must be a int, got {type(zfill_database).__name__}")
            else:
                raise TypeError("key missing in data_folder: zfill_database")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read folder_grid
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "folder_grid" in data_folder:
                folder_grid = data_folder["folder_grid"]
                if not isinstance(folder_grid, str):
                    raise TypeError(f"key folder_grid from dictionary data_folder must be a string, got {type(folder_grid).__name__}")
            else:
                raise TypeError("key missing in data_folder: folder_grid")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read file_grid
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "file_grid" in data_folder:
                file_grid = data_folder["file_grid"]
                if not isinstance(file_grid, str):
                    raise TypeError(f"key file_grid from dictionary data_folder must be a string, got {type(file_grid).__name__}")
            else:
                raise TypeError("key missing in data_folder: file_grid")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read statistics_folder
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "statistics_folder" in data_folder:
                statistics_folder = data_folder["statistics_folder"]
                if not isinstance(statistics_folder, str):
                    raise TypeError(f"key statistics_folder from dictionary data_folder must be a string, got {type(statistics_folder).__name__}")
            else:
                raise TypeError("key missing in data_folder: statistics_folder")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read mean_file
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "mean_file" in data_folder:
                mean_file = data_folder["mean_file"]
                if not isinstance(mean_file, str):
                    raise TypeError(f"key mean_file from dictionary data_folder must be a string, got {type(mean_file).__name__}")
            else:
                raise TypeError("key missing in data_folder: mean_file")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read xclip
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "xclip" in data_folder:
                xclip = data_folder["xclip"]
                if not isinstance(xclip, list):
                    raise TypeError(f"key xclip from dictionary data_folder must be a list, got {type(xclip).__name__}")
            else:
                raise TypeError("key missing in data_folder: xclip")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read zclip
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "zclip" in data_folder:
                zclip = data_folder["zclip"]
                if not isinstance(zclip, list):
                    raise TypeError(f"key zclip from dictionary data_folder must be a list, got {type(zclip).__name__}")
            else:
                raise TypeError("key missing in data_folder: zclip")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read padding
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "padding" in data_folder:
                padding = data_folder["padding"]
                if not isinstance(padding, int):
                    raise TypeError(f"key padding from dictionary data_folder must be a int, got {type(padding).__name__}")
            else:
                raise TypeError("key missing in data_folder: padding")
    else:
        raise TypeError("key missing: data_folder")
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read index
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "index" in data_in:
        index = data_in["index"]
        if not isinstance(index, int):
            raise TypeError(f"index must be a integer, got {type(index).__name__}")
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
    
    xclip     = data_folder["xclip"]
    zclip     = data_folder["zclip"]
    index_y15 = np.argmin(abs(data_grid["y_plus"][0,:]-15))
    dudy      = data_velocity["u"][xclip[0]:xclip[1],1,zclip[0]:zclip[1]]-data_velocity["u"][xclip[0]:xclip[1],0,zclip[0]:zclip[1]]
    u_y15     = data_velocity["u"][xclip[0]+padding:xclip[1]-padding,index_y15,zclip[0]+padding:zclip[1]-padding]
    data_out  = {"input":u_y15,"output":dudy}
    return data_out