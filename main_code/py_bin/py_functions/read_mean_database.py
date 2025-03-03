# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:09:15 2025

@author: Andres Cremades Botella andrescb@kth.se

File to read the mean values of the database
"""
#%%
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions
# -------------------------------------------------------------------------------------------------------------------------------------------------
def read_mean_database(data_in : dict[dict[str,str]] = 
                           {"data_folder" : {"statistics_folder" : "-",
                                             "mean_file"         : "-"
                                        }
                            }) -> dict:
    """
    Function to read the database velocity

    Parameters
    ----------
    data_in : dict[dict[str,str],int], optional
        The default is {"data_folder" : {"statistics_folder":"-",
                                         "mean_file":"-"
                                         }
                        }.
        Data:
            - data_folder : dictionary containing the data of the folders:
                + statistics_folder : folder for storing the statistics of the problem
                + mean_file         : file storing the mean values to read

    Returns
    -------
    dict
        Dictionary storing the velocity:

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
        data_folder = data_in["data_folder"]
        if not isinstance(data_folder, dict):
            raise TypeError(f"data_folder must be a dictionary, got {type(data_folder).__name__}")
        else:
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
    else:
        raise TypeError("key missing: data_folder")
        
        
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # File to read
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    file_path = statistics_folder+'/'+mean_file
    h5_file   = h5py.File(file_path,'r')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Store in output dictionary
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_out = {}
    data_out["Umean"]   = np.array(h5_file["Umean"])[::2,:]
    data_out["Vmean"]   = np.array(h5_file["Vmean"])[::2,:]
    data_out["Wmean"]   = np.array(h5_file["Wmean"])[::2,:]
    data_out["Urms"]    = np.array(h5_file["urms"])[::2,:]
    data_out["Vrms"]    = np.array(h5_file["vrms"])[::2,:]
    data_out["Wrms"]    = np.array(h5_file["wrms"])[::2,:]
    data_out["Retau"]   = np.array(h5_file["Retau"]).reshape(-1)
    data_out["delta99"] = np.array(h5_file["delta99"]).reshape(-1)
    data_out["utau"]    = np.array(h5_file["utau"]).reshape(-1)
    data_out["nu"]      = data_out["utau"][0]*data_out["delta99"][0]/data_out["Retau"][0]
    h5_file.close()
    return data_out                                  