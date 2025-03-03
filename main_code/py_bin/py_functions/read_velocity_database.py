# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:09:15 2025

@author: Andres Cremades Botella andrescb@kth.se

File to read the velocity database
"""
#%%
# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions
# -------------------------------------------------------------------------------------------------------------------------------------------------
def read_velocity_database(data_in : dict[dict[str,str],int] = 
                           {"data_folder" : {"folder_database" : "-",
                                             "file_database"   : "-"
                                        },
                            "index"       : 0,
                            }) -> dict:
    """
    Function to read the database velocity

    Parameters
    ----------
    data_in : dict[dict[str,str],int], optional
        The default is {"data_folder" : {"folder_database" : "-",
                                         "file_database"   : "-"
                                         },
                        "index"       : 0
                        }.
        Data:
            - data_folder : dictionary containing the data of the folders:
                + folder_database : folder storing the data base
                + file_database   : file storing the snapshot to read
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
    # File to read
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    file_path = folder_database+'/'+file_database
    file_path = file_path.replace('$INDEX$',str(index).zfill(zfill_database))
    h5_file   = h5py.File(file_path,'r')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Store in output dictionary
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_out = {}
    data_out["u"] = np.transpose(np.array(h5_file["u"]),axes=(2,1,0))
    data_out["v"] = np.transpose(np.array(h5_file["v"]),axes=(2,1,0))
    data_out["w"] = np.transpose(np.array(h5_file["w"]),axes=(2,1,0))
    h5_file.close()
    return data_out                                  