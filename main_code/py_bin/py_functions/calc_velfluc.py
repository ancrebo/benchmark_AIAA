# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:22:04 2025

@author: Andres Cremades Botella andrescb@kth.se

File to calculate the velocity fluctuations
"""
#%%
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the functions
# -------------------------------------------------------------------------------------------------------------------------------------------------
def calc_velfluc(data_in : dict[dict[np.ndarray,np.ndarray,np.ndarray],dict[np.ndarray,np.ndarray,np.ndarray],
                                dict[np.ndarray,np.ndarray,np.ndarray]] = 
                 {"data_velocity" : {"u" : np.array([0]),
                                     "v" : np.array([0]),
                                     "w" : np.array([0])
                                     },
                  "data_mean"      : {"Umean" : np.array([0]),
                                      "Vmean" : np.array([0]),
                                      "Wmean" : np.array([0])
                                      },
                  "data_grid"      : {"shape_x" : np.array([0]),
                                      "shape_y" : np.array([0]),
                                      "shape_z" : np.array([0]),}
                  }) -> dict:
    """
    Function to calculate the velocity fluctuations

    Parameters
    ----------
    data_in : dict[dict[str,str],int], optional
        The default is {"data_velocity" : {"u" : np.array([0]),
                                           "v" : np.array([0]),
                                           "w" : np.array([0])
                                           },
                        "data_mean"      : {"Umean" : np.array([0]),
                                            "Vmean" : np.array([0]),
                                            "Wmean" : np.array([0])
                                            }
                        }.
        Data:
            - data_velocity : dictionary containing the velocity data:
                + u : streamwise velocity
                + v : wall-normal velocty
                + w : spanwise velocity
            - data_mean     : dictionary containing the velocity data:
                + Umean : mean streamwise velocity
                + Vmean : mean wall-normal velocty
                + Wmean : mean spanwise velocity

    Returns
    -------
    dict
        Dictionary storing the velocity:

    """
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Check the inputs
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    for key_name,value in data_in.items():
        if key_name not in {"data_velocity","data_mean","data_grid"}:
            raise TypeError(f"{key_name} is not expected")
                        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_velocity
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "data_velocity" in data_in:
        data_velocity = data_in["data_velocity"]
        if not isinstance(data_velocity, dict):
            raise TypeError(f"data_velocity must be a dictionary, got {type(data_velocity).__name__}")
        else:
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read u
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "u" in data_velocity:
                u_vel = data_velocity["u"]
                if not isinstance(u_vel, np.ndarray):
                    raise TypeError(f"key u from dictionary data_velocity must be a numpy array, got {type(u_vel).__name__}")
            else:
                raise TypeError("key missing in data_velocity: u")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read v
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "v" in data_velocity:
                v_vel = data_velocity["v"]
                if not isinstance(v_vel, np.ndarray):
                    raise TypeError(f"key v from dictionary data_velocity must be a numpy array, got {type(v_vel).__name__}")
            else:
                raise TypeError("key missing in data_velocity: v")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read w
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "w" in data_velocity:
                w_vel = data_velocity["w"]
                if not isinstance(w_vel, np.ndarray):
                    raise TypeError(f"key w from dictionary data_velocity must be a numpy array, got {type(w_vel).__name__}")
            else:
                raise TypeError("key missing in data_velocity: w")
    else:
        raise TypeError("key missing: data_velocity")
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_mean
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "data_mean" in data_in:
        data_mean = data_in["data_mean"]
        if not isinstance(data_mean, dict):
            raise TypeError(f"data_mean must be a dictionary, got {type(data_mean).__name__}")
        else:
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read Umean
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "Umean" in data_mean:
                Umean = data_mean["Umean"]
                if not isinstance(Umean, np.ndarray):
                    raise TypeError(f"key Umean from dictionary data_mean must be a numpy array, got {type(Umean).__name__}")
            else:
                raise TypeError("key missing in data_mean: Umean")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read Vmean
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "Vmean" in data_mean:
                Vmean = data_mean["Vmean"]
                if not isinstance(Vmean, np.ndarray):
                    raise TypeError(f"key Vmean from dictionary data_mean must be a numpy array, got {type(Vmean).__name__}")
            else:
                raise TypeError("key missing in data_mean: Vmean")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read Wmean
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "Wmean" in data_mean:
                Wmean = data_mean["Wmean"]
                if not isinstance(Wmean, np.ndarray):
                    raise TypeError(f"key Wmean from dictionary data_mean must be a numpy array, got {type(Wmean).__name__}")
            else:
                raise TypeError("key missing in data_mean: Wmean")
    else:
        raise TypeError("key missing: data_mean")
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_grid
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "data_grid" in data_in:
        data_grid = data_in["data_grid"]
        if not isinstance(data_grid, dict):
            raise TypeError(f"data_grid must be a dictionary, got {type(data_grid).__name__}")
        else:
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read shape_x
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "shape_x" in data_grid:
                shape_x = data_grid["shape_x"]
                if not isinstance(shape_x, int):
                    raise TypeError(f"key Umean from dictionary data_grid must be a integer, got {type(shape_x).__name__}")
            else:
                raise TypeError("key missing in data_grid: shape_x")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read shape_y
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "shape_y" in data_grid:
                shape_y = data_grid["shape_y"]
                if not isinstance(shape_y, int):
                    raise TypeError(f"key shape_y from dictionary data_grid must be a integer, got {type(shape_y).__name__}")
            else:
                raise TypeError("key missing in data_grid: shape_y")
            # -------------------------------------------------------------------------------------------------------------------------------------
            # Read shape_z
            # -------------------------------------------------------------------------------------------------------------------------------------
            if "shape_z" in data_grid:
                shape_z = data_grid["shape_z"]
                if not isinstance(shape_z, int):
                    raise TypeError(f"key shape_z from dictionary data_grid must be a integer, got {type(shape_z).__name__}")
            else:
                raise TypeError("key missing in data_grid: shape_z")
    else:
        raise TypeError("key missing: data_grid")
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Calculate fluctuations
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_out      = {}
    data_out["u"] = u_vel - Umean.reshape(shape_x,shape_y,1)
    data_out["v"] = v_vel - Vmean.reshape(shape_x,shape_y,1)
    data_out["w"] = w_vel - Wmean.reshape(shape_x,shape_y,1)
    return data_out