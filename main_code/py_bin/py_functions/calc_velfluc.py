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
                  "data_grid"      : {"shape_x" : 0,
                                      "shape_y" : 0,
                                      "shape_z" : 0}
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
                                            },
                        "data_grid"      : {"shape_x" : 0,
                                            "shape_y" : 0}
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
            - data_grid     : dictionary containing the grid data:
                + shape_x : shape of the grid in the x direction
                + shape_y : shape of the grid in the y direction

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
        data_velocity = dict(data_in["data_velocity"])
    else:
        raise TypeError("key missing: data_velocity")
        
    if "u" in data_velocity:
        u_vel = np.array(data_velocity["u"])
    else:
        raise TypeError("key missing in data_velocity: u")
    if "v" in data_velocity:
        v_vel = np.array(data_velocity["v"])
    else:
        raise TypeError("key missing in data_velocity: v")
    if "w" in data_velocity:
        w_vel = np.array(data_velocity["w"])
    else:
        raise TypeError("key missing in data_velocity: w")
        
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_mean
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "data_mean" in data_in:
        data_mean = dict(data_in["data_mean"])   
    else:
        raise TypeError("key missing: data_mean")
        
    if "Umean" in data_mean:
        Umean = np.array(data_mean["Umean"])
    else:
        raise TypeError("key missing in data_mean: Umean")
    if "Vmean" in data_mean:
        Vmean = np.array(data_mean["Vmean"])
    else:
        raise TypeError("key missing in data_mean: Vmean")
    if "Wmean" in data_mean:
        Wmean = np.array(data_mean["Wmean"])
    else:
        raise TypeError("key missing in data_mean: Wmean")

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Read data_grid
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    if "data_grid" in data_in:
        data_grid = dict(data_in["data_grid"])
    else:
        raise TypeError("key missing: data_grid")
    
    if "shape_x" in data_grid:
        shape_x = int(data_grid["shape_x"])
    else:
        raise TypeError("key missing in data_grid: shape_x")
    if "shape_y" in data_grid:
        shape_y = int(data_grid["shape_y"])
    else:
        raise TypeError("key missing in data_grid: shape_y")

    
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    # Calculate fluctuations
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    data_out      = {}
    data_out["u"] = u_vel - Umean.reshape(shape_x,shape_y,1)
    data_out["v"] = v_vel - Vmean.reshape(shape_x,shape_y,1)
    data_out["w"] = w_vel - Wmean.reshape(shape_x,shape_y,1)
    return data_out