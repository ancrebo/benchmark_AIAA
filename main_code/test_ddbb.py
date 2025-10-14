import numpy as np
from py_bin.py_functions.load_dataset import load_dataset
from py_bin.py_functions.read_json import read_json
import os
import matplotlib.pyplot as plt

config_folder = "../../configurations"
folder_file   = "folders_structure_2025_10_01_train.json"

nn = 10

data_folder   = read_json(data_in={"folder" : config_folder,"file"   : folder_file})
maxin_vec = np.zeros((nn,))
maxout_vec = np.zeros((nn,))
minin_vec = np.zeros((nn,))
minout_vec = np.zeros((nn,))
stdin_vec = np.zeros((nn,))
stdin_vec2 = np.zeros((nn,))
for ind in range(1,nn):
    if not os.path.isfile("../../tfrecords/dataset_"+str(ind)+".tfrecord"):
        continue
    tfrecord_files_vali = ["../../tfrecords/dataset_"+str(ind)+".tfrecord"]
    data_test = iter(load_dataset(data_in={"tfrecord_files":tfrecord_files_vali,"data_folder":data_folder})["parsed_data"].batch(len(tfrecord_files_vali)))
    data_in_t, data_ou_t = next(data_test)
    data_in_t = data_in_t.numpy()
    data_ou_t = data_ou_t.numpy()
    maxdat_in = np.max(data_in_t)
    mindat_in = np.min(data_in_t)
    maxdat_ou = np.max(data_ou_t)
    mindat_ou = np.min(data_ou_t)
    maxin_vec[ind-1] = maxdat_in
    maxout_vec[ind-1] = maxdat_ou
    minin_vec[ind-1] = mindat_in
    minout_vec[ind-1] = mindat_ou
    stdin_vec[ind-1] = np.mean(data_ou_t)+5*np.std(data_ou_t)
    stdin_vec2[ind-1] = np.mean(data_ou_t)-5*np.std(data_ou_t)
    if abs(maxdat_in-1)<1e-1:
        print("max in:"+str(ind))

    if abs(maxdat_ou-1)<1e-1:
        print("max out:"+str(ind))
plt.figure()
plt.plot(maxin_vec)
plt.plot(maxout_vec)
plt.plot(minin_vec)
plt.plot(minout_vec)
plt.savefig("maxfig.png")
