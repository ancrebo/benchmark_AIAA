# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:33:26 2025

@author: Andres Cremades Botella andrescb@kth.se
File for training the database BLdns1_3D_t00001.h5 from https://deepblue.lib.umich.edu/data/concern/data_sets/nc580m959?locale=en
This file checks the results of the trained model.
    - input  : shear stress in the wall
    - output : velocity at y+=15
"""

#%%
from py_bin.py_functions.read_json import read_json
from py_bin.py_functions.load_dataset import load_dataset
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import h5py
from py_bin.py_functions.read_input_output import read_input_output
from py_bin.py_functions.read_grid_database import read_grid_database
tf.keras.backend.set_floatx('float64')

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the parameters and folders structure
# -------------------------------------------------------------------------------------------------------------------------------------------------

config_folder = "../../configurations"
folder_file   = "folders_structure_2025_02_19.json"

data_folder   = read_json(data_in={"folder" : config_folder,
                                   "file"   : folder_file})

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Definition of the model
# Define the list of devices used for the training. If the number of GPUs is negative, choose all the 
# available GPUs. In the case of int values of the number of GPUs enter the condition and limit the number 
# of GPUs that are taken.
# -------------------------------------------------------------------------------------------------------------------------------------------------
ngpu    = data_folder["ngpu"]
options = tf.data.Options()
if ngpu > 0:
    dev_list                           = str(np.arange(ngpu).tolist())
    cudadevice                         = dev_list.replace('[','').replace(']','')        
    os.environ["CUDA_VISIBLE_DEVICES"] = cudadevice

physical_devices = tf.config.list_physical_devices('GPU')
available_gpus   = len(physical_devices)
ngpu             = available_gpus
print("-"*100, flush = True)
print('Using TensorFlow version: ',tf.__version__, ', available GPU:', available_gpus,flush = True) 
print("-"*100, flush = True)
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth for GPU: "+str(gpu), flush = True)
    except RuntimeError as ee:
        print(ee, flush = True)
print("-"*100,flush=True)
list_compute   = ['CPU:0']
list_parameter = 'CPU:0'
for ii in np.arange(ngpu, dtype = 'int'):
    list_compute.append('GPU:'+str(ii))

strategy = tf.distribute.MirroredStrategy()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
print("-"*100,flush=True)
print('Number of devices in the strategy: {}'.format(strategy.num_replicas_in_sync), flush = True)
print("List of CPUs in use:", flush = True)
for cpu in tf.config.list_logical_devices('CPU'):
    print(cpu.name, flush = True)
print("List of GPUs in use:", flush = True)
for gpu in tf.config.list_logical_devices('GPU'):
    print(gpu.name, flush = True)        
print("-"*100, flush = True)
tf.keras.backend.set_floatx("float32")

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Create the model
# -------------------------------------------------------------------------------------------------------------------------------------------------
xclip             = data_folder["xclip"]
zclip             = data_folder["zclip"]
padding           = data_folder["padding"]
size_x            = xclip[1]-xclip[0]
size_z            = zclip[1]-zclip[0]
nfil              = data_folder["nfil"]
kernel            = data_folder["kernel"]
stride            = data_folder["stride"]
activ             = data_folder["activation"]
learat            = data_folder["learat"]
batch_size        = data_folder["batch_size"]
epochs            = data_folder["epochs"]
save_every        = data_folder["save_every"]
model_file        = data_folder["model_file"]
statistics_folder = data_folder["statistics_folder"]
train_hist        = data_folder["train_hist"]
train_hist_fig    = data_folder["train_hist_fig"]
file_norm         = data_folder["file_norm"]
field_ini         = data_folder["field_ini_train"]
field_fin         = data_folder["field_fin_train"]

# mixed_precision.set_global_policy('mixed_float16')
with strategy.scope():
    optimizer = RMSprop(learning_rate = learat, momentum = 0.9) 
    model = tf.keras.models.load_model(statistics_folder + '/' + model_file)
    model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = optimizer)
model.summary()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Selection of files for training
# -------------------------------------------------------------------------------------------------------------------------------------------------
field_ini_train = data_folder["field_ini_train"]
field_fin_train = data_folder["field_fin_train"]
field_range     = np.arange(field_ini_train,field_fin_train)
np.random.shuffle(field_range)
tfrecord_folder = data_folder["tfrecords_folder"]
test_ratio      = data_folder["test_ratio"]
prefetch        = data_folder["prefetch"]
all_files       = sorted([os.path.join(tfrecord_folder, ff) for ff in os.listdir(tfrecord_folder) if ff.endswith('.tfrecord')])

# -------------------------------------------------------------------------------------------------------------------------------------------------
# List all tfrecord files
# -------------------------------------------------------------------------------------------------------------------------------------------------
tfrecord_files = [tfrecord_folder+"/dataset_"+str(index)+".tfrecord" for index in field_range]

# -------------------------------------------------------------------------------------------------------------------------------------------------
# read history
# ------------------------------------------------------------------------------------------------------------------------------------------------- 
with open(statistics_folder+'/'+train_hist, 'r') as fread:
    hmat = np.array([[float(ii) for ii in line.split(',')] for line in fread])


# -------------------------------------------------------------------------------------------------------------------------------------------------
# plot folder
# -------------------------------------------------------------------------------------------------------------------------------------------------
fig = plt.figure()
plt.plot(hmat[:,0],hmat[:,1],label="loss")
plt.plot(hmat[:,0],hmat[:,2],label="val_loss")
plt.xlabel("epoch")
plt.ylabel("mse")
plt.yscale("log")
plt.legend()
plt.savefig(statistics_folder + '/' + train_hist_fig)

# -------------------------------------------------------------------------------------------------------------------------------------------------
# calculate error
# -------------------------------------------------------------------------------------------------------------------------------------------------
ff_norm   = h5py.File(statistics_folder + '/' + file_norm,"r")
u_y15_min = np.array(ff_norm["u_y15_min"])
u_y15_max = np.array(ff_norm["u_y15_max"])
dudy_min  = np.array(ff_norm["dudy_min"])
dudy_max  = np.array(ff_norm["dudy_max"])
ff_norm.close()
error     = 0
data_test = iter(load_dataset(data_in={"tfrecord_files":tfrecord_files,
                                       "data_folder":data_folder})["parsed_data"].batch(len(tfrecord_files)))
data_in_t, data_ou_t = next(data_test)
data_in_t = data_in_t.numpy()
data_ou_t = data_ou_t.numpy()
data_pred = model.predict(data_in_t)
error     = np.mean(abs(data_pred-data_ou_t)*(u_y15_max-u_y15_min))/np.max([abs(u_y15_max),abs(u_y15_min)])*100

data_grid    = read_grid_database(data_in={"data_folder" : data_folder})
zgrid, xgrid = np.meshgrid(data_grid["zd"][zclip[0]:zclip[1]],data_grid["x"][xclip[0]:xclip[1]])
fig          = plt.figure()
print(xgrid.shape)
print(zgrid.shape)
print(data_in_t[0,:,:,0].shape)
plt.pcolor(xgrid[padding:-padding,padding:-padding],zgrid[padding:-padding,padding:-padding],
           data_in_t[0,padding:-padding,padding:-padding,0])
plt.title(r"Input data: \frac{du}{dy}")
plt.xlabel("x")
plt.ylabel("z")
plt.savefig(statistics_folder + '/' + "input.png")
fig          = plt.figure()
plt.pcolor(xgrid[padding:-padding,padding:-padding],zgrid[padding:-padding,padding:-padding],
           data_ou_t[0,:,:,0])
plt.title(r"Output data: u")
plt.xlabel("x")
plt.ylabel("z")
plt.savefig(statistics_folder + '/' + "output.png")
fig          = plt.figure()
plt.pcolor(xgrid[padding:-padding,padding:-padding],zgrid[padding:-padding,padding:-padding],data_pred[0,:,:,0])
plt.title(r"Predicted data: u")
plt.xlabel("x")
plt.ylabel("z")
plt.savefig(statistics_folder + '/' + "pred.png") 


print(f"Mean error of the predictions: {error:.2f}%")

